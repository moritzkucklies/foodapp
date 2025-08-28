#!/usr/bin/env python3
import os, json, time
import numpy as np
from PIL import Image
import redis

import hailo_platform as hl
from hailo_platform.pyhailort import _pyhailort as ll  # low-level API

REDIS_URL  = os.environ.get("REDIS_URL", "redis://:password@127.0.0.1:6379/0")
HEF_PATH   = os.environ.get("HEF_PATH", "/usr/share/hailo-models/yolov8s_h8.hef")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/home/pi/uploads")
IMG_SIZE   = (640, 640)  # laut parse-hef

def log(msg): print(f"[hailo-worker] {msg}", flush=True)

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)          # NHWC uint8
    return np.ascontiguousarray(arr)

def to_np_f32(buf):
    try:
        return np.asarray(buf, dtype=np.float32).ravel()
    except Exception:
        try:
            return np.frombuffer(memoryview(buf), dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

class _HLShimCNG:
    """kleiner Wrapper, den hl.InferVStreams erwartet (._configured_network)"""
    def __init__(self, lowlevel_cng):
        self._configured_network = lowlevel_cng

def make_configure_params(vdl, hefl, group_info):
    """Erzeuge ConfigureParams – mehrere Varianten der 4.20-API unterstützt."""
    # 1) bevorzugt: vom VDevice mit group_info
    if hasattr(vdl, "create_configure_params"):
        try:
            return vdl.create_configure_params(hefl, group_info)
        except Exception:
            # 2) ohne group_info
            try:
                return vdl.create_configure_params(hefl)
            except Exception:
                pass
    # 3) statische Fabrik auf ll.ConfigureParams
    CP = getattr(ll, "ConfigureParams", None)
    if CP is not None:
        if hasattr(CP, "create_from_hef"):
            try:
                return CP.create_from_hef(hefl, group_info)
            except Exception:
                pass
        try:
            return CP()
        except Exception:
            pass
    raise RuntimeError("Konnte ConfigureParams nicht erzeugen (unbekannte API-Variante).")

def candidate_keys(hef_hl):
    """Liefere mögliche Keys für die configure()-Map (Group-Name, Network-Name, Varianten)."""
    keys = []
    infos = hef_hl.get_network_groups_infos()
    if infos:
        gi = infos[0]
        # Group-Name
        if hasattr(gi, "name"):
            keys.append(gi.name)                           # z.B. 'yolov8s'
        # Versuche Netzwerk-Namen aus dem Info-Objekt abzuleiten
        for attr in dir(gi):
            if "name" in attr.lower() and attr != "name":
                try:
                    val = getattr(gi, attr)
                    val = val() if callable(val) else val
                    if isinstance(val, str):
                        keys.append(val)
                    elif isinstance(val, (list, tuple)):
                        for v in val:
                            if isinstance(v, str):
                                keys.append(v)
                            elif hasattr(v, "name"):
                                keys.append(getattr(v, "name"))
                except Exception:
                    pass
    # zusätzliche Heuristiken
    more = []
    for k in keys:
        if "/" in k:
            base = k.split("/", 1)[0]
            if base not in keys:
                more.append(base)
    keys.extend(more)
    # eindeutige Reihenfolge bewahren
    seen, ordered = set(), []
    for k in keys:
        if k and k not in seen:
            seen.add(k); ordered.append(k)
    return ordered

def configure_lowlevel(vd_hl, hef_hl):
    """Probiere configure() mit verschiedenen Map-Keys, bis eine echte CNG zurückkommt."""
    vdl  = getattr(vd_hl,  "_vdevice", None)
    hefl = getattr(hef_hl, "_hef",     None)
    if vdl is None or hefl is None:
        raise RuntimeError("Low-level Handle fehlt (vd._vdevice oder hef._hef).")

    infos = hef_hl.get_network_groups_infos()
    if not infos:
        raise RuntimeError("Keine NetworkGroup im HEF gefunden.")
    gi = infos[0]

    cfgp = make_configure_params(vdl, hefl, gi)
    keys = candidate_keys(hef_hl)
    if not keys:
        # Minimale Fallbacks
        keys = [getattr(gi, "name", None), "default", "yolov8s", "yolov8s/yolov8s"]

    log(f"configure()-Keys werden probiert: {keys}")
    last_err = None
    for key in keys:
        try:
            cfg_list = vdl.configure(hefl, {key: cfgp})
            if cfg_list and "ConfiguredNetworkGroup" in type(cfg_list[0]).__name__:
                return cfg_list[0]
            # Manche Builds liefern direkt den CNG (kein Listen-Wrapper)
            if cfg_list and "ConfiguredNetworkGroup" in type(cfg_list).__name__:
                return cfg_list
        except Exception as e:
            last_err = e
            # typischerweise HailoRTStatusException: 61 (NOT_FOUND)
            log(f"configure('{key}') fehlgeschlagen: {e}")
            continue
    raise RuntimeError(f"configure() scheiterte für alle Keys. Letzter Fehler: {last_err}")

def main():
    r = redis.Redis.from_url(REDIS_URL)
    log(f"Verbunden mit Redis: {REDIS_URL}")

    hef = hl.HEF(HEF_PATH)
    log(f"HEF geladen: {HEF_PATH}")

    with hl.VDevice() as vd:
        # Konfiguration → echte low-level ConfiguredNetworkGroup
        cng_ll = configure_lowlevel(vd, hef)
        log(f"ConfiguredNetworkGroup: {type(cng_ll)}")

        # VStream Infos/Namen
        in_infos  = cng_ll.get_input_vstream_infos()
        out_infos = cng_ll.get_output_vstream_infos()
        in_names  = [i.name for i in in_infos]
        out_names = [o.name for o in out_infos]
        log(f"Inputs verfügbar:  {in_names}")
        log(f"Outputs verfügbar: {out_names}")

        # Default-Parameter direkt von der CNG:
        in_params  = cng_ll.make_input_vstream_params()
        out_params = cng_ll.make_output_vstream_params()

        # High-level InferVStreams mit Shim um die low-level CNG
        shim = _HLShimCNG(cng_ll)
        with hl.InferVStreams(shim, in_params, out_params) as infer, cng_ll.activate():
            input_name  = in_names[0]
            output_name = out_names[0]
            invs = infer.get_input_vstream(input_name)
            outvs = infer.get_output_vstream(output_name)

            log("Worker bereit – warte auf Jobs …")
            while True:
                _, job_id = r.brpop("jobs")
                job_id = job_id.decode("utf-8")
                image_path = os.path.join(UPLOAD_DIR, f"{job_id}.jpg")
                log(f"Job {job_id} erhalten: {image_path}")
                try:
                    arr = preprocess_image(image_path)
                    t0 = time.perf_counter()
                    invs.write(arr)
                    out = outvs.read()
                    t1 = time.perf_counter()

                    out_np = to_np_f32(out)
                    result = {
                        "job_id": job_id,
                        "model": "yolov8s_h8",
                        "inference_ms": round((t1 - t0) * 1000.0, 2),
                        "nms_output": {
                            "dtype": "float32",
                            "length": int(out_np.size),
                            "nonzero": int(np.count_nonzero(out_np)),
                            "preview_first20": out_np[:20].round(4).tolist()
                        },
                        "detections": []
                    }
                    r.setex(f"result:{job_id}", 3600, json.dumps(result))
                    r.publish("results", job_id)
                    log(f"Job {job_id} fertig → result:{job_id}")
                except Exception as e:
                    log(f"Fehler in Job {job_id}: {e}")
                    r.setex(f"result:{job_id}", 3600, json.dumps({"job_id": job_id, "error": str(e)}))
                    r.publish("results", job_id)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Beendet (CTRL+C)")
