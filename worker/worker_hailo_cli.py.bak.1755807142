#!/usr/bin/env python3
import os, json, subprocess, sys, time, tempfile
import redis
from PIL import Image
import numpy as np

REDIS_URL   = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
HEF_PATH    = os.environ.get("HEF_PATH", "/usr/share/hailo-models/yolov8s_h8.hef")
HAILO_INPUT = os.environ.get("HAILO_INPUT")   # z.B. yolov8s/input_layer1
UPLOAD_DIR  = os.environ.get("UPLOAD_DIR", "/home/pi/uploads")
QUEUE_NAME  = os.environ.get("QUEUE_NAME", "jobs")

IMG_W = 640
IMG_H = 640

def die(msg):
    print(f"[hailo-worker-cli] ERROR: {msg}", flush=True)
    sys.exit(1)

def check_env():
    if not os.path.isfile(HEF_PATH):
        die(f"HEF_PATH existiert nicht: {HEF_PATH}")
    if not HAILO_INPUT:
        die("HAILO_INPUT ist nicht gesetzt (z.B. export HAILO_INPUT='yolov8s/input_layer1')")
    if not os.path.isdir(UPLOAD_DIR):
        die(f"UPLOAD_DIR existiert nicht: {UPLOAD_DIR}")
    print(f"[hailo-worker-cli] HEF: {HEF_PATH}", flush=True)
    print(f"[hailo-worker-cli] Input-Stream: {HAILO_INPUT}", flush=True)

def connect_redis():
    r = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    r.ping()  # fail fast
    print(f"[hailo-worker-cli] Verbunden mit Redis: {REDIS_URL}", flush=True)
    return r

def jpeg_to_bin(jpg_path: str) -> str:
    with Image.open(jpg_path) as im:
        im = im.convert("RGB")
        resample = getattr(Image, "Resampling", Image).BILINEAR
        im = im.resize((IMG_W, IMG_H), resample=resample)
    arr = np.asarray(im, dtype=np.uint8)  # NHWC
    assert arr.shape == (IMG_H, IMG_W, 3), f"Unerwartete Form: {arr.shape}"
    fd, bin_path = tempfile.mkstemp(prefix="hailo_in_", suffix=".bin")
    os.close(fd)
    arr.tofile(bin_path)
    return bin_path

def run_infer(img_path):
    try:
        bin_path = jpeg_to_bin(img_path)
    except Exception as e:
        return False, f"Preprocess fehlgeschlagen: {e}"

    cmd = [
        "hailortcli", "run",
        HEF_PATH,
        "--frames-count", "1",
        "--dont-show-progress",
        "--input-files", f"{HAILO_INPUT}={bin_path}",
    ]
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=90)
        out = cp.stdout.strip()
        return True, out
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        return False, f"hailortcli exit {e.returncode}: {err[-1200:]}"
    except Exception as e:
        return False, f"hailortcli Exception: {e}"
    finally:
        try: os.remove(bin_path)
        except Exception: pass

def main():
    check_env()
    r = connect_redis()
    print(f"[hailo-worker-cli] Bereit – warte auf Jobs in Queue '{QUEUE_NAME}' …", flush=True)

    while True:
        item = r.brpop(QUEUE_NAME, timeout=0)
        if not item:
            continue
        _, raw = item

        try:
            job = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"[hailo-worker-cli] Ungültiges JSON: {e} → {raw[:200]!r}", flush=True)
            continue

        job_id   = job.get("job_id") or job.get("id") or str(int(time.time()))
        img_path = job.get("path") or os.path.join(UPLOAD_DIR, f"{job_id}.jpg")

        print(f"[hailo-worker-cli] Job {job_id}: {img_path}", flush=True)

        if not os.path.isfile(img_path):
            err = {"job_id": job_id, "error": f"Bild nicht gefunden: {img_path}"}
            r.setex(f"error:{job_id}", 3600, json.dumps(err).encode("utf-8"))
            print(f"[hailo-worker-cli] Fehler: {err['error']}", flush=True)
            continue

        ok, out = run_infer(img_path)
        if not ok:
            err = {"job_id": job_id, "error": "Hailo-Run fehlgeschlagen", "details": out}
            r.setex(f"error:{job_id}", 3600, json.dumps(err).encode("utf-8"))
            print(f"[hailo-worker-cli] Fehler: Hailo-Run fehlgeschlagen\n{out}", flush=True)
            continue

        result = {
            "job_id": job_id,
            "items": [{"name": "Demo-Erkennung", "confidence": 0.9}],
            "hailo_stdout": out[-800:]
        }
        r.setex(f"result:{job_id}", 3600, json.dumps(result).encode("utf-8"))
        print(f"[hailo-worker-cli] OK → result:{job_id}", flush=True)

if __name__ == "__main__":
    main()
