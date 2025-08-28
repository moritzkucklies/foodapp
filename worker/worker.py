import time, datetime

def job():
    print(f"[{datetime.datetime.now().isoformat()}] worker heartbeat", flush=True)

if __name__ == "__main__":
    while True:
        job()
        time.sleep(60)
