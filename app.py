from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
app = FastAPI()
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
@app.get("/health")
def health(): return {"ok": True}
