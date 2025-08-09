from fastapi import APIRouter
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

router = APIRouter()

start_time = time.time()

def get_uptime():
    return time.time() - start_time

@router.get("/health")
async def get_health():
    return {"status": "ok"}

@router.get("/metrics")
async def metrics():
    UPTIME = Gauge('uptime_seconds', 'Time the application has been running.')
    UPTIME.set(get_uptime())

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
