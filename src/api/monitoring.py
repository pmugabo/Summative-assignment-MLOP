from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import prometheus_client as prom
import time

app = FastAPI()

# Create metrics
REQUEST_TIME = prom.Histogram(
    'request_processing_seconds',
    'Time spent processing request',
    ['endpoint']
)
PREDICTION_COUNTER = prom.Counter(
    'predictions_total',
    'Number of predictions made',
    ['class']
)

@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)

@app.get("/metrics")
async def metrics():
    return await Instrumentator().generate_metrics()