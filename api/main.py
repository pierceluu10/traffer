from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import time
from model.detect import run_inference
from api.schemas import DetectionResponse

app = FastAPI(title="Kubera", description="Object Detection API")

MODEL_VERSION = "1.0.0"


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...), confidence: float = 0.5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    start = time.time()
    detections = run_inference(image, confidence_threshold=confidence)
    latency = round((time.time() - start) * 1000, 1)

    return DetectionResponse(
        detections=detections,
        count=len(detections),
        model_version=MODEL_VERSION
    )
