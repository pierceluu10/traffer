from pydantic import BaseModel
from typing import List


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float
    box: BoundingBox


class DetectionResponse(BaseModel):
    detections: List[Detection]
    count: int
    model_version: str
