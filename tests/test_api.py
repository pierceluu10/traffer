import pytest
from fastapi.testclient import TestClient
from api.main import app
from PIL import Image
import io

client = TestClient(app)


def create_test_image(width=640, height=480, fmt="JPEG"):
    img = Image.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_detect_valid_image():
    img = create_test_image()
    response = client.post("/detect", files={"file": ("test.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "count" in data
    assert "model_version" in data


def test_detect_with_confidence():
    img = create_test_image()
    response = client.post("/detect?confidence=0.9", files={"file": ("test.jpg", img, "image/jpeg")})
    assert response.status_code == 200


def test_detect_invalid_file():
    response = client.post("/detect", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_detect_response_structure():
    img = create_test_image()
    response = client.post("/detect", files={"file": ("test.jpg", img, "image/jpeg")})
    data = response.json()
    assert isinstance(data["detections"], list)
    assert isinstance(data["count"], int)
    for det in data["detections"]:
        assert "label" in det
        assert "confidence" in det
        assert "box" in det
        assert all(k in det["box"] for k in ["x1", "y1", "x2", "y2"])


def test_detect_large_image():
    img = create_test_image(width=1920, height=1080)
    response = client.post("/detect", files={"file": ("large.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data


def test_health_response_time():
    import time
    start = time.time()
    response = client.get("/health")
    elapsed = (time.time() - start) * 1000
    assert response.status_code == 200
    assert elapsed < 500  # Health check should be fast


def test_detect_png_format():
    img = create_test_image(fmt="PNG")
    response = client.post("/detect", files={"file": ("test.png", img, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
