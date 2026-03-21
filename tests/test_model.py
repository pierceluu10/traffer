import pytest
from PIL import Image
from model.detect import run_inference, COCO_LABELS


def create_test_image(width=640, height=480, color="blue"):
    return Image.new("RGB", (width, height), color=color)


def test_inference_returns_list():
    img = create_test_image()
    results = run_inference(img)
    assert isinstance(results, list)


def test_inference_structure():
    img = create_test_image()
    results = run_inference(img, confidence_threshold=0.01)
    for det in results:
        assert "label" in det
        assert "confidence" in det
        assert "box" in det
        assert isinstance(det["confidence"], float)
        assert 0 <= det["confidence"] <= 1


def test_high_confidence_filters():
    img = create_test_image()
    low = run_inference(img, confidence_threshold=0.1)
    high = run_inference(img, confidence_threshold=0.9)
    assert len(high) <= len(low)


def test_inference_different_sizes():
    for size in [(320, 240), (640, 480), (1920, 1080)]:
        img = Image.new("RGB", size, color="green")
        results = run_inference(img)
        assert isinstance(results, list)


def test_inference_empty_image():
    img = Image.new("RGB", (640, 480), color="black")
    results = run_inference(img)
    assert isinstance(results, list)


def test_inference_returns_coco_labels():
    img = create_test_image()
    results = run_inference(img, confidence_threshold=0.01)
    for det in results:
        assert det["label"] in COCO_LABELS


def test_inference_box_coordinates_valid():
    img = create_test_image()
    results = run_inference(img, confidence_threshold=0.01)
    for det in results:
        box = det["box"]
        assert box["x1"] < box["x2"]
        assert box["y1"] < box["y2"]
