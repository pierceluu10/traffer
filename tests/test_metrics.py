import pytest
from mlops.metrics import calculate_iou, calculate_precision, benchmark_latency


def test_iou_perfect_overlap():
    box = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
    assert calculate_iou(box, box) == 1.0


def test_iou_no_overlap():
    box1 = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
    box2 = {"x1": 20, "y1": 20, "x2": 30, "y2": 30}
    assert calculate_iou(box1, box2) == 0.0


def test_iou_partial_overlap():
    box1 = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
    box2 = {"x1": 5, "y1": 5, "x2": 15, "y2": 15}
    iou = calculate_iou(box1, box2)
    assert 0 < iou < 1


def test_iou_symmetry():
    box1 = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
    box2 = {"x1": 5, "y1": 0, "x2": 15, "y2": 10}
    assert calculate_iou(box1, box2) == calculate_iou(box2, box1)


def test_precision_all_correct():
    dets = [{"label": "cat", "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]
    gt = [{"label": "cat", "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]
    assert calculate_precision(dets, gt) == 1.0


def test_precision_none_correct():
    dets = [{"label": "cat", "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]
    gt = [{"label": "dog", "box": {"x1": 50, "y1": 50, "x2": 60, "y2": 60}}]
    assert calculate_precision(dets, gt) == 0.0


def test_precision_empty_detections():
    assert calculate_precision([], [{"label": "cat", "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]) == 0.0


def test_iou_contained_box():
    outer = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}
    inner = {"x1": 5, "y1": 5, "x2": 15, "y2": 15}
    iou = calculate_iou(outer, inner)
    assert 0 < iou < 1
    # Inner area = 100, outer area = 400, intersection = 100, union = 400
    assert abs(iou - 0.25) < 0.01


def test_precision_partial_match():
    dets = [
        {"label": "cat", "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        {"label": "dog", "box": {"x1": 50, "y1": 50, "x2": 60, "y2": 60}},
    ]
    gt = [{"label": "cat", "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]
    precision = calculate_precision(dets, gt)
    assert precision == 0.5


def test_benchmark_latency_returns_all_keys():
    def dummy_fn(img):
        pass
    result = benchmark_latency(dummy_fn, None, runs=5)
    assert "avg_ms" in result
    assert "min_ms" in result
    assert "max_ms" in result
    assert "p95_ms" in result
    assert result["runs"] == 5
