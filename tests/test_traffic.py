"""Tests for the traffic monitoring system."""

import json
import os
import tempfile

from traffic.stream_processor import filter_detections
from traffic.event_consumer import detect_congestion, detect_anomaly
from traffic.analytics import generate_report, load_events


# --- Stream Processor: filter_detections ---

def test_filter_vehicles_only():
    detections = [
        {"label": "car", "confidence": 0.9, "box": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
        {"label": "truck", "confidence": 0.8, "box": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}},
    ]
    result = filter_detections(detections)
    assert result["vehicle_count"] == 2
    assert result["pedestrian_count"] == 0
    assert len(result["detections"]) == 2


def test_filter_pedestrians_only():
    detections = [
        {"label": "person", "confidence": 0.95, "box": {"x1": 0, "y1": 0, "x2": 50, "y2": 150}},
    ]
    result = filter_detections(detections)
    assert result["vehicle_count"] == 0
    assert result["pedestrian_count"] == 1


def test_filter_mixed_labels():
    detections = [
        {"label": "car", "confidence": 0.9, "box": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
        {"label": "person", "confidence": 0.85, "box": {"x1": 200, "y1": 200, "x2": 250, "y2": 350}},
        {"label": "dog", "confidence": 0.7, "box": {"x1": 300, "y1": 300, "x2": 350, "y2": 350}},
        {"label": "bus", "confidence": 0.6, "box": {"x1": 400, "y1": 0, "x2": 600, "y2": 200}},
    ]
    result = filter_detections(detections)
    assert result["vehicle_count"] == 2
    assert result["pedestrian_count"] == 1
    assert len(result["detections"]) == 3


def test_filter_no_tracked_labels():
    detections = [
        {"label": "dog", "confidence": 0.9, "box": {"x1": 0, "y1": 0, "x2": 50, "y2": 50}},
        {"label": "cat", "confidence": 0.8, "box": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}},
    ]
    result = filter_detections(detections)
    assert result["vehicle_count"] == 0
    assert result["pedestrian_count"] == 0
    assert len(result["detections"]) == 0


# --- Congestion Detection ---

def test_congestion_detected():
    counts = [10, 16, 18, 20]
    assert detect_congestion(counts, threshold=15, window=3) is True


def test_congestion_not_detected_below_threshold():
    counts = [10, 16, 14, 20]
    assert detect_congestion(counts, threshold=15, window=3) is False


def test_congestion_not_detected_too_few_frames():
    counts = [20, 20]
    assert detect_congestion(counts, threshold=15, window=3) is False


def test_congestion_exact_window():
    counts = [16, 16, 16]
    assert detect_congestion(counts, threshold=15, window=3) is True


# --- Anomaly Detection ---

def test_anomaly_low_confidence():
    det = {"label": "car", "confidence": 0.2, "box": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}}
    assert detect_anomaly(det, 1920, 1080) is True


def test_anomaly_large_bounding_box():
    det = {"label": "truck", "confidence": 0.8, "box": {"x1": 0, "y1": 0, "x2": 1200, "y2": 900}}
    assert detect_anomaly(det, 1920, 1080) is True


def test_anomaly_normal_detection():
    det = {"label": "car", "confidence": 0.9, "box": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}
    assert detect_anomaly(det, 1920, 1080) is False


# --- Analytics ---

def test_report_empty_events():
    report = generate_report([])
    assert report["total_frames"] == 0
    assert report["anomaly_count"] == 0


def test_report_basic_counts():
    events = [
        {
            "timestamp": "2024-01-15T10:00:00", "frame_number": 0,
            "frame_width": 1920, "frame_height": 1080,
            "vehicle_count": 5, "pedestrian_count": 2,
            "detections": [
                {"label": "car", "confidence": 0.9, "box": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
            ],
        },
        {
            "timestamp": "2024-01-15T10:00:02", "frame_number": 1,
            "frame_width": 1920, "frame_height": 1080,
            "vehicle_count": 8, "pedestrian_count": 3,
            "detections": [
                {"label": "truck", "confidence": 0.8, "box": {"x1": 200, "y1": 200, "x2": 400, "y2": 400}},
            ],
        },
    ]
    report = generate_report(events)
    assert report["total_frames"] == 2
    assert report["avg_vehicles_per_frame"] == 6.5
    assert report["peak_vehicle_count"] == 8
    assert report["total_pedestrians"] == 5
    assert report["congestion_events"] == []
    assert report["anomaly_count"] == 0


def test_report_with_congestion():
    events = [
        {"timestamp": f"2024-01-15T10:00:{i * 2:02d}", "frame_number": i,
         "frame_width": 1920, "frame_height": 1080,
         "vehicle_count": 20, "pedestrian_count": 1, "detections": []}
        for i in range(5)
    ]
    report = generate_report(events, congestion_threshold=15, congestion_window=3)
    assert len(report["congestion_events"]) == 1


def test_report_with_anomalies():
    events = [
        {
            "timestamp": "2024-01-15T10:00:00", "frame_number": 0,
            "frame_width": 1920, "frame_height": 1080,
            "vehicle_count": 1, "pedestrian_count": 0,
            "detections": [
                {"label": "car", "confidence": 0.15, "box": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
            ],
        },
    ]
    report = generate_report(events)
    assert report["anomaly_count"] == 1


def test_load_events_from_jsonl():
    events = [
        {"timestamp": "2024-01-15T10:00:00", "frame_number": 0, "vehicle_count": 3,
         "pedestrian_count": 1, "frame_width": 1920, "frame_height": 1080, "detections": []},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        path = f.name

    try:
        loaded = load_events(path)
        assert len(loaded) == 1
        assert loaded[0]["frame_number"] == 0
    finally:
        os.unlink(path)
