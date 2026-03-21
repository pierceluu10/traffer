import time
from typing import List, Dict


def calculate_precision(detections: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.5) -> float:
    if not detections:
        return 0.0
    true_positives = 0
    for det in detections:
        for gt in ground_truth:
            if det["label"] == gt["label"] and calculate_iou(det["box"], gt["box"]) >= iou_threshold:
                true_positives += 1
                break
    return true_positives / len(detections)


def calculate_iou(box1: Dict, box2: Dict) -> float:
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def benchmark_latency(inference_fn, image, runs: int = 10) -> Dict:
    times = []
    for _ in range(runs):
        start = time.time()
        inference_fn(image)
        times.append((time.time() - start) * 1000)

    return {
        "avg_ms": round(sum(times) / len(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 1),
        "runs": runs
    }
