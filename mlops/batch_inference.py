import os
from PIL import Image
from typing import List, Dict
from model.detect import run_inference
import json


def run_batch(image_dir: str, confidence: float = 0.5, output_path: str = None) -> List[Dict]:
    results = []
    supported = (".jpg", ".jpeg", ".png", ".bmp")

    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith(supported):
            continue
        filepath = os.path.join(image_dir, filename)
        image = Image.open(filepath).convert("RGB")
        detections = run_inference(image, confidence_threshold=confidence)
        results.append({
            "file": filename,
            "detections": detections,
            "count": len(detections)
        })

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def summarize_batch(results: List[Dict]) -> Dict:
    total_detections = sum(r["count"] for r in results)
    label_counts = {}
    for r in results:
        for d in r["detections"]:
            label = d["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "total_images": len(results),
        "total_detections": total_detections,
        "avg_detections_per_image": round(total_detections / len(results), 1) if results else 0,
        "label_distribution": label_counts
    }
