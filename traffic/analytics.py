"""Traffic analytics. Reads events JSONL and generates summary reports."""

import argparse
import json
import os

from traffic.event_consumer import detect_anomaly


def load_events(path: str) -> list[dict]:
    """Load detection events from a JSONL file."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def generate_report(events: list[dict], congestion_threshold: int = 15, congestion_window: int = 3) -> dict:
    """Generate a summary report from detection events."""
    if not events:
        return {
            "total_frames": 0,
            "avg_vehicles_per_frame": 0.0,
            "peak_vehicle_count": 0,
            "total_pedestrians": 0,
            "congestion_events": [],
            "anomaly_count": 0,
        }

    vehicle_counts = [e["vehicle_count"] for e in events]
    pedestrian_counts = [e["pedestrian_count"] for e in events]

    congestion_events = []
    consecutive = 0
    for event in events:
        if event["vehicle_count"] > congestion_threshold:
            consecutive += 1
            if consecutive == congestion_window:
                congestion_events.append(event["timestamp"])
        else:
            consecutive = 0

    anomaly_count = 0
    for event in events:
        for det in event["detections"]:
            if detect_anomaly(det, event["frame_width"], event["frame_height"]):
                anomaly_count += 1

    return {
        "total_frames": len(events),
        "avg_vehicles_per_frame": round(sum(vehicle_counts) / len(vehicle_counts), 1),
        "peak_vehicle_count": max(vehicle_counts),
        "total_pedestrians": sum(pedestrian_counts),
        "congestion_events": congestion_events,
        "anomaly_count": anomaly_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Traffic analytics")
    parser.add_argument("--input", default="traffic/events.jsonl", help="Input JSONL path")
    parser.add_argument("--output", default="traffic/report.json", help="Output report path")
    args = parser.parse_args()

    events = load_events(args.input)
    report = generate_report(events)

    print(json.dumps(report, indent=2))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
