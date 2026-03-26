"""Event consumer for traffic monitoring. Reads from Redis, writes JSONL, detects congestion and anomalies."""

import argparse
import json
import logging
import os

import redis

logger = logging.getLogger(__name__)


def detect_congestion(recent_counts: list[int], threshold: int = 15, window: int = 3) -> bool:
    """Return True if vehicle count exceeds threshold for `window` consecutive frames."""
    if len(recent_counts) < window:
        return False
    return all(c > threshold for c in recent_counts[-window:])


def detect_anomaly(detection: dict, frame_width: int, frame_height: int) -> bool:
    """Flag if confidence < 0.3 or bounding box covers > 40% of frame area."""
    if detection["confidence"] < 0.3:
        return True
    box = detection["box"]
    box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
    frame_area = frame_width * frame_height
    if frame_area > 0 and box_area / frame_area > 0.4:
        return True
    return False


def consume(redis_url: str, output_path: str, congestion_threshold: int = 15, congestion_window: int = 3):
    """Consume detection events from Redis and write to JSONL."""
    r = redis.Redis.from_url(redis_url)
    recent_vehicle_counts: list[int] = []

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info("Consumer started, waiting for events...")

    with open(output_path, "a") as f:
        while True:
            _, raw = r.brpop("detections")
            event = json.loads(raw)

            f.write(json.dumps(event) + "\n")
            f.flush()

            recent_vehicle_counts.append(event["vehicle_count"])
            if detect_congestion(recent_vehicle_counts, congestion_threshold, congestion_window):
                logger.warning(
                    "CONGESTION ALERT at %s — %d vehicles for %d consecutive frames",
                    event["timestamp"], event["vehicle_count"], congestion_window,
                )

            for det in event["detections"]:
                if detect_anomaly(det, event["frame_width"], event["frame_height"]):
                    logger.warning(
                        "ANOMALY at %s — %s (confidence=%.2f)",
                        event["timestamp"], det["label"], det["confidence"],
                    )

            logger.info(
                "Frame %d: %d vehicles, %d pedestrians",
                event["frame_number"], event["vehicle_count"], event["pedestrian_count"],
            )


def main():
    parser = argparse.ArgumentParser(description="Traffic event consumer")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--output", default="traffic/events.jsonl", help="Output JSONL path")
    parser.add_argument("--congestion-threshold", type=int, default=15, help="Vehicle count threshold")
    parser.add_argument("--congestion-window", type=int, default=3, help="Consecutive frames for congestion")
    args = parser.parse_args()
    consume(args.redis_url, args.output, args.congestion_threshold, args.congestion_window)


if __name__ == "__main__":
    main()
