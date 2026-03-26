"""Stream processor for traffic monitoring. Reads video/images, POSTs to /detect, publishes to Redis."""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import cv2
import httpx
import redis

VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}
PEDESTRIAN_LABELS = {"person"}
TRACKED_LABELS = VEHICLE_LABELS | PEDESTRIAN_LABELS


def filter_detections(detections: list[dict]) -> dict:
    """Filter detections for vehicles and pedestrians."""
    filtered = [d for d in detections if d["label"] in TRACKED_LABELS]
    return {
        "vehicle_count": sum(1 for d in filtered if d["label"] in VEHICLE_LABELS),
        "pedestrian_count": sum(1 for d in filtered if d["label"] in PEDESTRIAN_LABELS),
        "detections": filtered,
    }


def post_frame(client: httpx.Client, api_url: str, frame_bytes: bytes, confidence: float = 0.1) -> list[dict]:
    """POST a frame to the /detect endpoint."""
    resp = client.post(
        f"{api_url}/detect",
        params={"confidence": confidence},
        files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
    )
    resp.raise_for_status()
    return resp.json()["detections"]


def iter_video_frames(video_path: str, interval: float):
    """Yield (frame_number, jpeg_bytes, width, height) from a video file at the given interval."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(fps * interval))
    frame_idx = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            h, w = frame.shape[:2]
            _, buf = cv2.imencode(".jpg", frame)
            yield frame_number, buf.tobytes(), w, h
            frame_number += 1
        frame_idx += 1

    cap.release()


def iter_image_frames(dir_path: str):
    """Yield (frame_number, jpeg_bytes, width, height) from images in a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in extensions
    )
    for i, fname in enumerate(files):
        frame = cv2.imread(os.path.join(dir_path, fname))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        _, buf = cv2.imencode(".jpg", frame)
        yield i, buf.tobytes(), w, h


def process(source: str, api_url: str, redis_url: str, interval: float = 2.0, confidence: float = 0.1):
    """Process a video file or image directory."""
    r = redis.Redis.from_url(redis_url)
    client = httpx.Client(timeout=30.0)

    if os.path.isdir(source):
        frames = iter_image_frames(source)
    else:
        frames = iter_video_frames(source, interval)

    for frame_number, frame_bytes, width, height in frames:
        detections = post_frame(client, api_url, frame_bytes, confidence)
        result = filter_detections(detections)

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_number": frame_number,
            "frame_width": width,
            "frame_height": height,
            **result,
        }

        r.lpush("detections", json.dumps(event))
        print(f"Frame {frame_number}: {result['vehicle_count']} vehicles, {result['pedestrian_count']} pedestrians")

        if os.path.isdir(source):
            time.sleep(interval)

    client.close()


def main():
    parser = argparse.ArgumentParser(description="Traffic stream processor")
    parser.add_argument("--source", required=True, help="Video file or image directory")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Detect API URL")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between frames")
    parser.add_argument("--confidence", type=float, default=0.1, help="Detection confidence threshold")
    args = parser.parse_args()
    process(args.source, args.api_url, args.redis_url, args.interval, args.confidence)


if __name__ == "__main__":
    main()
