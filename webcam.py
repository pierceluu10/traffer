"""Real-time object detection on webcam feed using Faster R-CNN."""

import cv2
from PIL import Image
from model.detect import run_inference

INFERENCE_WIDTH = 320

# Colors for bounding boxes (BGR format for OpenCV)
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def draw_detections(frame, detections, scale_x, scale_y):
    """Draw bounding boxes and labels on the frame."""
    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        box = det["box"]
        x1 = int(box["x1"] * scale_x)
        y1 = int(box["y1"] * scale_y)
        x2 = int(box["x2"] * scale_x)
        y2 = int(box["y2"] * scale_y)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det['label']} {det['confidence']:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Webcam opened. Press 'q' to quit.")
    print("Loading model on first frame...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # Resize for faster inference
        scale = INFERENCE_WIDTH / orig_w
        small = cv2.resize(frame, (INFERENCE_WIDTH, int(orig_h * scale)))

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        detections = run_inference(image, confidence_threshold=0.5)

        # Scale boxes back to original frame size
        scale_x = orig_w / INFERENCE_WIDTH
        scale_y = orig_h / (orig_h * scale)
        frame = draw_detections(frame, detections, scale_x, scale_y)

        cv2.imshow("Kubera - Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
