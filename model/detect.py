import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from torchvision import transforms

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

COCO_LABELS = weights.meta["categories"]

transform = transforms.Compose([
    transforms.ToTensor()
])


def run_inference(image: Image.Image, confidence_threshold: float = 0.5):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    results = []
    for i, score in enumerate(predictions["scores"]):
        if score >= confidence_threshold:
            box = predictions["boxes"][i].tolist()
            label_idx = predictions["labels"][i].item()
            results.append({
                "label": COCO_LABELS[label_idx],
                "confidence": round(score.item(), 3),
                "box": {
                    "x1": round(box[0], 1),
                    "y1": round(box[1], 1),
                    "x2": round(box[2], 1),
                    "y2": round(box[3], 1)
                }
            })
    return results
