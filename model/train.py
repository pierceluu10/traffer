import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(VOC_CLASSES)}  # 0 is background
NUM_CLASSES = len(VOC_CLASSES) + 1  # 20 classes + background


class VOCDatasetAdapter:
    """Wraps torchvision VOCDetection to produce Faster R-CNN compatible targets."""

    def __init__(self, root: str, year: str = "2012", image_set: str = "train", download: bool = True):
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, annotation = self.dataset[idx]
        img_tensor = self.transform(image)

        objects = annotation["annotation"]["object"]
        if not isinstance(objects, list):
            objects = [objects]

        boxes = []
        labels = []
        for obj in objects:
            label = obj["name"]
            if label not in CLASS_TO_IDX:
                continue
            bbox = obj["bndbox"]
            x1 = float(bbox["xmin"])
            y1 = float(bbox["ymin"])
            x2 = float(bbox["xmax"])
            y2 = float(bbox["ymax"])
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[label])

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }
        return img_tensor, target


def collate_fn(batch):
    """Detection models need list of tensors, not a stacked batch."""
    return tuple(zip(*batch))


def get_model(num_classes: int):
    """Load pretrained Faster R-CNN and replace the classifier head."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = VOCDatasetAdapter(root=args.data_dir, year="2012", image_set="train", download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    print(f"Dataset size: {len(dataset)} images")

    print("Loading pretrained model...")
    model = get_model(NUM_CLASSES)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on Pascal VOC")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--output", type=str, default="model/weights/finetuned_voc.pth", help="Output weights path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    args = parser.parse_args()
    train(args)
