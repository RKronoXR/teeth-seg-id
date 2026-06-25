import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tqdm import tqdm

from teeth_seg_id.datasets.ufba425 import UFBA425CocoDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def encode_binary_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/maskrcnn_baseline/epoch_1.pth")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    ann_file = f"data/processed/UFBA-425/coco/instances_{args.split}.json"

    dataset = UFBA425CocoDataset(
        ann_file=ann_file,
        image_dir="data/interim/UFBA-425/numbering_xrays/numbering_xrays",
    )

    if args.max_images:
        dataset.image_ids = dataset.image_ids[:args.max_images]

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=33)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            image = images[0].to(device)
            image_id = int(targets[0]["image_id"].item())

            pred = model([image])[0]

            boxes = pred["boxes"].detach().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()
            masks = pred["masks"].detach().cpu().numpy()

            for box, label, score, mask in zip(boxes, labels, scores, masks):
                x1, y1, x2, y2 = box.tolist()
                binary_mask = mask[0] > 0.5

                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                    "segmentation": encode_binary_mask(binary_mask),
                })

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / f"maskrcnn_{args.split}_predictions.json"
    pred_path.write_text(json.dumps(predictions))

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(str(pred_path))

    for metric in ["bbox", "segm"]:
        evaluator = COCOeval(coco_gt, coco_dt, metric)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    print("predictions:", len(predictions))
    print("saved:", pred_path)


if __name__ == "__main__":
    main()
