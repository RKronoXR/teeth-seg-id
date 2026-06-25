import argparse
import csv
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


def encode_mask(mask):
    mask = np.asfortranarray(mask.astype(np.uint8))
    encoded = mask_utils.encode(mask)
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def summarize_cat(coco_gt, coco_dt, image_ids, cat_id, iou_type):
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = [cat_id]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bbox and mask AP per FDI tooth class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum images to evaluate. Use 0 for all.")
    parser.add_argument("--output-csv", default="outputs/reports/per_tooth_ap_test.csv", help="Output CSV path.")
    args = parser.parse_args()

    ann_file = f"data/processed/UFBA-425/coco/instances_{args.split}.json"
    image_dir = "data/interim/UFBA-425/numbering_xrays/numbering_xrays"

    dataset = UFBA425CocoDataset(ann_file=ann_file, image_dir=image_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=33)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    results = []
    image_ids = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(loader, desc=f"Evaluating {args.split}")):
            if args.max_images > 0 and idx >= args.max_images:
                break

            image = images[0].to(device)
            target = targets[0]
            image_id = int(target["image_id"].item())
            image_ids.append(image_id)

            prediction = model([image])[0]

            boxes = prediction["boxes"].detach().cpu().numpy()
            labels = prediction["labels"].detach().cpu().numpy()
            scores = prediction["scores"].detach().cpu().numpy()
            masks = prediction["masks"].detach().cpu().numpy()[:, 0]

            for box, label, score, mask in zip(boxes, labels, scores, masks):
                x1, y1, x2, y2 = box.tolist()
                binary_mask = mask >= 0.5

                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                    "segmentation": encode_mask(binary_mask),
                })

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(results)

    rows = []
    categories = coco_gt.loadCats(coco_gt.getCatIds())

    for cat in categories:
        cat_id = cat["id"]
        tooth = cat.get("name", str(cat_id))
        n_gt = len(coco_gt.getAnnIds(imgIds=image_ids, catIds=[cat_id]))

        bbox = summarize_cat(coco_gt, coco_dt, image_ids, cat_id, "bbox")
        segm = summarize_cat(coco_gt, coco_dt, image_ids, cat_id, "segm")

        rows.append({
            "FDI": tooth,
            "n_gt": n_gt,
            "bbox_AP": bbox["AP"],
            "bbox_AP50": bbox["AP50"],
            "bbox_AP75": bbox["AP75"],
            "mask_AP": segm["AP"],
            "mask_AP50": segm["AP50"],
            "mask_AP75": segm["AP75"],
        })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output_csv}\n")
    print("FDI,n_gt,bbox_AP,bbox_AP50,mask_AP,mask_AP50")
    for row in rows:
        print(
            f"{row['FDI']},{row['n_gt']},"
            f"{row['bbox_AP']:.3f},{row['bbox_AP50']:.3f},"
            f"{row['mask_AP']:.3f},{row['mask_AP50']:.3f}"
        )


if __name__ == "__main__":
    main()
