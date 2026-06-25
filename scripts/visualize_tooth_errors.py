import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tqdm import tqdm

from teeth_seg_id.datasets.ufba425 import UFBA425CocoDataset


FDI_CLASSES = (
    list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)

FDI_TO_LABEL = {fdi: i + 1 for i, fdi in enumerate(FDI_CLASSES)}


def mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return np.logical_and(a, b).sum() / union


def draw_mask(ax, image, mask, title):
    ax.imshow(image)
    ax.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.45)
    ax.set_title(title)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize worst segmentation cases for selected FDI teeth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--teeth", default="24,14,31,41,18", help="Comma-separated FDI teeth.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction score threshold.")
    parser.add_argument("--top-k", type=int, default=10, help="Worst cases to save per tooth.")
    parser.add_argument("--output-dir", default="outputs/figures/tooth_errors", help="Output directory.")
    parser.add_argument("--csv", default="outputs/reports/tooth_error_cases.csv", help="Output CSV report.")
    args = parser.parse_args()

    teeth = [int(x.strip()) for x in args.teeth.split(",") if x.strip()]

    ann_file = f"data/processed/UFBA-425/coco/instances_{args.split}.json"
    image_dir = "data/interim/UFBA-425/numbering_xrays/numbering_xrays"

    dataset = UFBA425CocoDataset(ann_file=ann_file, image_dir=image_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=33)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    cases = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Scanning {args.split}"):
            image_tensor, target = dataset[idx]
            image = image_tensor.permute(1, 2, 0).cpu().numpy()

            prediction = model([image_tensor.to(device)])[0]

            pred_masks = prediction["masks"].squeeze(1).detach().cpu().numpy()
            pred_labels = prediction["labels"].detach().cpu().numpy()
            pred_scores = prediction["scores"].detach().cpu().numpy()

            gt_masks = target["masks"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            image_id = int(target["image_id"].item())

            for tooth in teeth:
                label_id = FDI_TO_LABEL[tooth]

                gt_indices = np.where(gt_labels == label_id)[0]
                pred_indices = np.where((pred_labels == label_id) & (pred_scores >= args.threshold))[0]

                for gt_i in gt_indices:
                    gt_mask = gt_masks[gt_i]

                    best_iou = 0.0
                    best_pred_mask = np.zeros_like(gt_mask)
                    best_score = 0.0

                    for pred_i in pred_indices:
                        pred_mask = pred_masks[pred_i] >= 0.5
                        iou = mask_iou(gt_mask, pred_mask)

                        if iou > best_iou:
                            best_iou = iou
                            best_pred_mask = pred_mask.astype(np.uint8)
                            best_score = float(pred_scores[pred_i])

                    cases.append({
                        "tooth": tooth,
                        "dataset_index": idx,
                        "image_id": image_id,
                        "iou": best_iou,
                        "score": best_score,
                        "image": image,
                        "gt_mask": gt_mask,
                        "pred_mask": best_pred_mask,
                    })

    output_root = Path(args.output_dir) / args.split
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for tooth in teeth:
        tooth_cases = [c for c in cases if c["tooth"] == tooth]
        tooth_cases = sorted(tooth_cases, key=lambda x: x["iou"])[:args.top_k]

        tooth_dir = output_root / f"FDI_{tooth}"
        tooth_dir.mkdir(parents=True, exist_ok=True)

        for rank, case in enumerate(tooth_cases, start=1):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(case["image"])
            axes[0].set_title(f"Image {case['dataset_index']} | FDI {tooth}")
            axes[0].axis("off")

            draw_mask(axes[1], case["image"], case["gt_mask"], "Ground truth")

            draw_mask(
                axes[2],
                case["image"],
                case["pred_mask"],
                f"Prediction | IoU={case['iou']:.3f} | score={case['score']:.2f}",
            )

            fig.tight_layout()

            out_path = tooth_dir / f"worst_{rank:02d}_idx_{case['dataset_index']}_iou_{case['iou']:.3f}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

            rows.append({
                "tooth": tooth,
                "rank": rank,
                "dataset_index": case["dataset_index"],
                "image_id": case["image_id"],
                "iou": case["iou"],
                "score": case["score"],
                "file": str(out_path),
            })

            print(f"Saved {out_path}")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tooth", "rank", "dataset_index", "image_id", "iou", "score", "file"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved report: {csv_path}")


if __name__ == "__main__":
    main()
