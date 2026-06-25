import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

from teeth_seg_id.datasets.ufba425 import UFBA425CocoDataset


FDI_CLASSES = (
    list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)


def label_to_fdi(label):
    return FDI_CLASSES[int(label) - 1]


def overlay(ax, image, masks, labels, boxes=None, scores=None, threshold=0.5, title=""):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    for i, mask in enumerate(masks):
        if scores is not None and scores[i] < threshold:
            continue

        mask = mask > 0.5
        if mask.sum() == 0:
            continue

        colored = np.zeros((*mask.shape, 4), dtype=float)
        colored[..., 0] = (i * 37 % 255) / 255
        colored[..., 1] = (i * 67 % 255) / 255
        colored[..., 2] = (i * 97 % 255) / 255
        colored[..., 3] = mask * 0.35
        ax.imshow(colored)

        if boxes is not None:
            x1, y1, x2, y2 = boxes[i]
            label = str(label_to_fdi(labels[i]))
            if scores is not None:
                label += f" {scores[i]:.2f}"
            ax.text(x1, y1, label, fontsize=6, color="white",
                    bbox=dict(facecolor="black", alpha=0.6, linewidth=0))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Mask R-CNN predictions against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output-dir", default="outputs/figures/predictions", help="Directory for saved images.")
    parser.add_argument("--count", type=int, default=10, help="Number of images to visualize.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction score threshold.")
    args = parser.parse_args()

    ann_file = f"data/processed/UFBA-425/coco/instances_{args.split}.json"
    image_dir = "data/interim/UFBA-425/numbering_xrays/numbering_xrays"

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = UFBA425CocoDataset(ann_file=ann_file, image_dir=image_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=33)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    n = min(args.count, len(dataset))

    with torch.no_grad():
        for idx in range(n):
            image_tensor, target = dataset[idx]
            prediction = model([image_tensor.to(device)])[0]

            image = image_tensor.permute(1, 2, 0).cpu().numpy()

            gt_masks = target["masks"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            gt_boxes = target["boxes"].cpu().numpy()

            pred_masks = prediction["masks"].squeeze(1).cpu().numpy()
            pred_labels = prediction["labels"].cpu().numpy()
            pred_boxes = prediction["boxes"].cpu().numpy()
            pred_scores = prediction["scores"].cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

            overlay(
                axes[0],
                image,
                gt_masks,
                gt_labels,
                boxes=gt_boxes,
                title="Ground truth",
            )

            overlay(
                axes[1],
                image,
                pred_masks,
                pred_labels,
                boxes=pred_boxes,
                scores=pred_scores,
                threshold=args.threshold,
                title=f"Prediction threshold={args.threshold}",
            )

            fig.tight_layout()
            out_path = output_dir / f"{args.split}_{idx:03d}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
