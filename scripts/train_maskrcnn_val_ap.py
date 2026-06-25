import argparse
import contextlib
import csv
import io
import json
from datetime import datetime
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


def evaluate_ap(model, dataset, device, max_images=0, score_threshold=0.0):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model.eval()

    results = []
    image_ids = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            if max_images > 0 and idx >= max_images:
                break

            image = images[0].to(device)
            target = targets[0]
            image_id = int(target["image_id"].item())
            image_ids.append(image_id)

            pred = model([image])[0]

            boxes = pred["boxes"].detach().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()
            masks = pred["masks"].detach().cpu().numpy()[:, 0]

            for box, label, score, mask in zip(boxes, labels, scores, masks):
                if float(score) < score_threshold:
                    continue

                x1, y1, x2, y2 = box.tolist()
                binary_mask = mask >= 0.5

                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                    "segmentation": encode_mask(binary_mask),
                })

    if not results:
        model.train()
        return {"bbox_AP": 0.0, "bbox_AP50": 0.0, "mask_AP": 0.0, "mask_AP50": 0.0}

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(results)

    with contextlib.redirect_stdout(io.StringIO()):
        bbox_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        bbox_eval.params.imgIds = image_ids
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

        mask_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        mask_eval.params.imgIds = image_ids
        mask_eval.evaluate()
        mask_eval.accumulate()
        mask_eval.summarize()

    model.train()

    return {
        "bbox_AP": float(bbox_eval.stats[0]),
        "bbox_AP50": float(bbox_eval.stats[1]),
        "mask_AP": float(mask_eval.stats[0]),
        "mask_AP50": float(mask_eval.stats[1]),
    }


def save_checkpoint(path, epoch, args, model, optimizer, best_metric, best_epoch):
    torch.save({
        "epoch": epoch,
        "run_name": args.run_name,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN and save the best model by validation mask AP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for outputs.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--val-every", type=int, default=5, help="Validate every N epochs.")
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic checkpoint every N epochs. Use 0 to disable.")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop after N epochs without val mask AP improvement. Use 0 to disable.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum val mask AP improvement.")
    parser.add_argument("--max-val-images", type=int, default=0, help="Maximum validation images. Use 0 for all.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Prediction score threshold for validation.")
    args = parser.parse_args()

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lr_text = str(args.lr).replace(".", "p")
        args.run_name = f"maskrcnn_b{args.batch_size}_lr{lr_text}_valap_{timestamp}"

    checkpoint_dir = Path("outputs/checkpoints") / args.run_name
    log_dir = Path("outputs/logs")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{args.run_name}_train_log.csv"
    best_model_path = checkpoint_dir / "best_model.pth"
    best_info_path = checkpoint_dir / "best_model_info.json"

    train_dataset = UFBA425CocoDataset(
        ann_file="data/processed/UFBA-425/coco/instances_train.json",
        image_dir="data/interim/UFBA-425/numbering_xrays/numbering_xrays",
    )

    val_dataset = UFBA425CocoDataset(
        ann_file="data/processed/UFBA-425/coco/instances_val.json",
        image_dir="data/interim/UFBA-425/numbering_xrays/numbering_xrays",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=33)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_metric = -1.0
    best_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", -1.0)
        best_epoch = checkpoint.get("best_epoch", 0)
        print(f"Resuming from epoch {checkpoint['epoch']}")

    write_header = not log_path.exists() or log_path.stat().st_size == 0

    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "step", "loss", "mean_loss",
                "val_bbox_AP", "val_bbox_AP50", "val_mask_AP", "val_mask_AP50",
                "best_val_mask_AP", "best_epoch",
            ],
        )

        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            running_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

            for step, (images, targets) in enumerate(progress, start=1):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                losses = model(images, targets)
                loss = sum(losses.values())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                running_loss += loss_value

                writer.writerow({
                    "epoch": epoch,
                    "step": step,
                    "loss": loss_value,
                    "mean_loss": "",
                    "val_bbox_AP": "",
                    "val_bbox_AP50": "",
                    "val_mask_AP": "",
                    "val_mask_AP50": "",
                    "best_val_mask_AP": best_metric,
                    "best_epoch": best_epoch,
                })

                progress.set_postfix(loss=loss_value)

            mean_loss = running_loss / len(train_loader)
            metrics = {"bbox_AP": "", "bbox_AP50": "", "mask_AP": "", "mask_AP50": ""}

            if args.val_every > 0 and epoch % args.val_every == 0:
                metrics = evaluate_ap(
                    model,
                    val_dataset,
                    device,
                    max_images=args.max_val_images,
                    score_threshold=args.score_threshold,
                )

                val_mask_ap = metrics["mask_AP"]
                improved = val_mask_ap > (best_metric + args.min_delta)

                if improved:
                    best_metric = val_mask_ap
                    best_epoch = epoch

                    save_checkpoint(best_model_path, epoch, args, model, optimizer, best_metric, best_epoch)

                    best_info = {
                        "best_epoch": best_epoch,
                        "best_val_mask_AP": best_metric,
                        "val_bbox_AP": metrics["bbox_AP"],
                        "val_bbox_AP50": metrics["bbox_AP50"],
                        "val_mask_AP": metrics["mask_AP"],
                        "val_mask_AP50": metrics["mask_AP50"],
                        "run_name": args.run_name,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                    }
                    best_info_path.write_text(json.dumps(best_info, indent=2))
                    print(f"New best model at epoch {best_epoch}: val mask AP={best_metric:.4f}")

            if args.save_every > 0 and epoch % args.save_every == 0:
                checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pth"
                save_checkpoint(checkpoint_path, epoch, args, model, optimizer, best_metric, best_epoch)
                print(f"Periodic checkpoint saved: {checkpoint_path}")

            writer.writerow({
                "epoch": epoch,
                "step": "epoch_end",
                "loss": "",
                "mean_loss": mean_loss,
                "val_bbox_AP": metrics["bbox_AP"],
                "val_bbox_AP50": metrics["bbox_AP50"],
                "val_mask_AP": metrics["mask_AP"],
                "val_mask_AP50": metrics["mask_AP50"],
                "best_val_mask_AP": best_metric,
                "best_epoch": best_epoch,
            })
            f.flush()

            print(f"Epoch {epoch} mean loss: {mean_loss:.4f}")
            if metrics["mask_AP"] != "":
                print(
                    f"Val bbox AP={metrics['bbox_AP']:.4f} | "
                    f"Val mask AP={metrics['mask_AP']:.4f} | "
                    f"Best epoch={best_epoch}"
                )

            if args.early_stop_patience > 0 and best_epoch > 0:
                epochs_without_improvement = epoch - best_epoch
                print(f"Early stopping counter: {epochs_without_improvement}/{args.early_stop_patience}")

                if epochs_without_improvement >= args.early_stop_patience:
                    print("Early stopping triggered")
                    break

    print("Training OK")


if __name__ == "__main__":
    main()
