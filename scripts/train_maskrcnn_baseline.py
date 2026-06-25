import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tqdm import tqdm

from teeth_seg_id.datasets.ufba425 import UFBA425CocoDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = Path("outputs/checkpoints/maskrcnn_baseline")
    log_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset = UFBA425CocoDataset(
        ann_file="data/processed/UFBA-425/coco/instances_train.json",
        image_dir="data/interim/UFBA-425/numbering_xrays/numbering_xrays",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=33,
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = log_dir / "maskrcnn_baseline_train_log.csv"

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "loss"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            running_loss = 0.0

            progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

            for step, (images, targets) in enumerate(progress, start=1):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                losses = model(images, targets)
                loss = sum(losses.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                running_loss += loss_value

                writer.writerow({
                    "epoch": epoch,
                    "step": step,
                    "loss": loss_value,
                })

                progress.set_postfix(loss=loss_value)

            checkpoint_path = out_dir / f"epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

            mean_loss = running_loss / len(loader)
            print(f"Epoch {epoch} mean loss: {mean_loss:.4f}")
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training OK")


if __name__ == "__main__":
    main()
