import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn

from teeth_seg_id.datasets.ufba425 import UFBA425CocoDataset


def collate_fn(batch):
    return tuple(zip(*batch))


dataset = UFBA425CocoDataset(
    ann_file="data/processed/UFBA-425/coco/instances_train.json",
    image_dir="data/interim/UFBA-425/numbering_xrays/numbering_xrays",
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=33)
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

images, targets = next(iter(loader))
images = [img.to(device) for img in images]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

losses = model(images, targets)
loss = sum(losses.values())

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("device:", device)
print("losses:", {k: float(v.detach().cpu()) for k, v in losses.items()})
print("total_loss:", float(loss.detach().cpu()))
print("Smoke training OK")
