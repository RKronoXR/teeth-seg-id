from pathlib import Path
import random
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FDI_NUMBERS = (
    list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)
CATEGORY_ID_TO_FDI = {i + 1: fdi for i, fdi in enumerate(FDI_NUMBERS)}

checkpoint_path = Path("outputs/checkpoints/maskrcnn_baseline/epoch_1.pth")
image_dir = Path("data/interim/UFBA-425/numbering_xrays/numbering_xrays")
val_file = Path("data/processed/UFBA-425/val.txt")
out_dir = Path("outputs/predictions")
out_dir.mkdir(parents=True, exist_ok=True)

image_name = val_file.read_text().splitlines()[0]
base = Path(image_name).stem
img_path = list(image_dir.glob(f"{base}_jpg.rf.*.jpg"))[0]

img = Image.open(img_path).convert("RGB")
img_array = np.array(img, dtype=np.float32) / 255.0
tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(device)

model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=33)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

with torch.no_grad():
    pred = model([tensor])[0]

scores = pred["scores"].detach().cpu()
labels = pred["labels"].detach().cpu()
masks = pred["masks"].detach().cpu()

keep = scores >= 0.30

overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

for score, label, mask in zip(scores[keep], labels[keep], masks[keep]):
    mask_np = mask[0].numpy() > 0.5
    color = (
        random.randint(40, 255),
        random.randint(40, 255),
        random.randint(40, 255),
        110,
    )
    alpha = Image.fromarray(mask_np.astype(np.uint8) * 110)
    overlay.paste(Image.new("RGBA", img.size, color), (0, 0), alpha)

    ys, xs = np.where(mask_np)
    if len(xs) > 0:
        draw.text((int(xs.mean()), int(ys.mean())), str(CATEGORY_ID_TO_FDI[int(label)]), fill=(255, 255, 0, 255))

result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
out_path = out_dir / "maskrcnn_epoch1_prediction.jpg"
result.save(out_path)

print("image:", img_path)
print("predictions kept:", int(keep.sum()))
print("output:", out_path)
