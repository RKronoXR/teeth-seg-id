from pathlib import Path
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

meta = pd.read_csv("data/raw/UFBA-425/metadata.csv")

img_dir = Path("data/interim/UFBA-425/segmentation_xrays/segmentation_xrays")
mask_dir = Path("data/interim/UFBA-425/polygon_masks/polygon_masks")
out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

row = meta.iloc[0]
image_name = row["filename"]
base = Path(image_name).stem

img_path = img_dir / image_name
if not img_path.exists():
    raise FileNotFoundError(img_path)

img = Image.open(img_path).convert("RGB")
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

mask_paths = sorted(mask_dir.rglob(f"{base}_*.ome.tiff"))
print(f"Image: {img_path}")
print(f"Masks found: {len(mask_paths)}")

for mask_path in mask_paths:
    fdi = mask_path.stem.split("_")[-1].replace(".ome", "")

    mask = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask)

    if mask.size != img.size:
        mask = mask.resize(img.size)
        mask_arr = np.array(mask)

    color = (
        random.randint(40, 255),
        random.randint(40, 255),
        random.randint(40, 255),
        110,
    )

    colored = Image.new("RGBA", img.size, color)
    alpha = Image.fromarray((mask_arr > 0).astype(np.uint8) * 110)
    overlay.paste(colored, (0, 0), alpha)

    ys, xs = np.where(mask_arr > 0)
    if len(xs) > 0:
        draw = ImageDraw.Draw(overlay)
        draw.text((int(xs.mean()), int(ys.mean())), str(fdi), fill=(255, 255, 0, 255))

result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
out_path = out_dir / "ufba425_mask_preview.jpg"
result.save(out_path)

print(out_path)
