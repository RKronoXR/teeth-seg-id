from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

split = "train"
ann_path = f"data/processed/UFBA-425/coco/instances_{split}.json"
img_dir = Path("data/interim/UFBA-425/numbering_xrays/numbering_xrays")
out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

coco = COCO(ann_path)
img_id = sorted(coco.imgs.keys())[0]
img_info = coco.imgs[img_id]

base = Path(img_info["file_name"]).stem
img_path = list(img_dir.glob(f"{base}_jpg.rf.*.jpg"))[0]

img = Image.open(img_path).convert("RGB")
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

ann_ids = coco.getAnnIds(imgIds=[img_id])
anns = coco.loadAnns(ann_ids)

for ann in anns:
    mask = mask_utils.decode(ann["segmentation"])
    color = (
        random.randint(40, 255),
        random.randint(40, 255),
        random.randint(40, 255),
        110,
    )
    alpha = Image.fromarray((mask > 0).astype(np.uint8) * 110)
    overlay.paste(Image.new("RGBA", img.size, color), (0, 0), alpha)

    ys, xs = np.where(mask > 0)
    cat = coco.cats[ann["category_id"]]["name"].replace("FDI_", "")
    if len(xs) > 0:
        draw.text((int(xs.mean()), int(ys.mean())), cat, fill=(255, 255, 0, 255))

result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
out_path = out_dir / "coco_sample_preview.jpg"
result.save(out_path)
print(out_path)
