from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

meta = pd.read_csv("data/raw/UFBA-425/metadata.csv")
img_dir = Path("data/interim/UFBA-425/numbering_xrays/numbering_xrays")
out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

row = meta.iloc[0]
base = Path(row["filename"]).stem

matches = list(img_dir.glob(f"{base}_jpg.rf.*.jpg"))
if not matches:
    raise FileNotFoundError(f"No image found for {base}")

img = Image.open(matches[0]).convert("RGB")
draw = ImageDraw.Draw(img)

fdi_numbers = (
    list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)

for fdi in fdi_numbers:
    x = row.get(f"FDI_{fdi}_x")
    y = row.get(f"FDI_{fdi}_y")
    w = row.get(f"FDI_{fdi}_w")
    h = row.get(f"FDI_{fdi}_h")

    if pd.isna(x) or pd.isna(y) or pd.isna(w) or pd.isna(h):
        continue

    x, y, w, h = int(x), int(y), int(w), int(h)
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    draw.text((x, y), str(fdi), fill="yellow")

out_path = out_dir / "ufba425_bbox_preview.jpg"
img.save(out_path)
print(out_path)
