from pathlib import Path
from collections import Counter
import json
import pandas as pd

root = Path("data/interim/UFBA-425")
meta_path = Path("data/raw/UFBA-425/metadata.csv")

numbering_dir = root / "numbering_xrays/numbering_xrays"
bbox_dir = root / "boundinng_boxes/boundinng_boxes"
seg_dir = root / "segmentation_xrays/segmentation_xrays"
mask_dir = root / "polygon_masks/polygon_masks"

df = pd.read_csv(meta_path)

fdi_order = (
    list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)

class_to_fdi = {i: fdi for i, fdi in enumerate(fdi_order)}

bbox_class_counts = Counter()
bbox_files = sorted(bbox_dir.glob("*.txt"))

for txt in bbox_files:
    for line in txt.read_text().strip().splitlines():
        if line.strip():
            cls = int(line.split()[0])
            bbox_class_counts[class_to_fdi[cls]] += 1

summary = {
    "metadata_rows": len(df),
    "numbering_images": len(list(numbering_dir.glob("*.jpg"))),
    "segmentation_images": len(list(seg_dir.glob("*.jpg"))),
    "bbox_files": len(bbox_files),
    "polygon_masks": len(list(mask_dir.rglob("*.tiff"))),
    "tooth_count_min": int(df["tooth_count"].min()),
    "tooth_count_max": int(df["tooth_count"].max()),
    "tooth_count_mean": float(df["tooth_count"].mean()),
    "bbox_counts_by_fdi": dict(sorted(bbox_class_counts.items())),
}

out = Path("outputs/reports/ufba425_summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
