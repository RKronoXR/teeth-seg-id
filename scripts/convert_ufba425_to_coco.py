from pathlib import Path
import json
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

DATA_ROOT = Path("data/interim/UFBA-425")
SPLIT_ROOT = Path("data/processed/UFBA-425")
OUT_ROOT = Path("data/processed/UFBA-425/coco")

IMG_DIR = DATA_ROOT / "numbering_xrays/numbering_xrays"
MASK_DIR = DATA_ROOT / "polygon_masks/polygon_masks"

FDI_NUMBERS = (
    list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)

FDI_TO_CATEGORY_ID = {fdi: i + 1 for i, fdi in enumerate(FDI_NUMBERS)}

CATEGORIES = [
    {
        "id": FDI_TO_CATEGORY_ID[fdi],
        "name": f"FDI_{fdi}",
        "fdi": fdi,
        "supercategory": "tooth",
    }
    for fdi in FDI_NUMBERS
]


def encode_mask(mask_array):
    binary = np.asfortranarray((mask_array > 0).astype(np.uint8))
    rle = mask_utils.encode(binary)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def convert_split(split_name):
    image_names = (SPLIT_ROOT / f"{split_name}.txt").read_text().splitlines()

    images = []
    annotations = []
    ann_id = 1

    for image_id, image_name in enumerate(image_names, start=1):
        base = Path(image_name).stem
        matches = list(IMG_DIR.glob(f"{base}_jpg.rf.*.jpg"))
        if not matches:
            raise FileNotFoundError(f"No numbering image found for {base}")
        img_path = matches[0]

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        base = Path(image_name).stem

        images.append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height,
        })

        mask_paths = sorted(MASK_DIR.rglob(f"{base}_*.ome.tiff"))

        for mask_path in mask_paths:
            fdi_text = mask_path.name.split("_")[-1].replace(".ome.tiff", "")

            if not fdi_text.isdigit():
                continue

            fdi = int(fdi_text)

            if fdi not in FDI_TO_CATEGORY_ID:
                continue

            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != img.size:
                mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)

            mask_array = np.array(mask_img)
            if not np.any(mask_array > 0):
                continue

            rle = encode_mask(mask_array)
            area = float(mask_utils.area(rle))
            bbox = mask_utils.toBbox(rle).tolist()

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": FDI_TO_CATEGORY_ID[fdi],
                "segmentation": rle,
                "area": area,
                "bbox": [float(x) for x in bbox],
                "iscrowd": 0,
            })

            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"instances_{split_name}.json"
    out_path.write_text(json.dumps(coco))

    print(split_name, "images:", len(images), "annotations:", len(annotations), "->", out_path)


def main():
    for split in ["train", "val", "test"]:
        convert_split(split)


if __name__ == "__main__":
    main()
