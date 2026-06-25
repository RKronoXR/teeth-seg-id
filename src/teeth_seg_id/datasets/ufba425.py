from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


class UFBA425CocoDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, image_dir):
        self.coco = COCO(ann_file)
        self.image_dir = Path(image_dir)
        self.image_ids = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def _find_image(self, file_name):
        base = Path(file_name).stem
        matches = list(self.image_dir.glob(f"{base}_jpg.rf.*.jpg"))
        if not matches:
            raise FileNotFoundError(f"No image found for {base}")
        return matches[0]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.imgs[image_id]

        img_path = self._find_image(img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])
            masks.append(mask_utils.decode(ann["segmentation"]))
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        boxes_xyxy = []
        for x, y, w, h in boxes:
            boxes_xyxy.append([x, y, x + w, y + h])

        image_array = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image_array).permute(2, 0, 1)

        target = {
            "boxes": torch.as_tensor(boxes_xyxy, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.from_numpy(np.stack(masks).astype(np.uint8)),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        return image, target
