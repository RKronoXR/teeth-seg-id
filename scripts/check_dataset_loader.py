from teeth_seg_id.datasets.ufba425 import UFBA425CocoDataset

dataset = UFBA425CocoDataset(
    ann_file="data/processed/UFBA-425/coco/instances_train.json",
    image_dir="data/interim/UFBA-425/numbering_xrays/numbering_xrays",
)

image, target = dataset[0]

print("dataset length:", len(dataset))
print("image shape:", tuple(image.shape))
print("boxes:", target["boxes"].shape)
print("labels:", target["labels"].shape)
print("masks:", target["masks"].shape)
print("first labels:", target["labels"][:10].tolist())
print("Dataset loader OK")
