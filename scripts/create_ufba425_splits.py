from pathlib import Path
import random
import json
import pandas as pd

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

meta = pd.read_csv("data/raw/UFBA-425/metadata.csv")
filenames = sorted(meta["filename"].tolist())

random.seed(SEED)
random.shuffle(filenames)

n = len(filenames)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

splits = {
    "train": filenames[:n_train],
    "val": filenames[n_train:n_train + n_val],
    "test": filenames[n_train + n_val:],
}

out_dir = Path("data/processed/UFBA-425")
out_dir.mkdir(parents=True, exist_ok=True)

for split, items in splits.items():
    (out_dir / f"{split}.txt").write_text("\n".join(items) + "\n")

summary = {k: len(v) for k, v in splits.items()}
summary["total"] = sum(summary.values())
summary["seed"] = SEED

(out_dir / "splits.json").write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
