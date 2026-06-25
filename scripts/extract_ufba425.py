from pathlib import Path
import zipfile

raw_dir = Path("data/raw/UFBA-425")
out_dir = Path("data/interim/UFBA-425")
out_dir.mkdir(parents=True, exist_ok=True)

for zip_path in sorted(raw_dir.glob("*.zip")):
    target = out_dir / zip_path.stem
    target.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path.name} -> {target}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)

print("Done")
