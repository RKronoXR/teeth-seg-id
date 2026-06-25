from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve
import hashlib
import json

ARTICLE_ID = "29827475"
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files"
OUT_DIR = Path("data/raw/UFBA-425")
MANIFEST_PATH = OUT_DIR / "figshare_files_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    req = Request(API_URL, headers={"User-Agent": "teeth-seg-id"})
    with urlopen(req) as response:
        files = json.loads(response.read().decode("utf-8"))

    print(f"Files found: {len(files)}")

    manifest = []

    for item in files:
        name = item["name"]
        url = item["download_url"]
        expected_size = item.get("size")
        out_path = OUT_DIR / name

        if out_path.exists() and expected_size and out_path.stat().st_size == expected_size:
            print(f"Already exists: {name}")
        else:
            print(f"Downloading: {name}")
            urlretrieve(url, out_path)

        manifest.append({
            "name": name,
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "figshare_file_id": item.get("id"),
        })

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest saved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
