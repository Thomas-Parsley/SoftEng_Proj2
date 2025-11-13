"""
download_dalek_images.py
Stable version using simple_image_download==0.4
"""

from simple_image_download import simple_image_download
from pathlib import Path
from PIL import Image
import hashlib

# create downloader
downloader = simple_image_download.simple_image_download()

CLASS_NAME = "Dalek"
TARGET_SIZE = (224, 224)
DATASET_ROOT = Path("dataset/processed") / CLASS_NAME
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

search_terms = [
    "Dalek robot",
    "Dalek prop",
    "Dalek statue",
    "Dalek toy",
    "Dalek 3D model",
    "Dalek miniature",
    "Dalek CGI render"
]

print("[+] Downloading Dalek images...")

for term in search_terms:
    print(f"    -> {term}")
    downloader.download(term, limit=200)

print("[+] Cleaning and resizing images...")

hashes = set()
count = 0
input_dir = Path("dalek_images")

for img_path in input_dir.rglob("*"):
    try:
        with Image.open(img_path) as img:
            h = hashlib.md5(img.tobytes()).hexdigest()
            if h in hashes:
                continue
            hashes.add(h)
            img = img.convert("RGB").resize(TARGET_SIZE)
            img.save(DATASET_ROOT / f"dalek_{count:05d}.jpg", "JPEG", quality=90)
            count += 1
    except Exception:
        pass

print(f"[✓] Saved {count} processed Dalek images in {DATASET_ROOT.resolve()}")