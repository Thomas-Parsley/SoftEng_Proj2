"""
lightsaber_datascraper.py

This script scrapes the web for Lightsaber images and downloads them.

Stable version using simple_image_download==0.4

Usage:
    python lightsaber_datascraper.py
"""

from simple_image_download import simple_image_download as sid
from pathlib import Path
from PIL import Image
import hashlib

def main():
    """
    Scrapes web for Lightsaber images.

    - Searches based on set search terms
    - Downloads the images found

    Returns:
        None
    """
    # create downloader
    downloader = sid.simple_image_download()

    CLASS_NAME = "LightSaber"
    TARGET_SIZE = (224, 224)
    DATASET_ROOT = Path("lightsaber_images")
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    search_terms = [
        "lightsaber battle",
        "lightsaber duel",
        "lightsaber fight",
        "lightsaber toy",
        "lightsaber prop",
        "lightsaber glowing",
    ]

    print("[+] Downloading Light Saber images...")

    for term in search_terms:
        print(f"    -> {term}")
        downloader.download(term, limit=1)

    print("[+] Cleaning and resizing images...")

    hashes = set()
    count = 0
    input_dir = Path("lightsaber_images")

    for img_path in input_dir.rglob("*"):
        try:
            with Image.open(img_path) as img:
                h = hashlib.md5(img.tobytes()).hexdigest()
                if h in hashes:
                    continue
                hashes.add(h)
                img = img.convert("RGB").resize(TARGET_SIZE)
                img.save(DATASET_ROOT / f"lightSaber_{count:05d}.jpg", "JPEG", quality=90)
                count += 1
        except Exception:
            pass

    print(f"[✓] Saved {count} processed LightSaber images in {DATASET_ROOT.resolve()}")

if __name__ == "__main__":
    main()