"""
download_lightsaber_images.py

This script scrapes the web for Lightsaber images and downloads them.
Compared to lightsaber_datascraper.py, this grabs more images using more restrictive
search terms to limit downloaded images to relevantt, clean data.

Stable version using simple_image_download==0.4

Usage:
    python download_lightsaber_images.py
"""

from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, BaiduImageCrawler
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
    # ---------------- CONFIG ----------------

    CLASS_NAME = "Lightsaber"  
    TARGET_SIZE = (224, 224)

    # Target number of processed images (will stop once we hit this)
    TARGET_COUNT = 1000

    # How many images to *attempt* per search term per engine
    MAX_PER_TERM = 500

    # Movie-focused search terms
    # - Emphasize: movie, film still, screenshot, scene, screencap
    # - Exclude: toy, prop, cosplay, replica, lego, fanart, figure, poster
    SEARCH_TERMS = [
        "star wars lightsaber duel movie still -toy -prop -cosplay -replica -lego -poster -fanart -figure",
        "star wars lightsaber battle film still -toy -prop -cosplay -replica -lego -poster -fanart -figure",
        "star wars lightsaber scene screenshot -toy -prop -cosplay -replica -lego -poster -fanart -figure",
        "star wars lightsaber screencap from movie -toy -prop -cosplay -replica -lego -poster -fanart -figure",
        "luke skywalker lightsaber movie still -toy -prop -cosplay -replica -lego -poster -fanart -figure",
        "darth vader lightsaber movie still -toy -prop -cosplay -replica -lego -poster -fanart -figure",
        "anakin skywalker lightsaber mustafar scene -toy -prop -cosplay -replica -lego -poster -fanart -figure",
        "obi-wan kenobi lightsaber duel film still -toy -prop -cosplay -replica -lego -poster -fanart -figure",
        "rey lightsaber movie still -toy -prop -cosplay -replica -lego -poster -fanart -figure",
        "kylo ren lightsaber scene screenshot -toy -prop -cosplay -replica -lego -poster -fanart -figure",
    ]

    # Engines to use (all from icrawler)
    ENGINES = [
        ("bing",   BingImageCrawler),
        ("google", GoogleImageCrawler),
        ("baidu",  BaiduImageCrawler),
    ]

    # ---------------- PATHS ----------------

    DATASET_ROOT = Path("dataset/processed") / CLASS_NAME
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    RAW_ROOT = Path("simple_images_multi")
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    # -------------- DOWNLOAD STEP -----------

    print("[+] Downloading Lightsaber MOVIE images with icrawler (multiple engines)...")

    for engine_name, EngineClass in ENGINES:
        print(f"[+] Engine: {engine_name}")
        for term in SEARCH_TERMS:
            print(f"    -> {term}")
            subdir = RAW_ROOT / engine_name / term.replace(" ", "_")
            subdir.mkdir(parents=True, exist_ok=True)

            try:
                crawler = EngineClass(storage={"root_dir": str(subdir)})
                crawler.crawl(
                    keyword=term,
                    max_num=MAX_PER_TERM,
                )
            except Exception as e:
                print(f"       [!] Error using {engine_name} for '{term}': {e}")

    print("[+] Finished raw downloads. Cleaning, deduplicating, and resizing...")

    # -------------- CLEAN + RESIZE ----------

    hashes = set()
    count = 0

    for img_path in RAW_ROOT.rglob("*"):
        if count >= TARGET_COUNT:
            break
        if not img_path.is_file():
            continue

        try:
            with Image.open(img_path) as img:
                # dedupe by image bytes
                h = hashlib.md5(img.tobytes()).hexdigest()
                if h in hashes:
                    continue
                hashes.add(h)

                img = img.convert("RGB").resize(TARGET_SIZE)
                out_path = DATASET_ROOT / f"lightsaber_{count:05d}.jpg"
                img.save(out_path, "JPEG", quality=90)

                count += 1
                if count % 50 == 0:
                    print(f"    [*] Processed {count} images...")
        except Exception:
            # skip corrupt/unreadable
            continue

    print(f"[✓] Saved {count} processed Lightsaber images in {DATASET_ROOT.resolve()}")

if __name__ == "__main__":
    main()