#!/usr/bin/env python3
"""Merge the five SpaceNet city splits used by the final OpenRSD experiments."""

import argparse
import shutil
from pathlib import Path


CITY_LAYOUTS = {
    "Rio": ("1_Rio/dstdata/{split}/JPEGImages", "1_Rio/dstdata/{split}/labelTxt"),
    "Vegas": ("fusion_3band/{split}/JPEGImages", "fusion_3band/{split}/labelTxt"),
    "Paris": ("AOI_3_Paris_Train/{split}/JPEGImages_png", "AOI_3_Paris_Train/{split}/labelTxt"),
    "Shanghai": ("AOI_4_Shanghai_Train/{split}/JPEGImages_png", "AOI_4_Shanghai_Train/{split}/labelTxt"),
    "Khartoum": ("AOI_5_Khartoum_Train/{split}/JPEGImages_png", "AOI_5_Khartoum_Train/{split}/labelTxt"),
}
EXPECTED = {"train": 11773, "val": 3308}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-count-check", action="store_true")
    return parser.parse_args()


def find_image(directory: Path, stem: str) -> Path:
    matches = [directory / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]
    for path in matches:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image for {stem} under {directory}")


def main():
    args = parse_args()
    for split in args.splits:
        image_output = args.output_root / split / "images"
        ann_output = args.output_root / split / "annotations"
        image_output.mkdir(parents=True, exist_ok=True)
        ann_output.mkdir(parents=True, exist_ok=True)
        total = 0
        seen = set()
        for city, (image_pattern, ann_pattern) in CITY_LAYOUTS.items():
            image_dir = args.source_root / image_pattern.format(split=split)
            ann_dir = args.source_root / ann_pattern.format(split=split)
            if not image_dir.is_dir() or not ann_dir.is_dir():
                raise FileNotFoundError(f"Missing {city} {split}: {image_dir} or {ann_dir}")
            city_count = 0
            for annotation in sorted(ann_dir.glob("*.txt")):
                if annotation.stem in seen:
                    raise ValueError(f"Duplicate image stem across cities: {annotation.stem}")
                seen.add(annotation.stem)
                image = find_image(image_dir, annotation.stem)
                target_image = image_output / image.name
                target_ann = ann_output / annotation.name
                if (target_image.exists() or target_ann.exists()) and not args.overwrite:
                    raise FileExistsError(f"Target exists: {target_image} or {target_ann}")
                shutil.copy2(image, target_image)
                shutil.copy2(annotation, target_ann)
                city_count += 1
            total += city_count
            print(f"{split} {city}: {city_count}")
        if not args.skip_count_check and total != EXPECTED[split]:
            raise ValueError(f"{split}: expected {EXPECTED[split]} pairs, found {total}")
        print(f"{split} total: {total}")


if __name__ == "__main__":
    main()
