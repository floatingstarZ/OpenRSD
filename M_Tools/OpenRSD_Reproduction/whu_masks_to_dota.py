#!/usr/bin/env python3
"""Convert binary/instance WHU-Mix masks to DOTA quadrilateral annotations."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-area", type=float, default=8.0)
    parser.add_argument("--class-name", default="building")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def components(mask):
    values = np.unique(mask)
    if len(values) > 2 and values.max() > 1:
        for value in values:
            if value:
                yield np.uint8(mask == value) * 255
    else:
        binary = np.uint8(mask > 0)
        count, labels = cv2.connectedComponents(binary, connectivity=8)
        for value in range(1, count):
            yield np.uint8(labels == value) * 255


def main():
    args = parse_args()
    if not args.mask_dir.is_dir():
        raise FileNotFoundError(args.mask_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    total_images = total_boxes = 0
    for path in sorted(p for p in args.mask_dir.rglob("*") if p.suffix.lower() in extensions):
        relative = path.relative_to(args.mask_dir).with_suffix(".txt")
        target = args.output_dir / relative
        if target.exists() and not args.overwrite:
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read {path}")
        if mask.ndim == 3:
            mask = np.max(mask, axis=2)
        lines = []
        for component in components(mask):
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < args.min_area:
                    continue
                points = cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1)
                coords = " ".join(f"{value:.2f}" for value in points)
                lines.append(f"{coords} {args.class_name} {args.difficulty}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total_images += 1
        total_boxes += len(lines)
    print(f"images={total_images} boxes={total_boxes} output={args.output_dir}")


if __name__ == "__main__":
    main()
