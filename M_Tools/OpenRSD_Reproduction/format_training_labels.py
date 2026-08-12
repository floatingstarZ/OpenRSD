#!/usr/bin/env python3
"""Convert extracted GT features to the OpenRSD Step6 annotation schema."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def obb_to_poly(rboxes: np.ndarray) -> np.ndarray:
    """Convert cx,cy,w,h,angle(rad) boxes to flattened clockwise polygons."""
    boxes = np.asarray(rboxes, dtype=np.float32).reshape(-1, 5)
    if not len(boxes):
        return np.empty((0, 8), dtype=np.float32)
    centers, wh, angles = boxes[:, :2], boxes[:, 2:4], boxes[:, 4]
    corners = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]], np.float32)
    local = corners[None] * wh[:, None]
    cos, sin = np.cos(angles), np.sin(angles)
    rotation = np.stack((cos, -sin, sin, cos), axis=1).reshape(-1, 2, 2)
    return (local @ np.swapaxes(rotation, 1, 2) + centers[:, None]).reshape(-1, 8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-area", type=float, default=64.0)
    parser.add_argument("--ignore-class", default="SAM_Obj")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.ann_dir.is_dir():
        raise FileNotFoundError(args.ann_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for source in sorted(args.ann_dir.glob("*.pkl")):
        target = args.out_dir / source.name
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        with source.open("rb") as stream:
            data = pickle.load(stream)
        missing = {"patch_feats", "rboxes", "cls_names"} - data.keys()
        if missing:
            raise KeyError(f"{source}: missing keys {sorted(missing)}")
        boxes = np.asarray(data["rboxes"], dtype=np.float32).reshape(-1, 5)
        embeds = np.asarray(data["patch_feats"])
        texts = np.asarray(data["cls_names"], dtype=object)
        if not (len(boxes) == len(embeds) == len(texts)):
            raise ValueError(f"{source}: boxes/features/classes have different lengths")
        keep = (texts != args.ignore_class) & (boxes[:, 2] * boxes[:, 3] >= args.min_area)
        output = {
            "visual_embeds": embeds[keep],
            "texts": texts[keep].tolist(),
            "text_embeds": None,
            "polys": obb_to_poly(boxes[keep]),
        }
        with target.open("wb") as stream:
            pickle.dump(output, stream, protocol=pickle.HIGHEST_PROTOCOL)
        written += 1
    print(f"written={written} skipped={skipped} output={args.out_dir}")


if __name__ == "__main__":
    main()

