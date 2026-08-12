#!/usr/bin/env python3
"""Finalize merged OpenRSD pseudo detections as LabelVer5 annotation PKLs."""

import argparse
import pickle
from pathlib import Path

import numpy as np

from format_training_labels import obb_to_poly


def array(value, columns=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    return result.reshape(-1, columns) if columns else result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--clip-threshold", type=float, default=0.24)
    parser.add_argument("--clip-min-area", type=float, default=32 * 32)
    parser.add_argument("--allow-missing-clip-scores", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input_pkl.open("rb") as stream:
        annotations = pickle.load(stream)
    class_map = {}
    if args.class_map:
        with args.class_map.open("rb") as stream:
            class_map = pickle.load(stream)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = pseudo_count = 0
    for ann in annotations:
        image_name = Path(ann["img_path"]).stem
        target = args.output_dir / f"{image_name}.pkl"
        if target.exists() and not args.overwrite:
            continue
        pred = ann["pred_instances"]
        boxes = array(pred["bboxes"], 5).astype(np.float32)
        texts = np.asarray(pred["texts"], dtype=object)
        if len(boxes) != len(texts):
            raise ValueError(f"{image_name}: pseudo boxes/texts length mismatch")
        areas = boxes[:, 2] * boxes[:, 3] if len(boxes) else np.empty(0)
        large = areas > args.clip_min_area
        scores = pred.get("clip_scores")
        if large.any() and scores is None and not args.allow_missing_clip_scores:
            raise KeyError(f"{image_name}: large pseudo boxes require pred_instances.clip_scores")
        keep = np.ones(len(boxes), dtype=bool)
        filtered_boxes = filtered_texts = filtered_scores = []
        if scores is not None:
            scores = array(scores).reshape(-1)
            if len(scores) != len(boxes):
                raise ValueError(f"{image_name}: clip_scores length mismatch")
            reject = large & (scores < args.clip_threshold)
            keep[reject] = False
            filtered_boxes, filtered_texts, filtered_scores = boxes[reject], texts[reject], scores[reject]
        boxes, texts = boxes[keep], texts[keep]
        gt = ann["gt_instances"]
        gt_boxes = array(gt["bboxes"], 5).astype(np.float32)
        gt_texts = np.asarray(gt["texts"], dtype=object)
        all_boxes = np.concatenate((gt_boxes, boxes))
        all_texts = np.concatenate((gt_texts, texts)).tolist()
        all_texts = [class_map.get(name, name) for name in all_texts]
        classes = list(ann.get("cls_list", [])) + list(gt_texts) + list(texts)
        classes = sorted({class_map.get(name, name) for name in classes})
        output = {
            "visual_embeds": None,
            "texts": all_texts,
            "text_embeds": None,
            "polys": obb_to_poly(all_boxes),
            "cls_list": classes,
            "hard_negatives": sorted(set(ann.get("hard_negatives", []))),
            "clip_filtered_boxes": array(filtered_boxes, 5) if len(filtered_boxes) else np.empty((0, 5)),
            "clip_filtered_texts": list(filtered_texts),
            "clip_filtered_scores": array(filtered_scores),
        }
        with target.open("wb") as stream:
            pickle.dump(output, stream, protocol=pickle.HIGHEST_PROTOCOL)
        written += 1
        pseudo_count += len(boxes)
    print(f"written={written} retained_pseudo_boxes={pseudo_count} output={args.output_dir}")


if __name__ == "__main__":
    main()
