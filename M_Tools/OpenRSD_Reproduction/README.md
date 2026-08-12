# OpenRSD data preparation and self-training tools

This directory documents and implements the missing public-facing steps reported
in GitHub issues #7--#11.  The scripts avoid machine-specific absolute paths and
accept every input/output location on the command line.

## 1. Format SpaceNet/FAIR1M/WHU training annotations

`format_training_labels.py` is the portable implementation of the historical
`Step6_Format_labels.py`.  It converts the PKL files produced by the DINOv2
feature extraction step into the annotation schema consumed by OpenRSD.

```bash
python M_Tools/OpenRSD_Reproduction/format_training_labels.py \
  --ann-dir data/Spacenet_Merge/Step4_Extract_DINOv2_Embeds_8_3_GT \
  --out-dir data/Spacenet_Merge/Step6_Format_labels \
  --min-area 64
```

The same command applies to FAIR1M and WHU-Mix by changing the two paths.
Compatibility entry points remain available in the individual dataset folders.

## 2. Merge the five SpaceNet cities

The final experiment uses Rio, Vegas, Paris, Shanghai and Khartoum.  It merges
the `train` splits for training and the `val` splits for evaluation.

```bash
python M_Tools/OpenRSD_Reproduction/merge_spacenet.py \
  --source-root /path/to/spacenet/cities \
  --output-root data/Spacenet_Merge
```

Expected counts are 11,773 train images and 3,308 val images.  Run with
`--help` if the city directories use a different naming convention.

## 3. Convert WHU-Mix segmentation masks to DOTA labels

```bash
python M_Tools/OpenRSD_Reproduction/whu_masks_to_dota.py \
  --mask-dir /path/to/WHU-Mix/masks \
  --output-dir data/WHU_Mix/val/labelTxt
```

Each connected component is converted to a minimum-area quadrilateral and
written in DOTA format as class `building`.  Use `--min-area` to discard mask
noise.  The script preserves the source stem, so images and labels can be
matched without a separate list.

## 4. Build self-training labels

The final LabelVer5 pipeline has the following stages:

1. run the unfrozen OpenRSD teacher on every other training dataset;
2. retain detections with detector score >= 0.30;
3. remove detections overlapping existing GT (IoU > 0.000005);
4. rotated NMS at IoU 0.10 and discard boxes smaller than 4x4;
5. for predicted boxes larger than 32x32, compute image/text CLIP similarity;
6. retain those large boxes at CLIP score >= 0.24;
7. merge retained pseudo boxes with GT and write `Formatted_SelfLabels_Ver5`.

The historical internal implementation called this sequence
`M_Tools/SelfTraining_Toolsv3`. The public `build_self_labels.py` replaces its
machine-specific final label-construction step and consumes a merged detection
PKL with the schema documented below:

```bash
python M_Tools/OpenRSD_Reproduction/build_self_labels.py \
  --input-pkl M_Tools/SelfTraining_Toolsv3/Step6_Find_NewLabels5/Data5_SpaceNet.pkl \
  --output-dir data/Formatted_SelfLabels_Ver5/Data5_SpaceNet \
  --class-map data/normalized_class_dict.pkl \
  --clip-threshold 0.24
```

If large pseudo boxes are present, each prediction must contain a `clip_scores`
array aligned with `pred_instances.bboxes`.  This separates expensive CLIP
inference from deterministic label construction and makes the filtering step
reproducible.  `--allow-missing-clip-scores` can be used for debugging only; it
is not equivalent to LabelVer5.

Each input item must contain `img_path`, `gt_instances` (`bboxes`, `texts`),
`pred_instances` (`bboxes`, `texts`, and `clip_scores`), plus optional
`cls_list` and `hard_negatives`. Boxes use `(cx, cy, width, height, angle)` with
the angle in radians. Detector inference and rotated NMS can be produced with
the repository evaluation stack or the MutDet pseudo-label workflow referenced
in issue #8; the deterministic OpenRSD-specific 0.24 filtering and output schema
are implemented here.

## 5. Multi-GPU launcher waiting

`EXP_CONFIG/multi_train_any_gpu.py` waits until each selected GPU uses less
than 0.5 GiB.  `-c` is a command/port counter, not a memory threshold.  If it
prints `Wait`, check `nvidia-smi`; the updated launcher reports memory usage and
the number of GPUs required by the pending task.

## Reproduction data and checkpoint

Processed evaluation data and the final checkpoint are hosted under the public
Baidu Netdisk `OpenRSD/` directory.  The final checkpoint path is:

```text
results/MMR_AD_A12_flex_rtm_v3_1_self_training_Labelver5/epoch_24.pth
```

Model construction also reads small support assets under `data/`, even during
checkpoint-only evaluation:

```text
7_25_pca_meta_DINOv2_256.pkl
Neg_supports_v2.pkl
normalized_class_dict.pkl
*/Step5_3_Prepare_Visual_Text_DINOv2_support*.pkl
```

The final `support_feat_dict` references DOTA-v1/v2, DIOR-R, FAIR1M, HRRSD,
SpaceNet, xView, HRSC2016, GLH-Bridge, FMoW, WHU-Mix, ShipRSImageNet and STAR.
All corresponding precomputed PKLs must retain the paths used in the final
configuration. With these assets, separate DINOv2/SkyCLIP weights are not
needed to evaluate the released checkpoint; they are needed only to regenerate
support features or self-training labels from raw images.
