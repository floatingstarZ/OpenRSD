import os
from bypy import ByPy

bp = ByPy()

# ============================================================
# 1. 基础 data_root（按你实际使用情况改一个即可）
# ============================================================
DATA_ROOTS = [
    "/data/space2/huangziyue",
]

# ============================================================
# 2. 数据集 image / ann 路径
# ============================================================
datasets = [
    # ---------- Train ----------
    ("data/DOTA2_1024_500/train", "images", None),
    ("data/DIOR_R_dota/train_val", "images", None),
    ("data/FAIR1M_2_800_400/train", "images", None),
    ("data/Spacenet_Merge/train", "images", None),
    ("data/xView_New_800_600/train", "images", None),
    ("data/HRSC2016_DOTA/train", "images", None),
    ("data/GLH-Bridge_1024_200/train", "images", None),
    ("data/FMoW/train", "images", None),
    ("data/WHU_Mix/train", "images", None),
    ("data/ShipRSImageNet_DOTA/train", "images", None),

    # ---------- Val / Test ----------
    ("data/DOTA2_1024_500", "ss_val/images", "ss_val/annfiles"),
]

# ============================================================
# 3. FederatedLabels & SelfLabels
# ============================================================
federated_labels = [
    "Data1_DOTA2", "Data2_DIOR_R", "Data3_FAIR1M",
    "Data5_SpaceNet", "Data6_Xview", "Data7_HRSC2016",
    "Data8_GLH_Bridge", "Data9_FMoW",
    "Data11_WHU_Mix", "Data12_ShipImageNet",
]

self_label_root = "Formatted_SelfLabels_Ver5"

# ============================================================
# 4. Support / Meta / Neg / Model
# ============================================================
extra_files = [
    "./data/7_25_pca_meta_DINOv2_256.pkl",
    "./data/Neg_supports_v2.pkl",
    "./data/normalized_class_dict.pkl",

    "./results/MMR_AD_A08_e_rtm_v2_base_recheck/epoch_36.pth",
    "./results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth",
]

support_files = [
    "./data/DOTA2_1024_500/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/DOTA_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/DIOR_R_dota/train_val/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/FAIR1M_2_800_400/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/TGRS_HRRSD/train_val/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/Spacenet_Merge/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/xView_New_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/HRSC2016_DOTA/train/Step5_3_Prepare_Visual_Text_DINOv2_support_New.pkl",
    "./data/GLH-Bridge_1024_200/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/FMoW/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/WHU_Mix/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
    "./data/ShipRSImageNet_DOTA/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
]

# ============================================================
# 5. 收集所有路径
# ============================================================
paths = []

# dataset images / anns
for root, img, ann in datasets:
    paths.append(os.path.join("./", root, img))
    if ann:
        paths.append(os.path.join("./", root, ann))

# federated & self labels
for dr in DATA_ROOTS:
    for d in federated_labels:
        paths.append(os.path.join(dr, "Formatted_FederatedLabels", d))
        paths.append(os.path.join(dr, self_label_root, d))

# extra
paths.extend(extra_files)
paths.extend(support_files)

paths = sorted(set(paths))

# ============================================================
# 6. 检查 & 上传
# ============================================================
print("\n========== Path Check ==========")
exist, miss = [], []

for p in paths:
    if os.path.exists(p):
        print(f"[OK]   {p}")
        exist.append(p)
    else:
        print(f"[MISS] {p}")
        miss.append(p)

print("\n========== Summary ==========")
print(f"Total   : {len(paths)}")
print(f"Exist   : {len(exist)}")
print(f"Missing : {len(miss)}")

print("\n========== Upload ==========")
for p in exist:
    name = os.path.basename(p.rstrip("/"))
    print(f"Uploading: {p} -> {name}")
    bp.upload(p, name)

print("\n========== Done ==========")
print(bp.list())
