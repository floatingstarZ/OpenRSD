import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from collections import Counter

# ============================================================
# BaiduPCS-Go 配置
# ============================================================
# BaiduPCS-Go 工具路径
BAIDUPCS_GO_PATH = "/opt/data/nfs/huangziyue/TOOLS/BaiduPCS-Go-v4.0.0-linux-amd64/BaiduPCS-Go"
# 上传策略：rsync（同步模式，跳过已存在的文件）
UPLOAD_POLICY = "rsync"


def upload_with_baidupcs_go(local_path, remote_path):
    """
    使用 BaiduPCS-Go 上传文件或文件夹

    Args:
        local_path: 本地文件/文件夹路径（绝对路径）
        remote_path: 远程路径（例如：/OpenRSD/data/xxx）
    """
    if not remote_path.startswith("/"):
        remote_path = "/" + remote_path
    # 确保本地路径是绝对路径
    local_abs = os.path.abspath(local_path)

    # 构建命令
    # 格式：BaiduPCS-Go upload <本地路径> <远程路径> --policy rsync
    if not os.path.isdir(local_abs):
        # 如果是文件，remote只保留父目录
        remote_path = os.path.dirname(remote_path)

    # 构建命令字符串
    cmd_str = f'"{BAIDUPCS_GO_PATH}" upload "{local_abs}" "{remote_path}" --policy {UPLOAD_POLICY}'

    print(f"  执行命令: {cmd_str}")
    exit_code = os.system(cmd_str)
    if exit_code == 0:
        print(f"  上传成功: {local_abs} -> {remote_path}")
        return True
    else:
        print(f"  上传失败: {local_abs} -> {remote_path} (退出码: {exit_code})")
        return False


# ============================================================
# 压缩文件夹函数
# ============================================================
def zip_directory(directory_path, zip_path):
    """
    压缩文件夹到 zip 文件

    Args:
        directory_path: 要压缩的文件夹路径
        zip_path: 输出的 zip 文件路径
    """
    print(f"  正在压缩: {directory_path} -> {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # 计算相对路径，保持目录结构（相对于要压缩的文件夹）
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)
    zip_size = os.path.getsize(zip_path)
    size_gb = zip_size / (1024 ** 3)
    print(f"  压缩完成: {zip_path} ({size_gb:.2f} GB)")


# ============================================================
# 1. 基础 data_root（按你实际使用情况改一个即可）
# ============================================================
DATA_ROOTS = [
    "/data/space2/huangziyue",
]

# ============================================================
# 2. 数据集 image / ann 路径
# ============================================================

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
# 5. 收集所有路径 & 对应远程路径
# ============================================================
paths, upload_paths = [], []

# support files（保持目录结构）
for f in support_files:
    paths.append(f)
    upload_paths.append(os.path.join("OpenRSD_Ckpoint_pkl", f[2:]))

# extra files（上传到 OpenRSD 根目录）
for f in extra_files:
    paths.append(f)
    upload_paths.append(os.path.join("OpenRSD_Ckpoint_pkl", f[2:]))

# 去重并排序
combined = sorted(set(zip(paths, upload_paths)), key=lambda x: x[0])

# ============================================================
# 6. 检查路径存在性 & 重复
# ============================================================
exist, miss = [], []
remote_counter = Counter([r for _, r in combined])

print("\n========== Path Check ==========")
for local, remote in combined:
    if remote_counter[remote] > 1:
        print(f"[DUPLICATE] {remote} 被多次上传！")
    if os.path.exists(local):
        print(f"[OK]   {local} -> {remote}")
        exist.append((local, remote))
    else:
        print(f"[MISS] {local} -> {remote}")
        miss.append((local, remote))

print("\n========== Summary ==========")
print(f"Total        : {len(combined)}")
print(f"Exist        : {len(exist)}")
print(f"Missing      : {len(miss)}")
print(f"Duplicate RP : {sum(1 for c in remote_counter.values() if c > 1)}")

# ============================================================
# 7. 用户确认上传
# ============================================================
input("\n按 Enter 确认开始上传...")
temp_dir = '/data/space1/huangziyue/temp_for_upload'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
print("\n========== Upload ==========")
for idx, (local, remote) in enumerate(exist):
    if os.path.isdir(local):
        # 如果是文件夹：压缩 -> 上传 -> 清理
        # 生成压缩文件名：在远程路径后加 .zip
        remote_zip = remote + ".zip"
        # 在temp_dir下建立同样的文件夹结构，将./data/替换为temp_dir/
        # 例如：./data/DOTA2_800_600/train/images -> temp_dir/DOTA2_800_600/train/images.zip
        if local.startswith('./data/'):
            relative_path = local[len('./data/'):]  # 去掉 './data/'
        else:
            # 如果不是以./data/开头，使用相对路径
            relative_path = local.lstrip('./').replace('\\', '/')

        # zip_name和原文件夹保持一致，只不过加了.zip
        zip_name = relative_path + ".zip"
        zip_path = os.path.join(temp_dir, zip_name)

        # 确保zip_path的目录存在
        zip_dir = os.path.dirname(zip_path)
        if zip_dir and not os.path.exists(zip_dir):
            os.makedirs(zip_dir, exist_ok=True)

        print(f"Uploading (folder): {local} -> {remote_zip}")
        # 1. 压缩
        if not os.path.exists(zip_path):
            zip_directory(local, zip_path)
        # 2. 上传
        upload_with_baidupcs_go(zip_path, remote_zip)
        # 3. 立即清理
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
                print(f"  已删除临时文件: {zip_path}")
            except Exception as e:
                print(f"  删除临时文件失败 {zip_path}: {e}")
    else:
        # 如果是文件，直接上传
        print(f"Uploading (file): {local} -> {remote}")
        upload_with_baidupcs_go(local, remote)

print("\n========== Done ==========")
