import os
import zipfile
import tempfile
import shutil
import subprocess
import glob as glob_module
from pathlib import Path
from collections import Counter

# ============================================================
# BaiduPCS-Go 配置
# ============================================================
# BaiduPCS-Go 工具路径
BAIDUPCS_GO_PATH = "/opt/data/nfs/huangziyue/TOOLS/BaiduPCS-Go-v4.0.0-linux-amd64/BaiduPCS-Go"
# 上传策略：rsync（同步模式，跳过已存在的文件）
UPLOAD_POLICY = "rsync"

# ============================================================
# 分卷压缩配置
# ============================================================
# 分卷大小（单位：MB），默认 2GB = 2048MB
SPLIT_VOLUME_SIZE_MB = 2048
# 分卷压缩阈值（单位：GB），大于此大小才分卷压缩
SPLIT_THRESHOLD_GB = 20


def check_remote_file_exists(remote_path):
    """
    检查云端文件是否存在

    Args:
        remote_path: 远程文件路径

    Returns:
        bool: 文件存在返回True，否则返回False
    """
    if not remote_path.startswith("/"):
        remote_path = "/" + remote_path

    # 获取父目录和文件名
    remote_dir = os.path.dirname(remote_path)
    filename = os.path.basename(remote_path)

    if not remote_dir:
        remote_dir = "/"

    # 使用 ls 命令列出父目录
    cmd = f'"{BAIDUPCS_GO_PATH}" ls "{remote_dir}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        # 父目录不存在
        return False

    # 检查输出中是否包含目标文件名
    output_lines = result.stdout.split('\n')
    for line in output_lines:
        if line.strip().endswith(filename):
            return True
    return False


def check_local_volumes(local_path, temp_dir):
    """
    检查本地是否存在压缩卷文件

    Args:
        local_path: 本地文件夹路径
        temp_dir: 临时目录路径

    Returns:
        list: 本地压缩卷文件列表（如果存在）
    """
    if not os.path.isdir(local_path):
        return []

    # 计算本地压缩文件路径
    if local_path.startswith('./data/'):
        relative_path = local_path[len('./data/'):]
    else:
        relative_path = local_path.lstrip('./').replace('\\', '/')

    zip_base_path = os.path.join(temp_dir, relative_path)
    if zip_base_path.endswith('.zip'):
        zip_base_path = zip_base_path[:-4]

    # 查找本地压缩文件（分卷）
    local_volume_files = []
    for vol_file in sorted(glob_module.glob(f"{zip_base_path}.z*")):
        if os.path.isfile(vol_file):
            local_volume_files.append(vol_file)

    # 添加 .zip 文件（如果有）
    zip_file = f"{zip_base_path}.zip"
    if os.path.exists(zip_file) and os.path.isfile(zip_file):
        local_volume_files.append(zip_file)

    return local_volume_files


def check_remote_has_volumes(remote_dir):
    """
    检查云端目录是否有分卷文件（不依赖本地卷列表）

    Args:
        remote_dir: 远程目录路径

    Returns:
        bool: 如果云端存在分卷文件（.z* 或 .zip），返回True，否则返回False
    """
    if not remote_dir.startswith("/"):
        remote_dir = "/" + remote_dir

    # 使用 ls 命令列出远程目录
    cmd = f'"{BAIDUPCS_GO_PATH}" ls "{remote_dir}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        # 目录不存在
        return False

    output = result.stdout
    # 检查是否存在 .zip 文件或分卷文件 (.z01, .z02, etc.)
    has_zip = ".zip" in output
    has_volumes = ".z" in output  # 匹配 .z01, .z02 等

    return has_zip or has_volumes


def check_remote_volumes(remote_dir, local_volumes):
    """
    检查云端哪些压缩卷已存在

    Args:
        remote_dir: 远程目录路径
        local_volumes: 本地压缩卷文件列表

    Returns:
        tuple: (已存在的卷列表, 缺失的卷列表)
    """
    if not local_volumes:
        return [], []

    if not remote_dir.startswith("/"):
        remote_dir = "/" + remote_dir

    # 使用 ls 命令列出远程目录
    cmd = f'"{BAIDUPCS_GO_PATH}" ls "{remote_dir}"'
    print(f"    检查云端路径: {remote_dir}")
    print(f"    执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        # 目录不存在，所有卷都缺失
        print(f"    ✗ 云端目录不存在或无法访问 (返回码: {result.returncode})")
        if result.stderr:
            print(f"    错误信息: {result.stderr}")
        return [], local_volumes

    output_lines = result.stdout.split('\n')
    # 打印云端列出的所有文件（用于调试）
    cloud_files = [line.strip() for line in output_lines if line.strip()]
    print(f"    云端文件列表 (共 {len(cloud_files)} 项):")
    for cf in cloud_files[:10]:  # 只打印前10个，避免输出过长
        print(f"      - {cf}")
    if len(cloud_files) > 10:
        print(f"      ... (还有 {len(cloud_files) - 10} 项)")

    existing_volumes = []
    missing_volumes = []

    for vol_file in local_volumes:
        vol_basename = os.path.basename(vol_file)
        found = False
        matched_line = None
        for line in output_lines:
            line_stripped = line.strip()
            if line_stripped.endswith(vol_basename):
                found = True
                matched_line = line_stripped
                break
        if found:
            existing_volumes.append(vol_file)
            print(f"    ✓ 找到: {vol_basename} (匹配行: {matched_line})")
        else:
            missing_volumes.append(vol_file)
            print(f"    ✗ 缺失: {vol_basename}")

    print(f"    检查结果: 已存在 {len(existing_volumes)} 个卷, 缺失 {len(missing_volumes)} 个卷")
    return existing_volumes, missing_volumes


def upload_with_baidupcs_go(local_path, remote_path):
    """
    使用 BaiduPCS-Go 上传文件

    Args:
        local_path: 本地文件路径（绝对路径）
        remote_path: 远程完整路径（包括文件名），例如：/OpenRSD/data/xxx/file.zip

    Returns:
        bool: 上传成功返回True，否则返回False
    """
    if not remote_path.startswith("/"):
        remote_path = "/" + remote_path

    # 确保本地路径是绝对路径
    local_abs = os.path.abspath(local_path)

    # BaiduPCS-Go upload 命令的第二个参数是目标目录，不是完整文件路径
    remote_dir = os.path.dirname(remote_path)
    if not remote_dir:
        remote_dir = "/"

    # 构建命令字符串
    cmd_str = f'"{BAIDUPCS_GO_PATH}" upload "{local_abs}" "{remote_dir}" --policy {UPLOAD_POLICY}'

    print(f"  执行命令: {cmd_str}")
    exit_code = os.system(cmd_str)

    if exit_code == 0:
        # 上传命令返回成功，验证云端文件是否真的存在
        print(f"  验证云端文件: {remote_path}")
        if check_remote_file_exists(remote_path):
            print(f"  ✓ 上传成功并验证: {local_abs} -> {remote_path}")
            return True
        else:
            print(f"  ✗ 上传命令成功但云端验证失败: {remote_path}")
            return False
    else:
        print(f"  ✗ 上传失败: {local_abs} -> {remote_path} (退出码: {exit_code})")
        return False


# ============================================================
# 压缩文件夹函数
# ============================================================
def get_directory_size(directory_path):
    """
    计算文件夹总大小

    Args:
        directory_path: 文件夹路径

    Returns:
        float: 文件夹大小（GB）
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 ** 3)


def zip_directory_split(directory_path, zip_base_path, volume_size_mb=None):
    """
    分卷压缩文件夹（使用Linux zip命令）

    Args:
        directory_path: 要压缩的文件夹路径
        zip_base_path: 输出的 zip 文件基础路径（不含.zip后缀）
        volume_size_mb: 每卷大小（MB），如果为None则不分卷

    Returns:
        list: 生成的所有压缩卷文件路径列表
    """
    directory_path = os.path.abspath(directory_path)
    zip_base_path = os.path.abspath(zip_base_path)

    if volume_size_mb and volume_size_mb > 0:
        # 分卷压缩
        print(f"  正在分卷压缩 (每卷{volume_size_mb}MB): {directory_path}")
        if zip_base_path.endswith('.zip'):
            zip_base_path = zip_base_path[:-4]

        cmd = f'cd "{os.path.dirname(directory_path)}" && zip -r -v -s {volume_size_mb}m "{zip_base_path}.zip" "{os.path.basename(directory_path)}"'
        print(f"  执行命令: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  压缩失败: {result.stderr}")
            return []

        # 查找所有生成的分卷文件
        volume_files = []
        for vol_file in sorted(glob_module.glob(f"{zip_base_path}.z*")):
            volume_files.append(vol_file)
        if os.path.exists(f"{zip_base_path}.zip"):
            volume_files.append(f"{zip_base_path}.zip")

        total_size = sum(os.path.getsize(f) for f in volume_files)
        size_gb = total_size / (1024 ** 3)
        print(f"  分卷压缩完成: {len(volume_files)} 个卷, 总大小 {size_gb:.2f} GB")
        for i, vol in enumerate(volume_files, 1):
            vol_size_mb = os.path.getsize(vol) / (1024 ** 2)
            print(f"    卷 {i}: {os.path.basename(vol)} ({vol_size_mb:.2f} MB)")

        return volume_files
    else:
        # 不分卷，使用普通压缩
        print(f"  正在压缩: {directory_path}")
        if not zip_base_path.endswith('.zip'):
            zip_base_path += '.zip'

        cmd = f'cd "{os.path.dirname(directory_path)}" && zip -r -v "{zip_base_path}" "{os.path.basename(directory_path)}"'
        print(f"  执行命令: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  压缩失败: {result.stderr}")
            return []

        zip_size = os.path.getsize(zip_base_path)
        size_gb = zip_size / (1024 ** 3)
        print(f"  压缩完成: {zip_base_path} ({size_gb:.2f} GB)")
        return [zip_base_path]


def zip_directory(directory_path, zip_path):
    """
    压缩文件夹到 zip 文件（保留旧函数用于兼容）

    Args:
        directory_path: 要压缩的文件夹路径
        zip_path: 输出的 zip 文件路径
    """
    print(f"  正在压缩: {directory_path} -> {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
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
dataset_images = [
    ######## Pretraining
    './data/million_aid/test_png',  # D0_MAID
    './data/DOTA2_800_600/train/images',  # D1_DOTA2
    './data/DIOR_R_dota/train_val/images',  # D2_DIOR_R
    './data/FAIR1M_1024_0/train/images',  # D3_FAIR1M
    './data/HRRSD_800_0/train/images',  # D4_HRRSD
    './data/Spacenet_Merge/train/images',  # D5_SpaceNet
    './data/xView_800_600/images',  # D6_Xview
    './data/HRSC2016_DOTA/train/images',  # D7_HRSC2016
    './data/GLH-Bridge_1024_200/train/images',  # D8_GLH_Bridge
    ######## Finetuning
    './data/DOTA2_1024_500/train/images',  # Data1_DOTA2
    './data/DIOR_R_dota/train_val/images',  # Data2_DIOR_R
    './data/FAIR1M_2_800_400/train/images',  # Data3_FAIR1M
    './data/Spacenet_Merge/train/images',  # Data5_SpaceNet
    './data/xView_New_800_600/train/images',  # Data6_Xview
    './data/HRSC2016_DOTA/train/images',  # Data7_HRSC2016
    './data/GLH-Bridge_1024_200/train/images',  # Data8_GLH_Bridge
    './data/FMoW/train/images',  # Data9_FMoW
    './data/WHU_Mix/train/images',  # Data11_WHU_Mix
    './data/ShipRSImageNet_DOTA/train/images',  # Data12_ShipImageNet
    ######## SelfTraining
    ######## Validation
    './data/DOTA2_1024_500/ss_val/images',  # Data1_DOTA2
    './data/DOTA_800_600/val/images',  # Data1_DOTA1
    './data/DIOR_R_dota/test/images',  # Data2_DIOR_R
    './data/DIOR_R_dota/mini_test/images',  # Data2_DIOR_R_mini
    './data/FAIR1M_2_800_400/ss_val/images',  # Data3_FAIR1M
    './data/TGRS_HRRSD/test/images',  # Data4_HRRSD
    './data/Spacenet_Merge_Val/images',  # Data5_SpaceNet
    './data/spacenet/AOI_3_Paris_Train/val/JPEGImages_png',  # Data5_SpaceNet_Paris
    './data/spacenet/AOI_4_Shanghai_Train/val/JPEGImages_png',  # Data5_SpaceNet_Shanghai
    './data/spacenet/AOI_5_Khartoum_Train/val/JPEGImages_png',  # Data5_SpaceNet_Khartoum
    './data/xView_New_800_600/test/images',  # Data6_Xview
    './data/HRSC2016_DOTA/test/images',  # Data7_HRSC2016
    './data/FMoW/test/images',  # Data9_FMoW
    './data/STAR_800_200',
    ######## MINI Test Dataset
    './data/MINI_Test_Dataset/Data1_DOTA2/images',  # MINI_Data1_DOTA2
    './data/MINI_Test_Dataset/Data1_DOTA1/images',  # MINI_Data1_DOTA1
    './data/MINI_Test_Dataset/Data2_DIOR_R/images',  # MINI_Data2_DIOR_R
    './data/MINI_Test_Dataset/Data3_FAIR1M/images',  # MINI_Data3_FAIR1M
    './data/MINI_Test_Dataset/Data4_HRRSD/images',  # MINI_Data4_HRRSD
    './data/MINI_Test_Dataset/Data5_SpaceNet/images',  # MINI_Data5_SpaceNet
    './data/MINI_Test_Dataset/Data6_Xview/images',  # MINI_Data6_Xview
    './data/MINI_Test_Dataset/Data7_HRSC2016/images',  # MINI_Data7_HRSC2016
    './data/MINI_Test_Dataset/Data9_FMoW/images'  # MINI_Data9_FMoW

]

dataset_anns = [
    ######## Pretraining
    './data/million_aid/Step8_Remain_HighResolutions',  # D0_MAID
    './data/DOTA2_800_600/train/Step6_Format_labels',  # D1_DOTA2
    './data/DIOR_R_dota/train_val/Step6_Format_labels',  # D2_DIOR_R
    './data/FAIR1M_1024_0/train/Step6_Format_labels',  # D3_FAIR1M
    './data/HRRSD_800_0/train/Step6_Format_labels',  # D4_HRRSD
    './data/Spacenet_Merge/Step6_Format_labels',  # D5_SpaceNet
    './data/xView_800_600/Step6_Format_labels',  # D6_Xview
    './data/HRSC2016_DOTA/train/Step6_Format_labels',  # D7_HRSC2016
    './data/GLH-Bridge_1024_200/train/Step6_Format_labels',  # D8_GLH_Bridge
    ######## Finetuning
    './data/Formatted_FederatedLabels/Data1_DOTA2',  # Data1_DOTA2
    './data/Formatted_FederatedLabels/Data2_DIOR_R',  # Data2_DIOR_R
    './data/Formatted_FederatedLabels/Data3_FAIR1M',  # Data3_FAIR1M
    './data/Formatted_FederatedLabels/Data5_SpaceNet',  # Data5_SpaceNet
    './data/Formatted_FederatedLabels/Data6_Xview',  # Data6_Xview
    './data/Formatted_FederatedLabels/Data7_HRSC2016',  # Data7_HRSC2016
    './data/Formatted_FederatedLabels/Data8_GLH_Bridge',  # Data8_GLH_Bridge
    './data/Formatted_FederatedLabels/Data9_FMoW',  # Data9_FMoW
    './data/Formatted_FederatedLabels/Data11_WHU_Mix',  # Data11_WHU_Mix
    './data/Formatted_FederatedLabels/Data12_ShipImageNet',  # Data12_ShipImageNet
    ######## SelfTraining
    './data/Formatted_SelfLabels_Ver5/Data1_DOTA2',  # Data1_DOTA2
    './data/Formatted_SelfLabels_Ver5/Data2_DIOR_R',  # Data2_DIOR_R
    './data/Formatted_SelfLabels_Ver5/Data3_FAIR1M',  # Data3_FAIR1M
    './data/Formatted_SelfLabels_Ver5/Data5_SpaceNet',  # Data5_SpaceNet
    './data/Formatted_SelfLabels_Ver5/Data6_Xview',  # Data6_Xview
    './data/Formatted_SelfLabels_Ver5/Data7_HRSC2016',  # Data7_HRSC2016
    './data/Formatted_SelfLabels_Ver5/Data8_GLH_Bridge',  # Data8_GLH_Bridge
    './data/Formatted_SelfLabels_Ver5/Data9_FMoW',  # Data9_FMoW
    './data/Formatted_SelfLabels_Ver5/Data11_WHU_Mix',  # Data11_WHU_Mix
    './data/Formatted_SelfLabels_Ver5/Data12_ShipImageNet',  # Data12_ShipImageNet
    ######## Validation
    './data/DOTA2_1024_500/ss_val/annfiles',  # Data1_DOTA2
    './data/DOTA_800_600/val/labelTxt',  # Data1_DOTA1
    './data/DIOR_R_dota/test/labelTxt',  # Data2_DIOR_R
    './data/DIOR_R_dota/mini_test/labelTxt',  # Data2_DIOR_R_mini
    './data/FAIR1M_2_800_400/ss_val/annfiles',  # Data3_FAIR1M
    './data/TGRS_HRRSD/test/Step1_Trans_HBB2OBB',  # Data4_HRRSD
    './data/Spacenet_Merge_Val/annotations',  # Data5_SpaceNet
    './data/spacenet/AOI_3_Paris_Train/val/labelTxt',  # Data5_SpaceNet_Paris
    './data/spacenet/AOI_4_Shanghai_Train/val/labelTxt',  # Data5_SpaceNet_Shanghai
    './data/spacenet/AOI_5_Khartoum_Train/val/labelTxt',  # Data5_SpaceNet_Khartoum
    './data/xView_New_800_600/test/annfiles',  # Data6_Xview
    './data/HRSC2016_DOTA/test/labelTxt',  # Data7_HRSC2016
    './data/FMoW/test/labelTxt',  # Data9_FMoW
    ######## MINI Test Dataset
    './data/MINI_Test_Dataset/Data1_DOTA2/annotations',  # MINI_Data1_DOTA2
    './data/MINI_Test_Dataset/Data1_DOTA1/annotations',  # MINI_Data1_DOTA1
    './data/MINI_Test_Dataset/Data2_DIOR_R/annotations',  # MINI_Data2_DIOR_R
    './data/MINI_Test_Dataset/Data3_FAIR1M/annotations',  # MINI_Data3_FAIR1M
    './data/MINI_Test_Dataset/Data4_HRRSD/annotations',  # MINI_Data4_HRRSD
    './data/MINI_Test_Dataset/Data5_SpaceNet/annotations',  # MINI_Data5_SpaceNet
    './data/MINI_Test_Dataset/Data6_Xview/annotations',  # MINI_Data6_Xview
    './data/MINI_Test_Dataset/Data7_HRSC2016/annotations',  # MINI_Data7_HRSC2016
    './data/MINI_Test_Dataset/Data9_FMoW/annotations'  # MINI_Data9_FMoW
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
    "./results/MMR_AD_A10_flex_rtm_v3_1_formal_with_hbb/epoch_8.pth",
    "./results/MMR_AD_A12_flex_rtm_v3_1_self_training_Labelver5/epoch_24.pth",
    "./results/MMR_AD_A12_flex_rtm_v3_1_maid_self_training/epoch_24.pth",
    "./results/MMR_AD_A10_flex_rtm_v3_1_formal_simple/epoch_7.pth",
    "./results/MMR_AD_A13_Hin_rtm_v2_NearestMem/epoch_9.pth",
    "./results/MMR_AD_A10_Gen_rtm_v3_0_pretrain_wo_slot/epoch_24.pth",
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
# 数据集
for f in dataset_images:
    paths.append(f)
    upload_paths.append(os.path.join("OpenRSD", f[2:]))
for f in dataset_anns:
    paths.append(f)
    upload_paths.append(os.path.join("OpenRSD", f[2:]))

# support files（保持目录结构）
for f in support_files:
    paths.append(f)
    upload_paths.append(os.path.join("OpenRSD", f[2:]))

# extra files（上传到 OpenRSD 根目录）
for f in extra_files:
    paths.append(f)
    upload_paths.append(os.path.join("OpenRSD", f[2:]))

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
# 7. 检查云端文件，找出需要上传的
# ============================================================
print("\n========== 检查云端文件 ==========")
already_uploaded = []  # 已上传成功的
need_upload = []       # 需要上传的

temp_dir = '/data/space1/huangziyue/temp_for_upload'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

for idx, (local, remote) in enumerate(exist):
    print(f"[{idx+1}/{len(exist)}] 检查: {local}")

    if os.path.isdir(local):
        # 文件夹：先检查远程是否存在*.zip文件
        remote_zip = remote + ".zip"
        if check_remote_file_exists(remote_zip):
            print(f"  ✓ 云端已存在压缩文件: {remote_zip}")
            already_uploaded.append((local, remote))
        else:
            # 不存在*.zip，检查分卷
            print(f"  ✗ 云端不存在压缩文件: {remote_zip}")
            # 检查本地是否存在压缩卷
            local_volumes = check_local_volumes(local, temp_dir)
            if local_volumes:
                # 3.2 本地存在分卷，开始检查云端分卷是否齐全
                print(f"    发现本地压缩卷 ({len(local_volumes)} 个)")
                # 检查云端哪些卷已存在
                existing_volumes, missing_volumes = check_remote_volumes(remote, local_volumes)
                if missing_volumes:
                    print(f"    云端缺失 {len(missing_volumes)} 个卷，需要续传")
                    need_upload.append({
                        'local': local,
                        'remote': remote,
                        'local_volumes': local_volumes,
                        'missing_volumes': missing_volumes,
                        'is_resume': True
                    })
                else:
                    print(f"    ✓ 云端所有卷已存在")
                    already_uploaded.append((local, remote))
            else:
                # 3.1 本地不存在分卷，检查云端是否有分卷
                print(f"    本地不存在压缩卷")
                if check_remote_has_volumes(remote):
                    # 云端有分卷，表明上传成功
                    print(f"    ✓ 云端存在分卷文件，表明上传成功")
                    already_uploaded.append((local, remote))
                else:
                    # 云端没分卷，甚至没文件夹，说明该文件夹没上传
                    print(f"    ✗ 云端不存在分卷文件，需要完整上传")
                    need_upload.append({
                        'local': local,
                        'remote': remote,
                        'is_resume': False
                    })
    else:
        # 文件：直接检查云端是否存在
        if check_remote_file_exists(remote):
            print(f"  ✓ 云端已存在: {remote}")
            already_uploaded.append((local, remote))
        else:
            print(f"  ✗ 云端不存在: {remote}")
            need_upload.append((local, remote))

print("\n========== 检查结果 ==========")
print(f"总数: {len(exist)}")
print(f"已上传: {len(already_uploaded)}")
print(f"待上传: {len(need_upload)}")

if not need_upload:
    print("\n所有文件都已上传成功！")
    exit(0)

# ============================================================
# 8. 用户确认上传
# ============================================================
input("\n按 Enter 确认开始上传待上传的文件...")

print("\n========== Upload ==========")
failed_uploads = []

for idx, item in enumerate(need_upload):
    # 处理不同的数据结构
    if isinstance(item, dict):
        local = item['local']
        remote = item['remote']
        is_resume = item.get('is_resume', False)
        local_volumes = item.get('local_volumes', [])
        missing_volumes = item.get('missing_volumes', [])
    else:
        local, remote = item
        is_resume = False
        local_volumes = []
        missing_volumes = []

    print(f"\n[{idx+1}/{len(need_upload)}] 处理: {local}")

    if os.path.isdir(local):
        # 文件夹处理
        if local.startswith('./data/'):
            relative_path = local[len('./data/'):]
        else:
            relative_path = local.lstrip('./').replace('\\', '/')

        zip_base_name = relative_path
        zip_base_path = os.path.join(temp_dir, zip_base_name)

        # 确保目录存在
        zip_dir = os.path.dirname(zip_base_path)
        if zip_dir and not os.path.exists(zip_dir):
            os.makedirs(zip_dir, exist_ok=True)

        if is_resume and local_volumes:
            # 续传：使用已有的压缩文件，只上传缺失的卷
            print(f"  续传模式: 使用已有压缩文件，上传 {len(missing_volumes)} 个缺失卷")
            volume_files = local_volumes
            volume_files_to_upload = missing_volumes
        else:
            # 新上传：需要压缩
            # 检查文件夹大小，决定是否分卷
            dir_size_gb = get_directory_size(local)
            print(f"  文件夹大小: {dir_size_gb:.2f} GB")

            if dir_size_gb > SPLIT_THRESHOLD_GB:
                # 大于阈值，分卷压缩
                print(f"  文件夹大于 {SPLIT_THRESHOLD_GB}GB，使用分卷压缩")
                volume_files = zip_directory_split(
                    local,
                    zip_base_path,
                    volume_size_mb=SPLIT_VOLUME_SIZE_MB
                )
            else:
                # 小于阈值，普通压缩
                print(f"  文件夹小于等于 {SPLIT_THRESHOLD_GB}GB，使用普通压缩")
                volume_files = zip_directory_split(
                    local,
                    zip_base_path,
                    volume_size_mb=None
                )

            if not volume_files:
                print(f"  ✗ 压缩失败，跳过上传")
                failed_uploads.append({
                    'local': local,
                    'remote': remote,
                    'reason': '压缩失败'
                })
                continue

            volume_files_to_upload = volume_files

        # 逐卷上传
        upload_results = []
        for vol_file in volume_files_to_upload:
            vol_basename = os.path.basename(vol_file)
            remote_vol_path = os.path.join(remote, vol_basename)

            print(f"\n  上传卷: {vol_basename}")
            success = upload_with_baidupcs_go(vol_file, remote_vol_path)
            upload_results.append({
                'file': vol_file,
                'remote': remote_vol_path,
                'success': success
            })

        # 检查是否所有卷都上传成功
        all_success = all(result['success'] for result in upload_results)

        if all_success:
            if is_resume:
                print(f"\n  ✓ 续传成功，所有 {len(volume_files_to_upload)} 个缺失卷已上传")
            else:
                print(f"\n  ✓ 所有 {len(volume_files)} 个卷上传成功")
            # 清理所有压缩卷（只有在所有卷都上传成功时才删除）
            for vol_file in volume_files:
                try:
                    if os.path.exists(vol_file):
                        os.remove(vol_file)
                        print(f"    已删除: {os.path.basename(vol_file)}")
                except Exception as e:
                    print(f"    删除失败 {vol_file}: {e}")
        else:
            # 有卷上传失败，保留所有卷
            print(f"\n  ✗ 部分卷上传失败，保留所有压缩卷用于重试")
            failed_vols = [r['file'] for r in upload_results if not r['success']]
            print(f"    失败的卷: {[os.path.basename(f) for f in failed_vols]}")
            failed_uploads.append({
                'local': local,
                'remote': remote,
                'reason': f'部分卷上传失败 ({len(failed_vols)}/{len(volume_files_to_upload)})',
                'failed_volumes': failed_vols,
                'all_volumes': volume_files
            })
    else:
        # 文件：直接上传
        print(f"  上传文件: {local} -> {remote}")
        success = upload_with_baidupcs_go(local, remote)
        if not success:
            failed_uploads.append({
                'local': local,
                'remote': remote,
                'reason': '文件上传失败'
            })

# ============================================================
# 9. 输出上传结果
# ============================================================
print("\n========== Upload Summary ==========")
print(f"云端已存在: {len(already_uploaded)}")
print(f"本次待上传: {len(need_upload)}")
print(f"本次成功: {len(need_upload) - len(failed_uploads)}")
print(f"本次失败: {len(failed_uploads)}")

if failed_uploads:
    print("\n========== 上传失败列表 ==========")
    for i, item in enumerate(failed_uploads, 1):
        print(f"\n[{i}] {item['local']}")
        print(f"    远程路径: {item['remote']}")
        print(f"    失败原因: {item['reason']}")
        if 'all_volumes' in item:
            print(f"    压缩卷位置: {[os.path.basename(v) for v in item['all_volumes']]}")
else:
    print("\n所有文件上传成功！")

print("\n========== Done ==========")