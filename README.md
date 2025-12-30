# OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ICCV%202025-blue)](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

## 📖 简介

欢迎来到 **OpenRSD** 的官方代码仓库！本项目提出了一种支持开放提示（open-prompts）的遥感图像目标检测方法，支持多模态提示并集成多任务检测头，以平衡单阶段或两阶段检测器的精度和实时性要求。

**本论文已被 ICCV 2025 接收。**

### 主要特性

- 🎯 **多模态提示支持**：支持文本、图像等多种模态的提示输入
- 🔄 **多任务检测头**：集成多任务检测头以平衡精度和实时性
- 🚀 **灵活架构**：兼容单阶段和两阶段检测器
- 📊 **高性能**：在多个遥感数据集上取得优异性能

## 🎨 方法概述

<div align="center">
  <img src="./src/images/Fig2_Method_01.png" width="800"/>
  <p><b>图 1: OpenRSD 方法架构</b></p>
</div>

<div align="center">
  <img src="./src/images/Fig3_Training_Pipeline_01.png" width="800"/>
  <p><b>图 2: 训练流程</b></p>
</div>

## 📋 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
  - [训练](#训练)
  - [测试](#测试)
- [项目结构](#项目结构)
- [结果](#结果)
- [引用](#引用)
- [许可证](#许可证)

## 🔧 环境要求

- Python >= 3.7
- PyTorch >= 1.8.0
- CUDA >= 10.2
- mmcv-full >= 1.4.0
- mmdetection
- mmrotate
- 其他依赖见 `requirements.txt`

## 💻 安装

### 1. 克隆仓库

```bash
git clone <repository-url>
cd MMRotate_AD_Pub
```

### 2. 创建 conda 环境（推荐）

```bash
conda create -n openrsd python=3.8 -y
conda activate openrsd
```

### 3. 安装 PyTorch

根据您的 CUDA 版本安装对应的 PyTorch：

```bash
# 例如 CUDA 11.1
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch
```

### 4. 安装依赖

```bash
# 安装 mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

# 安装项目
pip install -v -e .
# 或者
python setup.py develop
```

## 📦 数据集准备

### 数据集下载

所有数据集可通过以下链接下载：

**百度网盘**：
- 链接: https://pan.baidu.com/s/1c-EbjmQApNC8RBxeHlmHMQ?pwd=sxdc 
- 提取码: `sxdc`

### 数据集组织

下载后，请按照以下结构组织数据集：

```
data/
├── DIOR/
│   ├── annotations/
│   ├── images/
│   └── ...
├── DOTA/
│   ├── annotations/
│   ├── images/
│   └── ...
└── ...
```

具体的数据集准备步骤请参考各数据集的 README 文件（位于 `tools/data/` 目录下）。

## 🚀 快速开始

### 训练

#### 单 GPU 训练

```bash
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

#### 多 GPU 训练

```bash
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

#### 示例

```bash
# 单 GPU 训练
python tools/train.py M_configs/Step1_A08_Large_Pretrain/A08_e_rtm_v2_base.py

# 多 GPU 训练（例如 8 个 GPU）
bash tools/dist_train.sh M_configs/Step1_A08_Large_Pretrain/A08_e_rtm_v2_base.py 8
```

### 测试

#### 单 GPU 测试

```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

#### 多 GPU 测试

```bash
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```

#### 示例

```bash
# 单 GPU 测试
python tools/test.py M_configs/Step1_A08_Large_Pretrain/A08_e_rtm_v2_base.py work_dirs/checkpoint.pth

# 多 GPU 测试
bash tools/dist_test.sh M_configs/Step1_A08_Large_Pretrain/A08_e_rtm_v2_base.py work_dirs/checkpoint.pth 8
```

### 配置文件

配置文件位于 `M_configs/` 目录下，包含：

- `Step1_A08_Large_Pretrain/`: 大规模预训练配置
- `Step2_A10_Large_Pretrain_Stage3/`: 预训练第三阶段配置
- `Step3_A12_SelfTrain/`: 自训练配置
- `Other/`: 其他配置

## 📁 项目结构

```
MMRotate_AD_Pub/
├── M_AD/                    # 主要算法实现
│   ├── models/              # 模型定义
│   │   ├── detectors/       # 检测器
│   │   ├── dense_heads/     # 检测头
│   │   ├── backbones/       # 骨干网络
│   │   └── ...
│   ├── datasets/            # 数据集相关
│   ├── engine/              # 训练引擎
│   └── ...
├── M_configs/               # 配置文件
├── tools/                   # 工具脚本
│   ├── train.py            # 训练脚本
│   ├── test.py             # 测试脚本
│   └── ...
├── mmdet/                   # MMDetection 核心代码
├── mmrotate/                # MMRotate 核心代码
├── src/                     # 资源文件
│   └── images/             # 图片资源
├── requirements.txt        # 依赖列表
├── setup.py                # 安装脚本
└── README.md               # 本文件
```

## 📊 结果

详细的实验结果和模型权重请参考论文。主要结果包括：

- 在多个遥感数据集上的检测性能
- 不同配置下的精度和速度对比
- 消融实验结果

### 性能对比

<div align="center">
  <img src="./src/images/fig1_compare.png" width="800"/>
  <p><b>图 3: 性能对比</b></p>
</div>

## 📄 论文

如果您使用本代码或参考了我们的结果，请引用我们的论文：

```BibTeX
@inproceedings{huang2025openrsd,
  title={OpenRSD: Towards open-prompts for object detection in remote sensing images},
  author={Huang, Ziyue and Feng, Yongchao and Liu, Ziqi and Yang, Shuai and Liu, Qingjie and Wang, Yunhong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8384--8394},
  year={2025}
}
```

## 📜 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 🙏 致谢

本项目基于以下优秀的开源项目：

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMRotate](https://github.com/open-mmlab/mmrotate)
- [MMYOLO](https://github.com/open-mmlab/mmyolo)

感谢所有贡献者和相关工作的作者！

## ❓ 常见问题

### Q: 如何选择配置文件？

A: 根据您的训练阶段选择对应的配置文件：
- **Step1**: 大规模预训练阶段
- **Step2**: 预训练第三阶段
- **Step3**: 自训练阶段

### Q: 训练时出现 CUDA 内存不足怎么办？

A: 可以尝试以下方法：
- 减小 `batch_size`
- 减小输入图像尺寸 `img_scale`
- 使用梯度累积
- 使用更少的 GPU 数量

### Q: 如何在自己的数据集上训练？

A: 请参考以下步骤：
1. 准备数据集，格式参考 `tools/data/` 目录下的示例
2. 修改配置文件中的数据路径和类别数
3. 根据需要调整训练参数

### Q: 如何评估模型性能？

A: 使用测试脚本：
```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mAP
```

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/your-repo/issues)
- 发送邮件至项目维护者

## 🔗 相关链接

- [论文链接](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf)
- [MMDetection 文档](https://mmdetection.readthedocs.io/)
- [MMRotate 文档](https://mmrotate.readthedocs.io/)

---

<div align="center">
  <b>⭐ 如果这个项目对您有帮助，请给我们一个 Star！⭐</b>
</div>