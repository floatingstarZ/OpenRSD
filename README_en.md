# OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ICCV%202025-blue)](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

## 📖 Introduction

Welcome to the official repository of **OpenRSD**! This project proposes an open-prompt remote sensing object detection method that supports multimodal prompts and integrates multi-task detection heads to balance accuracy and real-time requirements for single-stage or two-stage detectors.

**This paper has been accepted by ICCV 2025.**

### Key Features

- 🎯 **Multimodal Prompt Support**: Supports text, image, and other modal prompt inputs
- 🔄 **Multiple Detection Heads**: Supports alignment detection head (high real-time performance, supports large vocabulary) and fusion detection head (high accuracy)

## 🎨 Method Overview

<div align="center">
  <img src="./src/images/Fig2_Method_01.png" width="800"/>
  <p><b>Figure 1: OpenRSD Method Architecture</b></p>
</div>

<div align="center">
  <img src="./src/images/Fig3_Training_Pipeline_01.png" width="800"/>
  <p><b>Figure 2: Multi-stage Training Pipeline</b></p>
</div>

## 📋 Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Quick Start](#-quick-start)
  - [Training](#training)
  - [Testing](#testing)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)

## 🔧 Requirements

- Python >= 3.7
- PyTorch >= 1.8.0
- CUDA >= 10.2
- mmcv-full >= 1.4.0
- mmdetection
- mmrotate
- Other dependencies can be found in `requirements.txt`

## 💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/floatingstarZ/OpenRSD.git
cd OpenRSD
```

### 2. Create Conda Environment (Recommended)

```bash
# Create environment only
conda create -n openrsd python=3.8 -y
# Or use environment.yml to install all dependencies (CUDA 10.2 environment)
conda env create -f environment.yml
```

### 3. Install PyTorch

Install PyTorch according to your CUDA version:

```bash
conda activate openrsd
# For example, CUDA 11.1
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch
```

### 4. Install Dependencies

```bash
# Install the project
pip install -v -e .
# Or
python setup.py develop
```

## 📦 Dataset Preparation

### Dataset Download

All datasets can be downloaded via the following link:

Baidu Netdisk Share: OpenRSD
Link: https://pan.baidu.com/s/1QWWZOfrjAWhEbk1eQASXVQ?pwd=usnc Extraction code: usnc
-- Shared by Baidu Netdisk Super Member v9

### Dataset Organization

All image and annotation folders in the dataset are compressed.
`./BaiduPCS_Upload.py` is the batch upload script used in this project. Its basic principle is: compress -> upload, and preserves the original directory structure.
You can refer to this script to write a batch decompression script. The dataset path is `./data`.

All datasets used in this project are downloaded from the internet as raw remote sensing data and processed through slicing. For specific data processing workflows and scripts, please refer to the `tools/data/` directory.

## 🚀 Quick Start

### Training

#### Single GPU Training

```bash
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

#### Multi-GPU Training

Use the intelligent multi-GPU training scheduling script that automatically detects available GPUs and intelligently allocates resources:

```bash
cd EXP_CONFIG
python multi_train_any_gpu.py -k XXX -c 90
```

**Parameter Description:**

- `-k`: Keyword filtering, specify one or more keywords to filter configurations to train (e.g., `-k A08 A10` means only train models whose configuration names contain A08 or A10)
- `-c`: Starting command count, used to set the starting port number (default 10, port number = 29500 + count)
- `-d`: Specify allowed GPU IDs, separated by commas (e.g., `-d 0,1,2,3` means only use GPUs 0-3, default uses all available GPUs)
- `-r`: Set runner type (options: `det`/`cls`/`few`/`resume`, default is `det`)

**Features:**

- ✅ Automatically detects GPU memory usage and intelligently allocates available GPUs
- ✅ Automatically skips already trained models (checks if checkpoint files exist)
- ✅ Supports multi-task parallel training, automatically manages GPU resources to avoid conflicts
- ✅ Supports batch training of multiple configurations without manual management

**Usage Examples:**

```bash
# Train all configurations containing "A08" keyword
python multi_train_any_gpu.py -k A08 -c 10

# Train configurations with multiple keywords (A08 or A10)
python multi_train_any_gpu.py -k A08 A10 -c 10

# Only use GPUs 0-3 for training
python multi_train_any_gpu.py -k A08 -c 10 -d 0,1,2,3

# Resume training
python multi_train_any_gpu.py -k A08 -c 10 -r resume
```

### Testing

#### Single GPU Testing

```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

#### Batch Testing

Use the batch evaluation script to automatically evaluate models of different epochs on multiple datasets.
You can modify the `model_info` configuration at Line 206 to evaluate different models, where `val_using_aux=True` uses the fusion head and `=False` uses the alignment head.

```bash
cd ./M_Tools/Eval_Tools
python eval_diff_epochs.py -d 0 -e 24
```

**Parameter Description:**

- `-d`: Specify the GPU ID to use (e.g., `-d 0` means use GPU 0)
- `-e`: Specify the list of epochs to evaluate, can specify one or more (e.g., `-e 24` or `-e 12 24 36`)

**Features:**

- ✅ Automatically evaluates on multiple remote sensing datasets (DOTA2, DIOR_R, FAIR1M, SpaceNet, Xview, HRSC2016, WHU_Mix, etc.)
- ✅ Supports batch evaluation of models at multiple epochs
- ✅ Automatically saves evaluation results (including `.pkl` prediction files and `.json` evaluation results)
- ✅ Supports evaluation using auxiliary branches (can configure `val_using_aux` in the script)

**Usage Examples:**

```bash
# Evaluate a single epoch (epoch 24)
python eval_diff_epochs.py -d 0 -e 24

# Evaluate multiple epochs (epochs 12, 24, 36)
python eval_diff_epochs.py -d 0 -e 12 24 36

# Use a different GPU
python eval_diff_epochs.py -d 1 -e 24
```

**Notes:**

- Before use, you need to configure model information (`model_info`) in the script, including:
  - `cfg_pth`: Configuration file path
  - `cfg_name`: Configuration name (used to construct checkpoint path)
  - `val_using_aux`: Which branch to use for evaluation
- Evaluation results will be saved in the `./results/TEST_EVAL/` directory
- Evaluation results for each epoch will be saved in separate subdirectories

### Configuration Files

Configuration files are located in the `M_configs/` directory, including:

- `Step1_A08_Large_Pretrain/`: Large-scale pretraining configurations
- `Step2_A10_Large_Pretrain_Stage3/`: Stage 3 pretraining configurations
- `Step3_A12_SelfTrain/`: Self-training configurations
- `Other/`: Other configurations

## 📁 Project Structure

```
MMRotate_AD_Pub/
├── M_AD/                    # Main algorithm implementation
│   ├── models/              # Model definitions
│   │   ├── detectors/       # Detectors (Flex_Rtmdet, E_Rtmdet, Hindsight_Rtmdet, etc.)
│   │   ├── dense_heads/     # Detection heads (Flex_Rrtmdet_head, E_Rrtmdet_head, etc.)
│   │   ├── backbones/       # Backbones (CSPNeXt, ViT, Swin, etc.)
│   │   ├── necks/           # Neck networks (PAFPN, Ace_fpn, etc.)
│   │   ├── roi_heads/       # ROI heads (Open_standard_roi_head, Hin_Box_Prompt_head, etc.)
│   │   ├── layers/          # Custom layers (Transformer, DINOv2 related layers)
│   │   ├── task_modules/    # Task modules (Assigner, etc.)
│   │   └── utils/           # Utility functions
│   ├── datasets/            # Dataset related
│   │   ├── samplers/        # Data samplers (multi-task samplers, etc.)
│   │   └── transforms/      # Data transforms
│   ├── engine/              # Training engine
│   │   ├── optimizers/      # Optimizers
│   │   └── runner/          # Training runners
│   ├── evaluation/          # Evaluation related
│   │   └── metrics/          # Evaluation metrics
│   └── structures/          # Data structures
│       └── bbox/            # Bounding box related
├── M_configs/               # Configuration files
│   ├── Step1_A08_Large_Pretrain/      # Large-scale pretraining configurations
│   ├── Step2_A10_Large_Pretrain_Stage3/  # Stage 3 pretraining configurations
│   ├── Step3_A12_SelfTrain/           # Self-training configurations
│   └── Other/                          # Other configurations (e.g., InContext learning)
├── M_Tools/                 # Tool script collection
│   ├── Eval_Tools/          # Evaluation tools
│   │   ├── eval_diff_epochs.py        # Batch evaluation for different epochs
│   │   ├── auto_eval.py              # Automatic evaluation script
│   │   ├── eval_cross_data.py        # Cross-dataset evaluation
│   │   └── eval_configs/             # Evaluation configurations
│   └── Base_Data_infos/     # Dataset information configurations
├── EXP_CONFIG/              # Experiment configuration management
│   ├── multi_train_any_gpu.py         # Multi-GPU training scheduling script
│   ├── multi_eval_any_gpu.py         # Multi-GPU evaluation scheduling script
│   ├── py_cmd.py                      # Training command wrapper script
│   └── CONFIGS/                       # Experiment configuration definitions
├── tools/                   # MMDetection/MMRotate tool scripts
│   ├── train.py            # Training script
│   ├── test.py             # Testing script
│   ├── data/               # Data processing tools (DOTA, DIOR, FAIR1M dataset processing, etc.)
│   ├── analysis_tools/     # Analysis tools (log analysis, result analysis, etc.)
│   └── model_converters/   # Model conversion tools
├── mmdet/                   # MMDetection core code
├── mmrotate/                # MMRotate core code
├── mmyolo/                  # MMYOLO core code (partial functionality dependency)
├── commonlibs/              # Common utility library
├── ctlib/                   # Custom utility library
├── src/                     # Resource files
│   └── images/             # Image resources (method diagrams, etc.)
├── requirements.txt        # Python dependency list
├── setup.py                # Installation script
├── environment.yml         # Conda environment configuration
└── README.md               # This file
```

**Main Directory Descriptions:**

- **M_AD/**: Core algorithm implementation, containing all custom models, datasets, training engines, etc.
- **M_configs/**: Experiment configuration files, organized by training stage
- **M_Tools/**: Evaluation and data processing tool collection
- **EXP_CONFIG/**: Experiment management and scheduling scripts, supporting automatic multi-GPU scheduling
- **tools/**: Standard tools provided by MMDetection/MMRotate framework

## 📊 Results

For detailed experimental results and model weights, please refer to the paper. Main results include:

- Detection performance on multiple remote sensing datasets
- Accuracy and speed comparisons under different configurations
- Ablation study results

### Performance Comparison

<div align="center">
  <img src="./src/images/fig1_compare.png" width="800"/>
  <p><b>Figure 3: Performance Comparison</b></p>
</div>

## 📄 Citation

If you use this codebase in your research or wish to refer to the baseline results published here, please use the following BibTeX entry:

```BibTeX
@inproceedings{huang2025openrsd,
  title={OpenRSD: Towards open-prompts for object detection in remote sensing images},
  author={Huang, Ziyue and Feng, Yongchao and Liu, Ziqi and Yang, Shuai and Liu, Qingjie and Wang, Yunhong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8384--8394},
  year={2025}
}
```

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 🙏 Acknowledgments

This project is based on the following excellent open-source projects:

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMRotate](https://github.com/open-mmlab/mmrotate)
- [MMYOLO](https://github.com/open-mmlab/mmyolo)

Thanks to all contributors and authors of related work!

## ❓ FAQ

### Q: How to choose configuration files?

A: Select the corresponding configuration file according to your training stage:
- **Step1**: Large-scale pretraining stage
- **Step2**: Stage 3 pretraining
- **Step3**: Self-training stage

### Q: What to do when CUDA out of memory occurs during training?

A: You can try the following methods:
- Reduce `batch_size`
- Reduce input image size `img_scale`
- Use gradient accumulation
- Use fewer GPUs

### Q: How to train on your own dataset?

A: Please refer to the following steps:
1. Prepare the dataset, format reference examples in the `tools/data/` directory
2. Modify the data path and number of classes in the configuration file
3. Adjust training parameters as needed

### Q: How to evaluate model performance?

A: Use the testing script:
```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mAP
```

## 📮 Contact

If you have any questions or suggestions, please contact us through:

- Submit an [Issue](https://github.com/floatingstarZ/OpenRSD/issues)
- Send an email to the project maintainer (ziyuehuang@buaa.edu.cn)

## 🔗 Related Links

- [Paper Link](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf)
- [MMDetection Documentation](https://mmdetection.readthedocs.io/)
- [MMRotate Documentation](https://mmrotate.readthedocs.io/)

---

<div align="center">
  <b>⭐ If this project is helpful to you, please give us a Star! ⭐</b>
</div>
