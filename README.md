
# OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images
[Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf)

Welcome to the official repository of [MutDet](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf). 
In this work, we propose an open-prompt remote sensing object detection method, which supports multimodal prompts and integrates multi-task detection heads to balance accuracy and real-time requirement single-stage or two-stage detectors.
Our paper is accepted by ICCV 2025. 

![diagram](./src/images/Fig2_Method_01.png)



## Preparation

```shell
pip install -v -e .
# or 
python setup.py
```

## Datasets

所有数据集：
通过网盘分享的文件：OpenRSD
链接: https://pan.baidu.com/s/1c-EbjmQApNC8RBxeHlmHMQ?pwd=sxdc 提取码: sxdc 
--来自百度网盘超级会员v9的分享


## Citation
If you use this toolbox in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

- Citing **MutDet**:

```BibTeX
@inproceedings{huang2025openrsd,
  title={Openrsd: Towards open-prompts for object detection in remote sensing images},
  author={Huang, Ziyue and Feng, Yongchao and Liu, Ziqi and Yang, Shuai and Liu, Qingjie and Wang, Yunhong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8384--8394},
  year={2025}
}
```