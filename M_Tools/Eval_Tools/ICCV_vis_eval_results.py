import json
import os

from commonlibs.common_tools import *
from commonlibs.drawing_tools import *

import pandas as pd
from functools import cmp_to_key
import scipy.stats as stats
from commonlibs.drawing_tools.diagram import *
from commonlibs.common_tools import *

import matplotlib.pyplot as plt
import pylab
import imageio
import skimage.io
import cv2
from io import BytesIO
import PIL
from pathlib import Path

def get_ax_obj(title, x_tick=None, figsize=(5, 5), xlabel='X', ylabel='Y'):
    fig_all = plt.figure(figsize=figsize)
    ax = fig_all.add_subplot(1, 1, 1)
    ax.set_xlabel(xlabel)  # 设置x轴标签
    ax.set_ylabel(ylabel)  # 设置y轴标签
    ax.plot()
    if x_tick:
        ax.set_xticks(x_tick)
    return fig_all, ax


def plt_2_img():
    # 申请缓冲地址
    buffer_ = BytesIO()  # using buffer,great way!
    # 保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
    plt.savefig(buffer_, format='png')
    buffer_.seek(0)
    # 用PIL或CV2从内存中读取
    dataPIL = PIL.Image.open(buffer_)
    dataPIL = dataPIL.convert("RGB")
    # 转换为nparrary，PIL转换就非常快了,data即为所需
    img = np.asarray(dataPIL)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 释放缓存
    buffer_.close()
    return img

def average(xs, ys, bin_size=10):
    """
    smooth the curves
    :param xs:
    :param ys:
    :param bin_size:
    :return:
    """
    assert len(xs) == len(ys)
    Intervals = list(range(0, len(xs), bin_size))
    new_xs = []
    new_ys = []
    for i in range(len(Intervals) - 1):
        new_xs.append(np.mean(xs[Intervals[i]: Intervals[i+1]]))
        new_ys.append(np.mean(ys[Intervals[i]: Intervals[i+1]]))
    return np.array(new_xs), np.array(new_ys)


def plot_curves(x, ys, ls,
                x_label, y_label,
                save_path,
                ylim):
    cs = ncolors(len(ys))
    cs = norm_colors(cs)

    fig_all, ax = get_ax_obj(save_path, figsize=(16, 16))
    plt.grid(True, linestyle='--', linewidth=2)

    for y, l, c in zip(ys, ls, cs):
        ax.plot(x, y, '--', label=l, color=c, linewidth=4,
                marker='s', markersize=12)

    # for y in ys:
    #     for i, value in enumerate(y):
    #         if i in [0, 1, 3, 5, 7, 9, 11]:
    #             plt.text(x[i], value + 0.001, f'{value:.3f}', ha='center', va='bottom',
    #                      fontsize=20)

    # ax.legend(loc='lower right', frameon=True, fontsize=20)
    ax.set_ylim(ylim[0], ylim[1])
    plt.xticks([2, 4, 8, 12, 16, 20, 24])
    plt.yticks(np.arange(ylim[0], ylim[1], 5))

    # 调整布局，避免图例和图形重叠


    ax.set_xlabel(x_label, fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    plt.tick_params(axis='both', labelsize=24)  # 设置刻度标签字号

    plt.savefig(save_path)
    plt.close()

# """
# Legend用的plot
# """
# def plot_curves(x, ys, ls,
#                 x_label, y_label,
#                 save_path,
#                 ylim):
#     cs = ncolors(len(ys))
#     cs = norm_colors(cs)
#
#     fig_all, ax = get_ax_obj(save_path, figsize=(72, 72))
#     plt.grid(True, linestyle='--', linewidth=2)
#
#     for y, l, c in zip(ys, ls, cs):
#         # ax.plot(x, y, '--', label=l, color=c, linewidth=4,
#         #         marker='s', markersize=12)
#         ax.plot(x, y, '-', label=l, color=c, linewidth=24)
#
#     # for y in ys:
#     #     for i, value in enumerate(y):
#     #         if i in [0, 1, 3, 5, 7, 9, 11]:
#     #             plt.text(x[i], value + 0.001, f'{value:.3f}', ha='center', va='bottom',
#     #                      fontsize=20)
#
#     # ax.legend(loc='lower right', frameon=True, fontsize=20)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=7, frameon=False
#               ,fontsize=48)
#     ax.set_ylim(ylim[0], ylim[1])
#     plt.xticks([2, 4, 8, 12, 16, 20, 24])
#     plt.yticks(np.arange(ylim[0], ylim[1], 5))
#
#     # 调整布局，避免图例和图形重叠
#
#
#     ax.set_xlabel(x_label, fontsize=30)
#     ax.set_ylabel(y_label, fontsize=30)
#     plt.tick_params(axis='both', labelsize=24)  # 设置刻度标签字号
#
#     plt.savefig(save_path)
#     plt.close()

# def plot_curves(x, ys, ls,
#                 x_label, y_label,
#                 save_path,
#                 ylim):
#     cs = ncolors(len(ys))
#     cs = norm_colors(cs)
#
#     fig_all, ax = get_ax_obj(save_path, figsize=(16, 16))
#     # plt.grid(axis='y', color='gray', linestyle='--', linewidth=3)
#
#     for y, l, c in zip(ys, ls, cs):
#         ax.plot(x, y, '--', label=l, color=c, linewidth=4,
#                 marker='s', markersize=12)
#
#     for y in ys:
#         for i, value in enumerate(y):
#             if i in [0, 1, 3, 5, 7, 9, 11]:
#                 plt.text(x[i], value + 0.001, f'{value:.3f}', ha='center', va='bottom',
#                          fontsize=20)
#
#     # ax.legend(loc='lower right', frameon=False, fontsize=20)
#     ax.set_ylim(ylim[0], ylim[1])
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=7, frameon=False
#               ,fontsize=20)
#
#     # 调整布局，避免图例和图形重叠
#
#
#     ax.set_xlabel(x_label, fontsize=24)
#     ax.set_ylabel(y_label, fontsize=24)
#     plt.tick_params(axis='both', labelsize=24)  # 设置刻度标签字号
#
#     plt.savefig(save_path)
#     plt.close()

import re
if __name__ == '__main__':
    """
    eval_root = '/data/space2/huangziyue/mmdet_results/TEST_EVAL/'\
                 'Formal_11_22_A12_flex_rtm_v3_1_self_training'
    fig_path = './Formal_11_22_A12_flex_rtm_v3_1_self_training_evals.png'
    
        eval_root = '/data/space2/huangziyue/mmdet_results/TEST_EVAL/'\
                 'Formal_11_22_A12_flex_rtm_v3_1_maid_self_training'
    fig_path = './Formal_11_22_A12_flex_rtm_v3_1_maid_self_training.png'
    
        eval_root = '/data/space2/huangziyue/mmdet_results/TEST_EVAL/'\
                 'Formal_11_25_cross_A12_flex_rtm_v3_1_maid_self_training'
    fig_path = './Formal_11_25_cross_A12_flex_rtm_v3_1_maid_self_training.png'
    
        eval_root = '/data/space2/huangziyue/mmdet_results/TEST_EVAL/'\
                 'Formal_11_22_A12_flex_rtm_v3_1_self_training'
    fig_path = './Formal_11_22_A12_flex_rtm_v3_1_self_training.png'
    """
    eval_root = '/data/space2/huangziyue/mmdet_results/TEST_EVAL/'\
                 'Formal_12_11_A10_flex_rtm_v3_1_formal'
    eval_name = Path(eval_root).stem
    fig_path = f'ICCV_{eval_name}.png'
    eval_results = dict()

    data_name_mapping = {
        'Data1_DOTA1': 'DOTA-v1.0',
        'Data1_DOTA2': 'DOTA-v2.0',
        'Data2_DIOR_R': 'DIOR-R',
        'Data3_FAIR1M': 'FAIR1M-2.0',
        'Data5_SpaceNet': 'SpaceNet',
        'Data7_HRSC2016': 'HRSC2016',
        'Data11_WHU_Mix': 'WHU-Mix',
    }


    # ---- 每个Epoch
    for eval_dir in os.listdir(eval_root):
        pattern = r"^Formal_Results_Epoch_(\d+)$"
        match = re.match(pattern, eval_dir)
        if match is None:
            continue
        epoch_number = int(match.group(1))
        eval_dir_pth = f'{eval_root}/{eval_dir}'
        # ---- 每个数据集
        e_results = dict()
        for eval_file in os.listdir(eval_dir_pth):
            if Path(eval_file).suffix != '.json':
                continue
            eval_file_pth = f'{eval_dir_pth}/{eval_file}'
            eval_result = jsonload(eval_file_pth, msg=False)
            ap50 = eval_result['dota/AP50']
            data_name = Path(eval_file).stem
            if data_name not in data_name_mapping.keys():
                continue
            e_results[data_name_mapping[data_name]] = ap50
        eval_results[epoch_number] = e_results

    # ---- 整合所有的data
    data_names = sorted(list(set().union(*[list(per_epoch.keys()) for per_epoch in eval_results.values()])))

    xs = sorted(list(eval_results.keys()))
    ys = []
    for data_name in data_names:
        aps = []
        for epoch in xs:
            e_r = eval_results[epoch]
            if data_name not in e_r.keys():
                print(f'Epoch {epoch} has not {data_name} eval results')
                aps.append(0.0)
                continue
            aps.append(e_r[data_name])
        aps = np.array(aps) * 100
        ys.append(aps)

    plot_curves(xs,
                ys,
                data_names,
                'Epoch', 'AP50',
                fig_path,
                ylim=[30, 95])

