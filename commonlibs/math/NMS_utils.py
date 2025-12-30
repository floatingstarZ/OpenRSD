import torch
import numpy as np
import torch.nn as nn
import cv2
from commonlibs.math_tools.fun_tools import *
# copy from toy detection

def local_max(heat, kernel=7):
    """
    from CenterNet/src/lib/models/decode.py
    :param heat: batch * C * W * H
    :return: l_max : batch * C * W * H
    """
    C, H, W = heat.shape
    flat_pad = (kernel - 1) // 2
    c_pad = (C - 1) // 2
    # 水平 pooling
    # hmax = nn.functional.max_pool2d(
    #     heat, (kernel, kernel), stride=1, padding=pad)
    m = nn.MaxPool3d((C, kernel, kernel), stride=(1, 1, 1),
                     padding=(0, flat_pad, flat_pad))
    heat = heat.unsqueeze(0)
    heat = heat.unsqueeze(0)
    hmax = m(heat).squeeze(0).squeeze(0)
    hmax = hmax.expand(C, H, W)
    heat[hmax != heat] = 0
    # a = (hmax == heat).numpy()
    # heat = heat.unsqueeze(0)
    # hmax = nn.functional.max_pool2d(
    #     heat, (kernel, kernel), stride=1, padding=pad)
    # b = (hmax == heat).numpy()

    return heat

def decode_heat_map(reg, score, threshold=0, K=100):
    """
    
    :param reg: A * 4 x H x W
    :param score: A x H x W
    :param threshold: 
    :return: 
    """
    # A x H x W
    A, H, W = score.shape
    score = score.clone()
    score = local_max(score)
    reg = reg.clone()
    # score = torch.Tensor([1])
    # A * H * W
    score = score.view(-1)
    score[score < threshold] = 0
    K = min(K, len(score))
    topk_scores, ind = torch.topk(score, K)
    # A * H * W -> A x H x W
    empty_ind = torch.zeros_like(score).long()
    # ind = empty_ind.scatter(0, ind, 1)
    # ind = ind.view(A, H, W)
    # split reg
    x1, y1 = reg[0:32 + 1:4], reg[1:32 + 2:4]
    x2, y2 = reg[2:32 + 3:4], reg[3:32 + 4:4]
    x1, y1 = x1.contiguous().view(-1),  y1.contiguous().view(-1)
    x2, y2 = x2.contiguous().view(-1),  y2.contiguous().view(-1)

    x1 = x1[ind].view(-1, 1)
    y1 = y1[ind].view(-1, 1)
    x2 = x2[ind].view(-1, 1)
    y2 = y2[ind].view(-1, 1)
    bbox = torch.cat([x1, y1, x2, y2, topk_scores.view(-1, 1)], dim=1)
    return bbox


if __name__ == '__main__':
    a = np.array([[[1, 2],
                   [1, 3]],
                  [[3, 2],
                   [4, 0]]])
    b = np.array([[[10, 20],
                   [10, 30]],
                  [[10, 20],
                   [10, 30]],
                  [[10, 20],
                   [10, 30]],
                  [[10, 20],
                   [10, 30]],

                  [[30, 20],
                   [40, 0]],
                  [[30, 20],
                   [40, 0]],
                  [[30, 20],
                   [40, 0]],
                  [[30, 20],
                   [40, 0]]])
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    bbox = decode_heat_map(b, a, K=3)
    print(bbox)


    a = torch.ones([1, 1, 2, 10, 10])
    m = nn.MaxPool3d((3, 3, 3), stride=(1, 1, 1),
                     padding=(1, 1, 1))
    a[0, 0, 0, 0, 0] = 10
    print(m(a) == a)




























