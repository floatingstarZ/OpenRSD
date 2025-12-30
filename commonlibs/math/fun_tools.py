import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def normal2D(x, y, mu, s):
    mu = np.array(mu).reshape(2, 1)
    p = np.array([[x, y]]).T
    a = np.exp(-np.matmul((p - mu).T, (p - mu)) / (2 * s))[0, 0]
    b = 1 / (2 * np.pi * s)
    return b * a


def lap2D(x, y, mu, s):
    mu = np.array(mu).reshape(2)
    p = np.array([x, y])
    a = np.exp(-(abs(p[0] - mu[0]) + abs(p[1] - mu[1])) / (s))
    b = 1 / (2 * s)
    return b * a


def nal2D(x, y, mu, s):
    return lap2D(x, y, mu, s) + normal2D(x, y, mu, s)


def M2D(p1, p2):
    (x1, y1, w1, h1) = p1
    (x2, y2, w2, h2) = p2
    l1 = np.sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2 + (w1 - w2) ** 2 + (h1 - h2) ** 2)
    # l1 = abs(x2-x1)+abs(y1-y2)+abs(w1-w2)+abs(h1-h2)

    s1 = w1 + h1
    s2 = w2 + h2
    # 可行的一些组合（200个点随机排列）
    # lS = max(s1, s2, s1 + s2)
    # lS = abs(s1 - s2)
    # lS = w1+h1+w2+h2
    lS = min(w1 + h1, w2 + h2)  # 这个是比较好的
    # lS = math.sqrt(w1*h1)+math.sqrt(w2*h2)

    # 不可行的
    # lS = w1*h1+w2*h2
    # lS = abs(w1*h1 - w2*h2)



    if lS + l1 == 0:
        return 1
    # return l2 / (s1 + s2 + l2)
    return l1 / (lS + l1)


def IOU2D(p1, p2):
    (x1, y1, w1, h1) = p1
    (x2, y2, w2, h2) = p2
    # p1 = wh2lt(*p1)
    # p2 = wh2lt(*p2)

    rd = (min(p1[2], p2[2]), min(p1[3], p2[3]))
    lt = (max(p1[0], p2[0]), max(p1[1], p2[1]))
    inter = max(rd[0] - lt[0], 0) * max(rd[1] - lt[1], 0)
    S1 = w1 * h1
    S2 = w2 * h2
    if S1 + S2 + inter == 0:
        return 1
    return 1 - inter / (S1 + S2 - inter)


#  from https://blog.csdn.net/lyl771857509/article/details/84113177
def blur(input, kernel):
    input = torch.Tensor(input)
    dim_input = 2
    if len(input.shape) == 2:
        (H, W) = input.shape
        input = input.expand(1, 1, H, W)
        C = 1
    elif len(input.shape) == 3:
        dim_input = 3
        (C, H, W) = input.shape
        input = input.expand(1, C, H, W)
    else:
        dim_input = 4
        C = input.shape[1]
    H, W = len(kernel), len(kernel[0])
    # kernel有C个参数，每个参数为1*H*W
    kernel = torch.Tensor(kernel).expand(C, 1, H, W)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    # groups = C：每个channel分别卷积
    result = F.conv2d(input, weight, stride=1, padding=1, groups=C)
    if dim_input == 2:
        return result[0][0]
    elif dim_input == 3:
        return result[0]
    else:
        return result


def laplacian(input):
    kernel = [[1, 1, 1],
              [1, -8, 1],
              [1, 1, 1]]
    return blur(input, kernel)

def average(input):
    # kernel = np.array([[1, 1, 1],
    #                    [1, 1, 1],
    #                    [1, 1, 1]])
    kernel = np.array([[0.2, 0.2, 0.2],
                       [0.2, 1, 0.2],
                       [0.2, 0.2, 0.2]])
    kernel = kernel / np.sum(kernel)
    return blur(input, kernel)


if __name__ == '__main__':
    kernel = [[1, 1, 1],
              [1, -8, 1],
              [1, 1, 1]]
    # input = torch.ones([3, 3, 3])
    # print(input)
    # print(blur(input, kernel))
    #############################################################
    import pickle

    fp = './Lap_Results/Results_Step_0.100000_B_0.100000_O_0.100000_S_0.300000.pkl'
    # './Results/Results_Step_0.100000_B_0.100000_O_0.100000_S_0.300000.pkl'
    with open(fp, 'rb') as f:
        result = pickle.load(f)
    Z = laplacian(result['Z']).numpy()
    b = 0
    Xs = np.arange(-2, 5, 0.1)
    Ys = np.arange(-2, 5, 0.1)
    mesh_Xs, mesh_Ys = np.meshgrid(Xs, Ys)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(mesh_Xs, mesh_Ys, Z, rstride=1, cstride=1)
    plt.pause(1)
    plt.close()

    Z = -torch.Tensor(Z)
    H, W = Z.shape
    values, inds = torch.topk(Z.view(-1), 100)
    print('RRRRRRRRRRRRRRRRRRRRRRRRRRRR')
    results = []
    for id, ind in enumerate(inds):
        i = ind // W
        j = ind - i * W
        # print((i, j), ind, W, values[id])
        print('coordinate: (%f, %f) | score: %f' %
              (mesh_Xs[i, j], mesh_Ys[i, j], values[id].data))












