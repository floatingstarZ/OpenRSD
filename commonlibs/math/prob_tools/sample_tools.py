import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from commonlibs.transform_tools.type_transform import all_to_numpy


def normal(loc, scale, N):
    return np.random.normal(loc, scale, N)

def mul_normal(mean, cov, N):
    """
    
    :param mean: M
    :param cov: M x M
    :param N: data number
    :return: N x M gaussian sample
    """
    mean = all_to_numpy(mean)
    cov = all_to_numpy(cov)
    return np.random.multivariate_normal(mean, cov, N)

def uniform(low, high, N):
    return np.random.uniform(low, high, N)

def reject(D, reject_prob):
    """
    resample from D, with reject probability reject_prob
    :param D: N x M
    :param reject_prob: N
    :return: 
    """
    D = all_to_numpy(D)
    p_rej = all_to_numpy(reject_prob)
    N = len(D)
    assert N == len(p_rej)
    if N == 0:
        return []
    p_s = uniform(0, 1, N)
    acc_ind = p_s > reject_prob
    return D[acc_ind], acc_ind

def accept(D, accept_prob):
    accept_prob = all_to_numpy(accept_prob)
    reject_prob = 1 - accept_prob
    return reject(D, reject_prob)


if __name__ == '__main__':
    # m = [0, 0]
    # cov = [[3, 2], [2, 2]]
    # points = mul_normal(m, cov, 1000)
    # plt.scatter(points[:, 0], points[:, 1], c='b', marker='*')
    # plt.show()
    # plt.pause(3)
    from commonlibs.drawing_tools.draw_hist import interval_hist
    N = 10000
    data = uniform(0, 1, N)
    x, h_y1 = interval_hist(data, 10)
    plt.plot(x, h_y1, '-o',  color=(0, 1, 0))

    p_acc = data + normal(0, 0.1, N)
    p_acc = p_acc - p_acc.min()
    # p_acc = p_acc / p_acc.max()
    # plt.scatter(data, p_acc, c='b', marker='*')

    data = accept(data, p_acc)
    x, h_y_acc = interval_hist(data, 10)
    plt.plot(x, h_y_acc, '-o',  color=(0, 0, 1))


    plt.pause(5)









