import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import colorsys

def interval_hist(data, n_interval=1000):
    """
    
    :param data: N, array
    :return: 
    """
    m_d = float(np.min(data))
    m_u = float(np.max(data))
    n_int = min(len(data), n_interval)
    step = (m_u - m_d) / float(n_int)
    x = np.arange(m_d, m_u, step)
    y = np.zeros_like(x)
    for d in data:
        y[min(int((d-m_d)//step), n_int-1)] += 1
    return x, y


if __name__ == '__main__':
    a = np.random.rand(1000)
    x, y = interval_hist(a, 10)
    print(x, y)