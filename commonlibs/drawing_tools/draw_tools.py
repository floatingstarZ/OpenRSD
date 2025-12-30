import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import colorsys

def get_ax_obj(title, x_tick=None, figsize=(5, 5), xlabel='X', ylabel='Y'):
    fig_all = plt.figure(figsize=figsize)
    ax = fig_all.add_subplot(1, 1, 1)
    ax.set_xlabel(xlabel)  # 设置x轴标签
    ax.set_ylabel(ylabel)  # 设置y轴标签
    ax.set_title(title)
    ax.plot()
    if x_tick:
        ax.set_xticks(x_tick)
    return fig_all, ax

def scatter_2D(ax, points, color='b', marker='*'):
    ax.scatter(points[:, 0], points[:, 1], c=color, marker=marker)

def scatter_2D_in_3D(ax, points, color='b', marker='*'):
    ax.scatter(points[:, 0], points[:, 1], np.zeros_like(points[:, 0]), c=color, marker=marker)

def draw_Normal_3D(mu, s, ax):
    """
    
    :param mu: [1, 2]
    :param s: float       Sigma = s * I
    :param ax: 
    :return: 
    """
    X = np.arange(-3, 3, 0.25)
    Y = np.arange(-3, 3, 0.25)
    plt_X, plt_Y = np.meshgrid(X, Y)

    Z = np.zeros_like(plt_X)
    mu = np.array(mu).reshape(2, 1)
    for idx, x in enumerate(X):
        for idy, y in enumerate(Y):
            p = np.array([[x, y]]).T
            a = np.exp(-np.matmul((p - mu).T, (p - mu)) / (2 * s))[0, 0]
            b = 1 / (2 * np.pi * s)
            Z[idx, idy] = b * a

    # ax.plot_surface(plt_X, plt_Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_wireframe(plt_X, plt_Y, Z, rstride=1, cstride=1)

# from NMSExp/utils
def draw_vector_field(mesh_X, mesh_Y, offsets, time=0, path=None):
    fig_all = plt.figure(figsize=(12, 12))
    ax = fig_all.add_subplot(1, 1, 1)
    ax.set_xlabel('X')  # 设置x轴标签
    ax.set_ylabel('Y')  # 设置y轴标签
    ax.set_title('Org')
    # scatter_2D(ax, predicts, color='g', marker='.')
    # scatter_2D(ax, anchors, color='r', marker='p')

    # ax.quiver(mesh_X, mesh_Y, offsets[:, :, 0], offsets[:, :, 1], scale=1,
    #           angles="xy", scale_units='xy', width=0.0005, color="#666666")
    #
    ax.quiver(mesh_X, mesh_Y, offsets[0, :, :], offsets[1, :, :], scale=1,
              angles="xy", scale_units='xy', width=0.001, color="#666666")

    # ax.quiver(mesh_X, mesh_Y, 0.2 * np.ones_like(mesh_X), np.zeros_like(mesh_X), scale=1,
    #           angles="xy", scale_units='xy', color="#666666")
    # plt.show()
    if time != 0:
        plt.pause(time)
    if path:
        plt.savefig(path)

def splicing(images, row, column, img_size, gap=1, RGB=True):
    """
    input: RGB three channel image
    you can change #创建新图像 to dell with gray img
    """
    if row * column != len(images):
        raise Exception('Img number(%d) is not match with row * column(%d)'
                        % (len(images), row * column))
    height = img_size[0]
    width = img_size[1]
    # 创建空图像
    if RGB:
        target = np.zeros(((height + gap) * row, (width + gap) * column, 3), np.uint8)
    else:
        target = np.zeros(((height + gap) * row, (width + gap) * column), np.uint8)
    target.fill(200)
    # splicing images
    for i in range(row):
        for j in range(column):
            target[i * (height + gap): (i + 1) * height + i * gap, j * (width + gap): (j + 1) * width + j * gap] = \
                images[i * column + j].copy()
            # print(i, row, j, i * row + j)
    return target

def norm_heat(heat, norm_int=(0, 1), dtype=np.float64):
    """

    :param heat: heat map
    :param norm_int: normalization interval
    :param dtype: normed data type
    :return: normalized heat map
    """
    # norm to [0, 1]
    normed_heat = heat.astype(np.float64)
    normed_heat = normed_heat - np.min(normed_heat)
    normed_heat = normed_heat / max(np.max(normed_heat), 1e-10)
    # norm to norm_int
    int_len = norm_int[1] - norm_int[0]
    assert(int_len >= 0)
    normed_heat = (normed_heat + norm_int[0]) * int_len
    return normed_heat.astype(dtype)


def fusion_img(org_img, heat, resize_shape=(0, 0)):
    """

    :param resize_shape: if reasize_shape == (0, 0), outputs' shape is same as org_img
    :param org_img: 原始图像，cv2格式 C * H * W
    :param heat:    heatmap, H * W
    :return: fusion_img
    """
    (H, W, C) = org_img.shape
    org = org_img.astype(np.float64)
    heatmap = norm_heat(heat, norm_int=(0, 255), dtype=np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float64)
    fusion_img = norm_heat(org + heatmap, norm_int=(0, 255), dtype=np.uint8)
    if resize_shape[0] == 0 and resize_shape[1] == 0:
        return fusion_img
    else:
        return cv2.resize(fusion_img, resize_shape, interpolation=cv2.INTER_NEAREST)

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    if num == 0:
        return []
    step = 360.0 / num
    while i < 360:
        if len(hls_colors) == num:
            break
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


if __name__ == '__main__':
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('X')  # 设置x轴标签
    ax.set_ylabel('Y')  # 设置y轴标签
    ax.set_title('Org')
    draw_Normal_3D([0, 1], 2, ax)
    plt.show()
    plt.pause(10)


