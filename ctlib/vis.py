import cv2
import numpy as np
import cv2
import colorsys
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def lable_score_info(labels, scores):
    out_strs = []
    assert len(labels) == len(scores)
    for l, s in zip(labels, scores):
        info = '%d_%.3f' % (l, s)
        out_strs.append(info)
    return out_strs

def score_info(scores):
    out_strs = []
    for s in scores:
        info = '%.3f' % s
        out_strs.append(info)
    return out_strs

def draw_polys(img,
              polys,
              texts=[],
              thickness=1,
              color=(0, 200, 200),
               text_color=(0, 200, 200),
              with_text=False,
              fill_poly=False):
    if len(texts) != len(polys):
        assert not with_text
        texts = ['' for i in range(len(polys))]
    for poly, text in zip(polys, texts):
        draw_poly(img, poly, text, thickness, color, text_color, with_text, fill_poly)

def draw_poly(img,
              poly,
              text='',
              thickness=1,
              color=(0, 200, 200), text_color=(0, 200, 200),
              with_text=False,
              fill_poly=False):
    # points = np.array([[910, 650], [206, 650], [458, 500], [696, 500]])
    points = np.array(poly, dtype=np.int32).reshape(-1, 2)
    # img = img.copy()
    if fill_poly:
        cv2.fillConvexPoly(img, points.reshape(-1, 2), color)
    else:
        cv2.polylines(img, [points], 1, color, thickness=thickness)
    if with_text:
        cv2.putText(img, text,
                    (int(points[0][0]), int(points[0][1]) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    text_color,
                    thickness=1)


def draw_bboxs(img,
              bboxes,
              texts='',
              thickness=1,
              color=(200, 200, 0), text_color=(200, 200, 0),
              with_text=False):
    if len(texts) != len(bboxes):
        assert not with_text
        texts = ['' for i in range(len(bboxes))]
    for bbox, text in zip(bboxes, texts):
        draw_bbox(img, bbox, text, thickness, color, text_color, with_text)

def draw_bbox(img,
              bbox,
              text='',
              thickness=1,
              color=(200, 200, 0), text_color=(200, 200, 0),
              with_text=False):
    (x1, y1, x2, y2) = bbox
    (x1, y1) = (int(x1), int(y1))
    (x2, y2) = (int(x2), int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
    if with_text:
        cv2.putText(img, text,
                    (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.3, text_color,
                    thickness=1)

def draw_hist(states,
              num_bins=100,
              hist_range=(0, 200),
              plot_x_lim=(0, 100),
              plot_y_lim=(0, 100),
              x_ticks=[0, 50, 100],
              y_ticks=[0, 50, 100],
              y_tick_names=['0', '50', '100'],
              out_pth='./hist.png',
              description='box_area',
              fig_size=(24, 6),
              dpi=600,
              fontsize=36):
    """

    :param states:      待统计的值
    :param num_bins:    柱状图bin个数
    :param hist_range:  hist统计的范围
    :param plot_x_lim:      绘图的x范围：(min_x_lim, max_x_lim)
    :param plot_y_lim:      绘图的y范围：(min_x_lim, max_x_lim)
    :param x_ticks:      [int]，  显示的x坐标，并且也是x坐标名字
    :param y_ticks:      [float]，显示的y坐标
    :param y_tick_names: [str]，  显示的y坐标名字
    :param out_pth:      输出路径
    :param description:  值的名字，title的名字
    :param fig_size:
    :param dpi:
    :param fontsize:
    :return:
    """
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = fig_size
    (hist_min_x, hist_max_x) = hist_range
    (min_x_lim, max_x_lim) = plot_x_lim
    (min_y_lim, max_y_lim) = plot_y_lim

    hist, bins = np.histogram(states, bins=num_bins,
                              range=(hist_min_x,
                                     hist_max_x))
    min_s = np.min(states)
    max_s = np.max(states)
    print('%s (real x_lim) in (%.3f, %.3f)' % (description, min_s, max_s))
    print('Hist of %s (real y_lim) in (%.3f, %.3f)' % (description,
                                                       np.min(hist),
                                                       np.max(hist)))

    plt.xlim(min_x_lim, max_x_lim)
    tick_names = [str(x) for x in x_ticks]
    plt.xticks(x_ticks, tick_names, fontsize=fontsize)


    plt.ylim(min_y_lim, max_y_lim)
    tick_names = [str(s) for s in y_tick_names]
    plt.yticks(y_ticks, tick_names, fontsize=fontsize)

    width = bins[1] - bins[0]
    # [1, 5] -> [1 + 2,  5 + 2] = [3, 7] -> [3]
    bin_ctrs = (bins + width / 2)[:len(bins) - 1]
    plt.bar(x=bin_ctrs,
            height=hist, width=width,
            alpha=1, edgecolor='k',
            color=np.array((48, 118, 182)) / 255)
    plt.savefig(out_pth)
    plt.close()
    print('Save: %s' % out_pth)

def simple_hist(states,
                num_bins=100,
                plot_x_lim=None,
                out_pth='./hist.png',
                description='box_area',
                norm_hist=False,
                y_lim=None,
                fig_size=(24, 6),
                dpi=600,
                fontsize=36):
    """

    :param states:      待统计的值
    :param num_bins:    柱状图bin个数
    :param hist_range:  hist统计的范围
    :param plot_x_lim:      绘图的x范围：(min_x_lim, max_x_lim)
    :param plot_y_lim:      绘图的y范围：(min_x_lim, max_x_lim)
    :param x_ticks:      [int]，  显示的x坐标，并且也是x坐标名字
    :param y_ticks:      [float]，显示的y坐标
    :param y_tick_names: [str]，  显示的y坐标名字
    :param out_pth:      输出路径
    :param description:  值的名字，title的名字
    :param fig_size:
    :param dpi:
    :param fontsize:
    :return:
    """
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = fig_size

    if plot_x_lim is not None:
        (min_x_lim, max_x_lim) = plot_x_lim
        plt.xlim(min_x_lim, max_x_lim)

    min_x_lim = np.min(states)
    max_x_lim = np.max(states)

    hist, bins = np.histogram(states, bins=num_bins,
                              range=(min_x_lim,
                                     max_x_lim))
    if norm_hist:
        hist = hist / len(states)
    min_y_lim = np.min(hist)
    max_y_lim = np.max(hist)

    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])

    min_s = np.min(states)
    max_s = np.max(states)
    print('$'*50)
    print('%s (real x_lim) in (%.3f, %.3f)' % (description, min_s, max_s))
    print('Hist of %s (real y_lim) in (%.3f, %.3f)' % (description, min_y_lim, max_y_lim))

    width = bins[1] - bins[0]
    # [1, 5] -> [1 + 2,  5 + 2] = [3, 7] -> [3]
    bin_ctrs = (bins + width / 2)[:len(bins) - 1]
    plt.bar(x=bin_ctrs,
            height=hist, width=width,
            alpha=1, edgecolor='k',
            color=np.array((48, 118, 182)) / 255)
    plt.title('%s in : m=%.3f, [%.3f, %.3f]' % (description, float(np.mean(states)), min_s, max_s))
    plt.savefig(out_pth)
    plt.close()
    print('Save: %s' % out_pth)
    print('$'*50)

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
def norm_colors(colors):
    """
    (a, b, c) -> ele in [0, 1]
    :param colors: [(a, b, c)]
    :return:
    """
    r_c = []
    def norm(c):
        c = np.array(c)
        if np.max(c) == 0:
            return list(c)
        else:
            return list(c/np.max(c))

    for c in colors:
        c = norm(c)
        r_c.append(c)
    return r_c


def plot_embedding_2d(X, labels, save_pth):
    """Plot an embedding X with the class label y colored by the domain d."""
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.rcParams['figure.dpi'] = 400
    plt.figure(figsize=(24, 24))

    # Start from zero
    max_labels = max(set(labels))

    colors = ncolors(max_labels + 1)
    colors = norm_colors(colors)

    plt.xlim(np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1)
    plt.ylim(np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1)
    for i, c in tqdm(list(enumerate(colors))):
        x = X[labels == i]
        for j in range(x.shape[0]):
            # print(j, c, ys[i], x[j])
            # plot colored number
            plt.text(x[j, 0], x[j, 1], str(i),
                     color=c,
                     fontdict={'weight': 'bold', 'size': 4})

    plt.savefig(save_pth)




