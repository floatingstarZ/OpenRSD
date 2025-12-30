import numpy as np
import torch
def split_data(data):
    """

    :param data: 36 * H * W
    :return: 
    """
    x1, y1 = data[0:32 + 1:4], data[1:32 + 2:4]
    x2, y2 = data[2:32 + 3:4], data[3:32 + 4:4]
    return x1, y1, x2, y2

def merge_data(x1, y1, x2, y2):
    (_, H, W) = x1.shape
    data = np.zeros([36, H, W])
    data[0:32 + 1:4] = x1
    data[1:32 + 2:4] = y1
    data[2:32 + 3:4] = x2
    data[3:32 + 4:4] = y2
    return data

def value2x1y1x2y2offset(reg):
    [x1, y1, x2, y2] = split_data(reg)
    _, H, W = reg.shape
    # the location of points
    xl = np.arange(0, W, 1)
    yl = np.arange(0, H, 1)
    Xl, Yl = np.meshgrid(xl, yl)
    x1 = -x1 + Xl
    y1 = -y1 + Yl
    # x1 = x1 - XC
    # y1 = y1 - YC
    x2 = x2 - Xl
    y2 = y2 - Yl
    return merge_data(x1, y1, x2, y2)

def value2xyhwoffset(reg):
    [x1, y1, x2, y2] = split_data(reg)
    _, H, W = reg.shape
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    # the location of points
    xl = np.arange(0, W, 1)
    yl = np.arange(0, H, 1)
    Xl, Yl = np.meshgrid(xl, yl)
    xc = xc - Xl
    yc = yc - Yl
    # 大小除以4
    w = w / 4
    h = h / 4
    return merge_data(xc, yc, w, h)


def fovea_value2x1y1x2y2offset(reg):
    [x1, y1, x2, y2] = split_data(reg)
    _, H, W = reg.shape
    xc = np.arange(0, W, 1)
    yc = np.arange(0, H, 1)
    XC, YC = np.meshgrid(xc, yc)
    x1 = -x1 + XC
    y1 = -y1 + YC
    x2 = x2 - XC
    y2 = y2 - YC
    data = merge_data(x1, y1, x2, y2)
    return np.log((data + 0.5) / 4)

def reverse_value2x1y1x2y2offset(reg):

    [x1, y1, x2, y2] = split_data(reg)
    _, H, W = reg.shape
    xc = np.arange(0, W, 1)
    yc = np.arange(0, H, 1)
    XC, YC = np.meshgrid(xc, yc)
    x1 = -x1 + XC
    y1 = -y1 + YC
    x2 = x2 + XC
    y2 = y2 + YC
    return merge_data(x1, y1, x2, y2)

def reverse_off_sets_results(result):
    """
    
    :param result: Tensor 4 * H * W
    :return: 
    """
    _, H, W = result.shape
    # result = result * 4
    x1, y1 = result[0:1], result[1:2]
    x2, y2 = result[2:3], result[3:4]
    # torch的meshgrid和numpy的相反，x代表i（行），y代表j
    xc = torch.arange(0, W, 1).float()
    yc = torch.arange(0, H, 1).float()
    YC, XC = torch.meshgrid(xc, yc)
    x1 = -x1 + XC
    y1 = -y1 + YC
    x2 = x2 + XC
    y2 = y2 + YC
    return torch.cat([x1, y1, x2, y2], dim=0)


def reverse_xywh_off_sets_results(result):
    """

    :param result: Tensor 4 * H * W
    :return: 
    """
    _, H, W = result.shape
    result = result
    xc_off, yc_off = result[0:1], result[1:2]
    w, h = result[2:3], result[3:4]
    # torch的meshgrid和numpy的相反，x代表i（行），y代表j
    xp = torch.arange(0, W, 1).float()
    yp = torch.arange(0, H, 1).float()
    YP, XP = torch.meshgrid(xp, yp)
    xc = xc_off + XP
    yc = yc_off + YP
    w = w * 4  #  * 1.33
    h = h * 4  #  * 1.33
    return torch.cat([xc-w/2, yc-h/2, xc+w/2, yc+h/2], dim=0)


def reverse_fovea_off_sets_results(result):
    """

    :param result: Tensor 4 * H * W
    :return: 
    """
    _, H, W = result.shape
    result = torch.exp(result) * 4 - 0.5


    x1, y1 = result[0:1], result[1:2]
    x2, y2 = result[2:3], result[3:4]
    # torch的meshgrid和numpy的相反，x代表i（行），y代表j
    xc = torch.arange(0, W, 1).float()
    yc = torch.arange(0, H, 1).float()
    YC, XC = torch.meshgrid(xc, yc)
    x1 = -x1 + XC
    y1 = -y1 + YC
    x2 = x2 + XC
    y2 = y2 + YC
    return torch.cat([x1, y1, x2, y2], dim=0)

if __name__ == '__main__':
    pass




