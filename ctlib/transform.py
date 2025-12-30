import numpy as np
import torch
from collections import OrderedDict

def to_list(x):
    if type(x) == torch.Tensor:
        return x.detach().cpu().numpy().tolist()
    elif type(x) == np.array:
        return x.tolist()
    elif type(x) == list:
        return [to_array(i) for i in x]
    elif type(x) == tuple:
        return [to_array(i) for i in x]
    elif type(x) == dict:
        return {k: to_array(v) for k, v in x.items()}
    elif type(x) == OrderedDict:
        d = OrderedDict()
        for k, v in x.items():
            d[k] = to_array(v)
        return d
    else:
        return x

def to_array(x):
    if type(x) == torch.Tensor:
        return x.detach().cpu().numpy()
    elif type(x) == np.array:
        return x
    elif type(x) == list:
        return [to_array(i) for i in x]
    elif type(x) == tuple:
        return [to_array(i) for i in x]
    elif type(x) == dict:
        return {k: to_array(v) for k, v in x.items()}
    elif type(x) == OrderedDict:
        d = OrderedDict()
        for k, v in x.items():
            d[k] = to_array(v)
        return d
    else:
        return x

def to_tensor(x):
    if type(x) == torch.Tensor:
        return x
    elif type(x) == np.array:
        return torch.Tensor(x)
    elif type(x) == list:
        return [to_array(i) for i in x]
    elif type(x) == tuple:
        return [to_array(i) for i in x]
    elif type(x) == dict:
        return {k: to_array(v) for k, v in x.items()}
    elif type(x) == OrderedDict:
        d = OrderedDict()
        for k, v in x.items():
            d[k] = to_array(v)
        return d
    else:
        return x

def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = img * std
    img = img + mean
    return img

def auto_denorm(img):
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    img = img.transpose(1, 2, 0)
    img = imdenormalize(img.astype(np.float32),
                        np.array(mean),
                        np.array(std))
    img = img.clip(min=0, max=255)
    img = img.astype(np.uint8)
    out_img = np.ascontiguousarray(img)
    return out_img
