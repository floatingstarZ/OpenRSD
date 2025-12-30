import numpy as np
import torch
import inspect

def all_to_list(data):
    if isinstance(data, torch.Tensor):
        return data.numpy().tolist()
    elif isinstance(data, list):
        return data
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        d = data.copy()
        for k, v in d.items():
            d[k] = all_to_numpy(v)
        return d
    else:
        raise Exception('Data with type %s dose not support numpy transform' % (str(type(data))))



def all_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, list):
        for i, v in enumerate(data):
            data[i] = all_to_numpy(v)
        return data
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, dict):
        d = data.copy()
        for k, v in d.items():
            d[k] = all_to_numpy(v)
        return d
    else:
        return data
        # raise Exception('Data with type %s dose not support numpy transform' % (str(type(data))))


def to_bbox_type(bboxes, dtype=torch.FloatTensor):
    """
    :param bboxes: list, numpy, tensor
    :return: bbox: n x m, all to float tensor
    """
    bboxes = dtype(bboxes)
    shape = bboxes.shape
    assert len(shape) <= 2

    if len(shape) == 1:
        bboxes = bboxes.unsqueeze(0)
    return bboxes


if __name__ == '__main__':
    # pass
    # # test to_bbox_type
    a = torch.Tensor([1,2, 3, 4])
    print(to_bbox_type(a))
    a = torch.Tensor([[1,2, 3, 4]])
    print(to_bbox_type(a))
    a = np.array([1,2, 3, 4])
    print(to_bbox_type(a))
    a = np.array([[1,2, 3, 4]])
    print(to_bbox_type(a))
    a = np.array([])
    print(to_bbox_type(a))
    a = [1, 2, 3, 4]
    print(to_bbox_type(a))
    a = []
    print(to_bbox_type(a).shape)
    a = [[1,2,3,4]]
    print(to_bbox_type(a))
