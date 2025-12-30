# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

import torch
from mmengine.dist import all_gather, get_data_device, get_comm_device, \
    cast_data_device, get_default_group, get_world_size
from torch import distributed as torch_dist

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.distributed as dist


def gather_tensors(input_array):
    """Gather tensor from all GPUs."""
    world_size = dist.get_world_size()
    # gather shapes first
    myshape = input_array.shape
    mycount = input_array.size
    shape_tensor = torch.Tensor(np.array(myshape)).cuda()
    all_shape = [
        torch.Tensor(np.array(myshape)).cuda() for i in range(world_size)
    ]
    dist.all_gather(all_shape, shape_tensor)
    # compute largest shapes
    all_shape = [x.cpu().numpy() for x in all_shape]
    all_count = [int(x.prod()) for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    max_count = max(all_count)
    # padding tensors and gather them
    output_tensors = [
        torch.Tensor(max_count).cuda() for i in range(world_size)
    ]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:mycount] = input_array.reshape(-1)
    input_tensor = torch.Tensor(padded_input_array).cuda()
    dist.all_gather(output_tensors, input_tensor)
    # unpadding gathered tensors
    padded_output = [x.cpu().numpy() for x in output_tensors]
    output = [
        x[:all_count[i]].reshape(all_shape[i])
        for i, x in enumerate(padded_output)
    ]
    return output


def gather_tensors_batch(input_array, part_size=100, ret_rank=-1):
    """batch-wise gathering to avoid CUDA out of memory."""
    rank = dist.get_rank()
    all_features = []
    part_num = input_array.shape[0] // part_size + 1 if input_array.shape[
        0] % part_size != 0 else input_array.shape[0] // part_size
    for i in range(part_num):
        part_feat = input_array[i *
                                part_size:min((i + 1) *
                                              part_size, input_array.shape[0]),
                                ...]
        assert part_feat.shape[
            0] > 0, f'rank: {rank}, length of part features should > 0'
        gather_part_feat = gather_tensors(part_feat)
        all_features.append(gather_part_feat)
    if ret_rank == -1:
        all_features = [
            np.concatenate([all_features[i][j] for i in range(part_num)],
                           axis=0) for j in range(len(all_features[0]))
        ]
        return all_features
    else:
        if rank == ret_rank:
            all_features = [
                np.concatenate([all_features[i][j] for i in range(part_num)],
                               axis=0) for j in range(len(all_features[0]))
            ]
            return all_features
        else:
            return None


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def concat_all_gather_diff_size(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Performs all_gather operation on the provided tensors.

    Args:
        tensor (torch.Tensor): Tensor to be broadcast from current process.

    Returns:
        torch.Tensor: The concatnated tensor.
    """


    world_size = get_world_size(group)
    # print('world_size', world_size)
    if world_size <= 1:
        return tensor

    group = get_default_group()
    input_device = get_data_device(tensor)
    backend_device = get_comm_device(group)
    data_on_device = cast_data_device(tensor, backend_device)

    local_size = tensor.size()[0]
    local_size = torch.tensor(local_size, device=input_device)
    all_sizes = [
        torch.empty_like(local_size, device=backend_device)
        for _ in range(world_size)
    ]
    torch_dist.all_gather(all_sizes, local_size)

    max_size = max(all_sizes)

    size_diff = torch.Size([max_size - local_size, *tensor.size()[1:]])
    padding = tensor.new_zeros(size_diff)
    pad_tensor = torch.cat([tensor, padding])

    pad_tensors_gather = all_gather(pad_tensor)

    tensors_gather = []
    for t, size in zip(pad_tensors_gather, all_sizes):
        tensors_gather.append(t[:size])

    output = torch.cat(tensors_gather)
    return output
