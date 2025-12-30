# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
import mmrotate.registry as mmr_registry
from mmdet.models.necks.fpn import FPN

@mmr_registry.MODELS.register_module()
class AceFPN(FPN):

    def __init__(
        self,
            *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)