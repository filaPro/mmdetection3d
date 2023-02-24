# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor, TensorField
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = TensorField = None
    pass

import torch
from torch import Tensor

from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class TD3DSegmentationHead(BaseModule):
    def __init__(self, voxel_size, train_cfg=None, test_cfg=None):
        super(TD3DSegmentationHead, self).__init__()
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def loss(self,
        x: Tuple[SparseTensor],
        pts_targets: Tensor,
        batch_data_samples: SampleList,
        **kwargs) -> dict:
        
        return {'loss': torch.sum(x[0].features * 0)}
    
    def predict(self, 
        x: SparseTensor, 
        field: TensorField, 
        batch_data_samples: SampleList,
        **kwargs) -> Tuple:
        pass
