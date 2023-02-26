# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

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
from mmdet3d.structures import PointData


@MODELS.register_module()
class TD3DSegmentationHead(BaseModule):
    def __init__(self,
                 voxel_size,
                 multiclass_loss=None,  # todo: add focal loss here
                 binary_loss=None,  # todo: abbd bce losss here
                 train_cfg=None,
                 test_cfg=None):
        super(TD3DSegmentationHead, self).__init__()
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def loss(self,
             x: SparseTensor,
             proposals: List[InstanceData],
             pts_targets: Tensor,
             batch_data_samples: SampleList,
             **kwargs) -> dict:
        
        return dict(
            multiclass_loss=torch.sum(x.features * 0),
            binary_loss=torch.sum(x.features * 0))
    
    def predict(self,
                x: SparseTensor, 
                field: TensorField,
                proposals: List[InstanceData],
                batch_data_samples: SampleList,
                **kwargs) -> List[PointData]:
        results = []
        for _ in range(len(batch_data_samples)):
            results.append(PointData(
                pts_instance_mask=torch.ones(
                    (2, field.coordinates.shape[0]), dtype=torch.bool),
                instance_labels=torch.tensor([1, 2]),
                instance_scores=torch.tensor([0.7, 0.8])))
        return results
