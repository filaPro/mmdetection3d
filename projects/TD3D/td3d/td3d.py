# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/td3d/blob/master/mmdet3d/detectors/td3d_instance_segmentor.py # noqa
from typing import List, Union

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import (
        SparseTensor, SparseTensorQuantizationMode, TensorField)
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = SparseTensorQuantizationMode = TensorField = None
    pass

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.models import Base3DDetector


@MODELS.register_module()
class TD3D(Base3DDetector):
    r"""`TD3D <https://arxiv.org/abs/2302.02871>`_.
   
    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        seg_head (dict, optional): Config dict of instance segmentation head.
            Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 seg_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(TD3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to install MinkowskiEngine.`'  # noqa: E501
            )
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        seg_head.update(train_cfg=train_cfg)
        seg_head.update(test_cfg=test_cfg)
        self.seg_head = MODELS.build(seg_head)
        self.voxel_size = bbox_head['voxel_size']
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    def extract_feat(self, x: SparseTensor) -> List[SparseTensor]:
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor.
        
        Returns:
            List[SparseTensor]: Features after backbone and neck.
        """
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def collate(self, points: List[Tensor],
                quantization_mode: SparseTensorQuantizationMode) \
                -> TensorField:
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.
        
        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            dtype=points[0].dtype,
            device=points[0].device)
        return ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=quantization_mode,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=points[0].device)
    
    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector"""
        pass
    
    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        annotated_points = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            annotated_points.append(torch.cat((
                batch_inputs_dict['points'][i],
                gt_pts_seg.pts_instance_mask.unsqueeze(1),
                gt_pts_seg.pts_semantic_mask.unsqueeze(1)), dim=1))

        field = self.collate(
            annotated_points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        x = field.sparse()
        pts_targets = x.features[:, 3:].round().long()
        x = ME.SparseTensor(
            x.features[:, :3],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        x = self.extract_feat(x)
        losses, proposals = self.bbox_head.loss_and_predict(
            x[1:], batch_data_samples, **kwargs)
        seg_losses = self.seg_head.loss(
            x[0], proposals, pts_targets, batch_data_samples, **kwargs)
        losses.update(seg_losses)
        return losses
    
    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.

                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.

        """
        points = batch_inputs_dict['points']
        field = self.collate(
            points, ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        x = field.sparse()
        x = self.extract_feat(x)
        proposals = self.bbox_head.predict(
            x[1:], batch_data_samples, **kwargs)
        results_list = self.seg_head.predict(
            x, field, proposals, batch_data_samples, **kwargs)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples
