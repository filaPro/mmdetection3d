# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Optional

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
class TD3DHead(BaseModule):
    def __init__(self,
                 n_classes: int,
                 in_channels: int,
                 n_levels: int,
                 n_reg_outs: int,
                 voxel_size: int,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None):
        super(TD3DHead, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.n_levels = n_levels
        self.n_reg_outs = n_reg_outs
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to install MinkowskiEngine.`'  # noqa: E501
            )




    def loss(self,
        x: Tuple[SparseTensor],
        pts_targets: Tensor,
        batch_data_samples: SampleList,
        **kwargs) -> dict:
        
        #first stage
        bbox_preds, cls_preds, locations = self._forward_first(x[1:])
        losses = self._loss_first(bbox_preds, cls_preds, locations, 
                            gt_bboxes, gt_labels, img_metas)



        return {'loss': torch.sum(x[0].features * 0)}

    def _forward_first(self, x):
        reg_preds, cls_preds, locations = [], [], []
        for i in range(len(x)):
            reg_pred, cls_pred, point = self._forward_first_single(x[i])
            reg_preds.append(reg_pred)
            cls_preds.append(cls_pred)
            locations.append(point)
        return reg_preds, cls_preds, locations

    # per level
    def _forward_first_single(self, x):
        reg_pred = torch.exp(self.reg_conv(x).features)
        cls_pred = self.cls_conv(x).features

        reg_preds, cls_preds, locations = [], [], []
        for permutation in x.decomposition_permutations:
            reg_preds.append(reg_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            locations.append(x.coordinates[permutation][:, 1:] * self.voxel_size)
        return reg_preds, cls_preds, locations

    def _loss_first(self, bbox_preds, cls_preds, points,
              gt_bboxes, gt_labels, img_metas):
        bbox_losses, cls_losses = [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss = self._loss_first_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(bbox_loss=torch.mean(torch.stack(bbox_losses)),
                    cls_loss=torch.mean(torch.stack(cls_losses)))

    # per scene
    def _loss_first_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):

        assigned_ids = self.first_assigner.assign(points, gt_bboxes, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0
        cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        avg_factor = max(pos_mask.sum(), 1)
        cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=avg_factor)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            pos_bbox_targets = torch.cat((
                pos_bbox_targets[:, :3],
                pos_bbox_targets[:, 3:6] + self.padding,
                pos_bbox_targets[:, 6:]), dim=1)
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = pos_bbox_preds.sum()
        return bbox_loss, cls_loss

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
                bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)





































    def predict(self, 
        x: SparseTensor, 
        field: TensorField, 
        batch_data_samples: SampleList,
        **kwargs) -> Tuple:
        pass
