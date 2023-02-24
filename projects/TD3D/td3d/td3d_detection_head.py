# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/td3d/blob/master/mmdet3d/models/decode_heads/td3d_instance_head.py # noqa
from typing import List, Optional, Tuple

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

import torch
from mmcv.ops import nms3d_normal
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet3d.models import Base3DDenseHead
from mmdet3d.registry import MODELS
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.utils import InstanceList, OptInstanceList
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class TD3DDetectionHead(Base3DDenseHead):
    r"""Bbox head of `TD3D <https://arxiv.org/abs/2302.02871>`_.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in input tensors.
        voxel_size (float): Voxel size in meters.
        pts_assign_threshold (int): Box to location assigner parameter.
            Assigner selects the maximum feature level with more locations
            inside the box than pts_assign_threshold.
        pts_center_threshold (int): Box to location assigner parameter.
            After feature level for the box is determined, assigner selects
            pts_center_threshold locations closest to the box center.
        bbox_loss (dict): Config of bbox loss. Defaults to
            dict(type='AxisAlignedIoULoss', mode='diou').
        cls_loss (dict): Config of classification loss. Defaults to
            dict = dict(type='mmdet.FocalLoss').
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 voxel_size: int,
                 padding: float,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 bbox_loss: dict = dict(
                     type='TD3DAxisAlignedIoULoss',
                     mode='diou'),
                 cls_loss: dict = dict(
                     type='mmdet.FocalLoss'),
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(TD3DDetectionHead, self).__init__(init_cfg)
        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to install MinkowskiEngine.`'  # noqa: E501
            )
        self.voxel_size = voxel_size
        self.padding = padding
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.bbox_loss = MODELS.build(bbox_loss)
        self.cls_loss = MODELS.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(num_classes, in_channels)

    def _init_layers(self, num_classes: int, in_channels: int):
        """Initialize layers.

        Args:
            in_channels (int): Number of channels in input tensors.
            num_classes (int): Number of classes.
        """
        self.conv_reg = ME.MinkowskiConvolution(
            in_channels, 6, kernel_size=1, bias=True, dimension=3)
        self.conv_cls = ME.MinkowskiConvolution(
            in_channels, num_classes, kernel_size=1, bias=True, dimension=3)

    def init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.conv_reg.kernel, std=.01)
        nn.init.normal_(self.conv_cls.kernel, std=.01)
        nn.init.constant_(self.conv_cls.bias, bias_init_with_prob(.01))

    def _forward_single(self, x: SparseTensor) -> Tuple[Tensor, ...]:
        """Forward pass per level.

        Args:
            x (SparseTensor): Per level neck output tensor.

        Returns:
            tuple[Tensor]: Per level head predictions.
        """
        bbox_pred = torch.exp(self.conv_reg(x).features)
        cls_pred = self.conv_cls(x).features

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, points

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], ...]:
        """Forward pass.

        Args:
            x (list[Tensor]): Features from the backbone.

        Returns:
            Tuple[List[Tensor], ...]: Predictions of the head.
        """
        bbox_preds, cls_preds, points = [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, point = self._forward_single(x[i])
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
        return bbox_preds, cls_preds, points

    def _loss_by_feat_single(self, bbox_preds: List[Tensor],
                             cls_preds: List[Tensor], points: List[Tensor],
                             gt_bboxes: BaseInstance3DBoxes, gt_labels: Tensor,
                             input_meta: dict) -> Tuple[Tensor, ...]:
        """Loss function of single sample.

        Args:
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor, ...]: Bbox and classification loss values and
                assigned indexes.
        """
        num_classes = cls_preds[0].shape[1]
        bbox_targets, cls_targets, target_ids = self.get_targets(
            points, gt_bboxes, gt_labels)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        pos_mask = cls_targets >= 0
        avg_factor = max(pos_mask.sum(), 1)
        cls_targets = torch.where(pos_mask, cls_targets, num_classes)
        cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=avg_factor)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            pos_bbox_targets = bbox_targets[pos_mask]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = pos_bbox_preds
        return bbox_loss, cls_loss, target_ids

    def loss_by_feat(self,
                     bbox_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     points: List[List[Tensor]],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     return_ids: bool = False,
                     **kwargs) -> dict:
        """Loss function about feature.

        Args:
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
                The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_input_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            return_ids (bool): Whether to return assigned indexes.
                Defaults to False.

        Returns:
            dict: Bbox and classification losses.
        """
        bbox_losses, cls_losses, target_ids = [], [], []
        for i in range(len(batch_input_metas)):
            bbox_loss, cls_loss, target_id = self._loss_by_feat_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=batch_input_metas[i],
                gt_bboxes=batch_gt_instances_3d[i].bboxes_3d,
                gt_labels=batch_gt_instances_3d[i].labels_3d)
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            target_ids.append(target_id)
        losses = dict(
            bbox_loss=torch.mean(torch.stack(bbox_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses)))
        if return_ids:
            return losses, target_ids
        return losses

    def _predict_by_feat_single(self, bbox_preds: List[Tensor],
                                     cls_preds: List[Tensor],
                                     points: List[Tensor],
                                     input_meta: dict) -> InstanceData:
        """Generate boxes for single sample during.

        Args:
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            input_meta (dict): Scene meta info.

        Returns:
            InstanceData: Predicted bounding boxes, scores and labels.
        """
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        bboxes = self._bbox_pred_to_bbox(points, bbox_preds)
        bboxes = torch.cat((
            bboxes[:, :3],
            bboxes[:, 3:6] - self.padding,
            bboxes[:, 6:]), dim=1)
        bboxes, scores, labels = self._single_scene_multiclass_nms(
            bboxes, scores, input_meta)

        bboxes = input_meta['box_type_3d'](
            bboxes,
            box_dim=6,
            with_yaw=False,
            origin=(.5, .5, .5))

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        return results

    def predict_by_feat(self, bbox_preds: List[List[Tensor]], cls_preds,
                        points: List[List[Tensor]],
                        batch_input_metas: List[dict],
                        proposal_cfg: Optional[ConfigDict] = None,
                        **kwargs) -> List[InstanceData]:
        """Generate boxes for all scenes.

        Args:
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            batch_input_metas (list[dict]): Meta infos for all scenes.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            list[InstanceData]: Predicted bboxes, scores, and labels for
            all scenes.
        """
        results = []
        for i in range(len(batch_input_metas)):
            result = self._predict_by_feat_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                input_meta=batch_input_metas[i])
            results.append(result)
        return results

    def loss_and_predict(self,
                         x: Tuple[Tensor],
                         batch_data_samples: SampleList,
                         proposal_cfg: Optional[ConfigDict] = None,
                         **kwargs) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each image and
                corresponding annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_input_metas,
                              batch_gt_instances_ignore)
        losses, target_ids = self.loss_by_feat(
            *loss_inputs, return_ids=True)

        bbox_preds, cls_preds, points = outs
        results = []
        for i in range(len(batch_input_metas)):
            bbox_pred = torch.cat([x[i] for x in bbox_preds])
            score = torch.cat([x[i] for x in cls_preds]).sigmoid()
            point = torch.cat([x[i] for x in points])
            input_meta = batch_input_metas[i]
            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            bboxes = torch.cat((
                bboxes[:, :3],
                bboxes[:, 3:6] - self.padding,
                bboxes[:, 6:]), dim=1)

            bboxes = input_meta['box_type_3d'](
                bboxes,
                box_dim=6,
                with_yaw=False,
                origin=(.5, .5, .5))

            result = InstanceData()
            result.bboxes_3d = bboxes
            result.scores_3d = score
            result.labels_3d = target_ids[i]
            results.append(result)
        return losses, results

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned iou loss format.

        Args:
            bbox (Tensor): 3D box of shape (N, 6).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6).
        """
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
            points (Tensor): Final locations of shape (N, 3).
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6).
        
        Returns:
            Tensor: Transformed 3D box of shape (N, 6).
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
        return base_bbox
    
    @staticmethod
    def _get_face_distances(points: Tensor, boxes: Tensor) -> Tensor:
        """Compute distances from points to box faces.

        Args:
            points (Tensor): Final locations.
            boxes (Tensor): Ground truth boxes of shape (..., 6).
        
        Returns:
            Tesnor: Distances from points to box faces of shape (..., 6).

        """
        dx_min = points[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - points[..., 0]
        dy_min = points[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - points[..., 1]
        dz_min = points[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - points[..., 2]
        return torch.stack((
            dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)

    @torch.no_grad()
    def get_targets(self, points: Tensor, gt_bboxes: BaseInstance3DBoxes,
                    gt_labels: Tensor) -> Tuple[Tensor, ...]:
        """Compute targets for final locations for a single scene.

        Args:
            points (list[Tensor]): Final locations for all levels.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.

        Returns:
            tuple[Tensor, ...]: Bbox, classification, and id
                targets for all locations.
        """
        float_max = points[0].new_tensor(1e8)
        n_levels = len(points)
        levels = torch.cat([
            points[i].new_tensor(i).expand(len(points[i]))
            for i in range(len(points))
        ])
        points = torch.cat(points)
        gt_bboxes = gt_bboxes.to(points.device)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.unsqueeze(0).expand(n_points, n_boxes)

        # condition 1: point inside enlarged box
        boxes = torch.cat((
            gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:6] + self.padding),
            dim=1)
        boxes = boxes.expand(n_points, n_boxes, 6)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        face_distances = self._get_face_distances(points, boxes)
        inside_box_condition = face_distances.min(dim=-1).values > 0

        # condition 2: positive points per level >= limit
        # calculate positive points per scale
        n_pos_points_per_level = []
        for i in range(n_levels):
            n_pos_points_per_level.append(
                torch.sum(inside_box_condition[levels == i], dim=0))
        # find best level
        n_pos_points_per_level = torch.stack(n_pos_points_per_level, dim=0)
        lower_limit_mask = n_pos_points_per_level < self.pts_assign_threshold
        lower_index = torch.argmax(lower_limit_mask.int(), dim=0) - 1
        lower_index = torch.where(lower_index < 0, 0, lower_index)
        all_upper_limit_mask = torch.all(
            torch.logical_not(lower_limit_mask), dim=0)
        best_level = torch.where(all_upper_limit_mask, n_levels - 1,
                                 lower_index)
        # keep only points with best level
        best_level = best_level.expand(n_points, n_boxes)
        levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = best_level == levels

        # condition 3: limit topk points per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        center_distances = torch.where(
            inside_box_condition, center_distances, float_max)
        center_distances = torch.where(
            level_condition, center_distances, float_max)
        topk_distances = torch.topk(center_distances,
                                    min(self.pts_center_threshold + 1,
                                        len(center_distances)),
                                    largest=False, dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 4: min volume box per point
        volumes = torch.where(inside_box_condition, volumes, float_max)
        volumes = torch.where(level_condition, volumes, float_max)
        volumes = torch.where(topk_condition, volumes, float_max)
        min_volumes, min_inds = volumes.min(dim=1)

        bbox_targets = boxes[torch.arange(n_points), min_inds]
        cls_targets = gt_labels[min_inds]
        cls_targets = torch.where(min_volumes == float_max, -1, cls_targets)
        target_ids = torch.where(min_volumes == float_max, -1, min_inds)
        return bbox_targets, cls_targets, target_ids

    def _single_scene_multiclass_nms(self, bboxes: Tensor, scores: Tensor,
                                     input_meta: dict) -> Tuple[Tensor, ...]:
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            input_meta (dict): Scene meta data.

        Returns:
            tuple[Tensor, ...]: Predicted bboxes, scores and labels.
        """
        num_classes = scores.shape[1]
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(num_classes):
            ids = scores[:, i] > self.test_cfg.det_score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            class_bboxes = torch.cat(
                (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                dim=1)
            nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        nms_bboxes = nms_bboxes[:, :6]
        return nms_bboxes, nms_scores, nms_labels
