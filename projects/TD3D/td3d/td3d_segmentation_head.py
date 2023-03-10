# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Optional

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
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.structures import PointData
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.structures import AxisAlignedBboxOverlaps3D

@MODELS.register_module()
class TD3DSegmentationHead(BaseModule):
    r"""Refinement head (second stage) of 
            `TD3D <https://arxiv.org/abs/2302.02871>`_.
    Args:
        n_classes (int): Number of classes.
        voxel_size (float): Voxel size in meters.
        target_iou_thr (float): IoU threshold for
        iou assigner. 
        padding (float): Padding for roi extractor.
        min_pts_threshold (int): Min number of points
        in proposals after roi extractor.
        unet (dict): Config of unet for binary segmentation.
        multiclass_loss (dict): Config of semantic segmentation loss. 
        Defaults to dict(type='mmdet.FocalLoss').
        binary_loss (dict): Config of binary segmentation loss.
        Defaults to dict = dict(type='mmdet.CrossEntropyLoss',
                             use_sigmoid=True).
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
    """
    def __init__(self,
                 n_classes: int,
                 voxel_size: int,
                 target_iou_thr: float,
                 padding: float,
                 min_pts_threshold: int,
                 unet: Optional[dict],
                 multiclass_loss: Optional[dict] = dict(type='mmdet.FocalLoss'),
                 binary_loss: Optional[dict] = dict(type='mmdet.CrossEntropyLoss', 
                                                    use_sigmoid=True),
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None):
        super(TD3DSegmentationHead, self).__init__()
        self.voxel_size = voxel_size
        self.target_iou_thr = target_iou_thr
        self.padding = padding
        self.n_classes = n_classes
        self.min_pts_threshold = min_pts_threshold
        self.unet = MODELS.build(unet)
        self.multiclass_loss =  MODELS.build(multiclass_loss)
        self.binary_loss = MODELS.build(binary_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self,
             x: SparseTensor,
             proposals: List[InstanceData],
             pts_targets: Tensor,
             batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Loss function.

        Args:
            x (SparseTensor): Input feature map for proposals generation.
            proposals (List[InstanceData]): Predicted bounding boxes for all
                scenes in a mini-batch. Proposals[i].labels_3d contains assigned 
                idxs for gt bboxes from the first stage.
            pts_targets (Tensor): voxel-wise instance and semantic markup. 
            batch_data_samples (SampleList): Batch of meta info.

        Returns:
            dict: instance and semantic losses.
        """

        assigned_bboxes = []
        for i in range(len(batch_data_samples)):
            assigned_idxs = proposals[i].labels_3d
            bboxes = proposals[i].bboxes_3d[assigned_idxs != -1]
            assigned_idxs = assigned_idxs[assigned_idxs != -1]

            assigned_iou_idxs = self.get_targets(bboxes,
                                    batch_data_samples[i].gt_instances_3d.bboxes_3d)
            
            assigned_idxs[assigned_idxs != assigned_iou_idxs] = -1
            bboxes = bboxes[assigned_idxs != -1]
            assigned_idxs = assigned_idxs[assigned_idxs != -1]

            if len(bboxes) != 0:
                assigned_idxs_one_hot = torch.nn.functional.one_hot(assigned_idxs)
                mask, idxs = torch.topk(assigned_idxs_one_hot, 
                                        min(self.train_cfg.num_proposals, 
                                            len(bboxes)), 0)
                box_type = batch_data_samples[i].box_type_3d
                sampled_bboxes = box_type(bboxes.tensor[idxs].view(-1, 
                                                            bboxes.tensor.shape[1]))
                sampled_assigned_idxs = assigned_idxs[idxs].view(-1)
                mask = mask.view(-1).bool()
                assigned_bboxes.append((sampled_bboxes[mask],
                                        sampled_assigned_idxs.new_zeros(len(sampled_bboxes[mask])),
                                        sampled_assigned_idxs[mask]))
            else:
                assigned_bboxes.append((bboxes, 
                                        assigned_idxs.new_zeros(0),                      
                                        assigned_idxs))

        cls_preds, targets, v2r, r2scene, rois, _ , assigned_idxs = self._forward(x, pts_targets, assigned_bboxes)
        loss = self._loss(cls_preds, targets, v2r, 
                          r2scene, rois, assigned_idxs, 
                          batch_data_samples)

        return loss

    def get_targets(self, rois: BaseInstance3DBoxes, 
                    gt_bboxes: BaseInstance3DBoxes) -> Tensor:
        """Compute targets for rois for a single scene.
        
        Args:
            rois (BaseInstance3DBoxes): predicted rois.
            gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
        
        Returns:
            Tensor: id targets for all rois.
        """
        overlaps = AxisAlignedBboxOverlaps3D()
        ious = overlaps(torch.unsqueeze(self._bbox_to_loss(rois.tensor), 0), 
                        torch.unsqueeze(self._bbox_to_loss(gt_bboxes.tensor), 
                                        0)).squeeze(0)
        values, indices = ious.max(dim=1)
        indices = torch.where(values > self.target_iou_thr, indices, -1)
        return indices

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


    def _forward(self, 
                 x: SparseTensor, 
                 targets: Tensor, 
                 proposals: List[BaseInstance3DBoxes]) -> Tuple:
        """Extract proposals and pass them through unet.
        
        Args:
            x (SparseTensor): Input feature map for proposals generation.
            pts_targets (Tensor): Voxel-wise instance and semantic markup.
            proposals (List[BaseInstance3DBoxes]): Predicted bounding boxes for all
                scenes in a mini-batch.
        
        Returns:
            preds (Tensor): Extracted feature maps for all valid proposals.
            targets (Tensor): Extracted voxel-wise instance and semantic markup 
                for all valid proposals.
            v2r (Tensor): voxel to roi mapping.
            r2scene (Tensor): roi to scene mapping.
            boxes (List[BaseInstance3DBoxes]): valid boxes for all
                scenes in a mini-batch.
            scores (List[Tensor]): valid scores for all
                scenes in a mini-batch.
            labels (List[Tensor]): valid labels for all
                scenes in a mini-batch.
        """      
        rois = [p[0] for p in proposals]
        feats_with_targets = ME.SparseTensor(torch.cat((x.features, 
                                                        targets), axis=1), 
                                                        x.coordinates)
        tensor, r2scene, valid_roi_idxs = self.extract(feats_with_targets, rois)

        if tensor.features.shape[0] == 0:
            return (x.features.new_zeros((0, 1)),
                    targets.new_zeros((0, 1)),
                    targets.new_zeros(0),
                    targets.new_zeros(0),
                    [targets.new_zeros((0, 7)) for i in range(len(proposals))],
                    [targets.new_zeros(0) for i in range(len(proposals))],
                    [targets.new_zeros(0) for i in range(len(proposals))])

        feats = ME.SparseTensor(tensor.features[:, :-2], tensor.coordinates)
        targets = tensor.features[:, -2:]
        v2r = feats.coordinates[:, 0].long()
        preds = self.unet(feats).features

        boxes = [p[0][mask] for p, mask in zip(proposals, valid_roi_idxs)]
        scores = [p[1][mask] for p, mask in zip(proposals, valid_roi_idxs)]
        labels = [p[2][mask] for p, mask in zip(proposals, valid_roi_idxs)]

        return preds, targets, v2r, r2scene, boxes, scores, labels

    def extract(self, x: SparseTensor, 
                bboxes: List[BaseInstance3DBoxes]) -> Tuple:
        """Extract features from x for each bbox (proposal) in bboxes.
        
        Args:
            x (SparseTensor): Input feature map for proposals generation.
            bboxes (List[BaseInstance3DBoxes]): Predicted bounding boxes 
                    for one scene.
        
        Returns:
            new_tensor (SparseTensor): Extracted feature maps for 
                all valid proposals.
            targets (Tensor): Extracted voxel-wise instance and semantic markup 
                for all valid proposals.
            r2scene (Tensor): roi to scene mapping.
            valid_roi_idxs (List[Tensor]): valid boxes idxs after 
                feature extraction.
        """
        rois = []
        for bbox in bboxes:
            rois.append(torch.cat((
                            bbox.gravity_center,
                            bbox.tensor[:,  3:6] + self.padding), 
                            dim=1))

        new_coordinates, new_features, valid_roi_idxs, ids = [], [], [], []
        n_rois = 0
        for i, (coordinates, features) in enumerate(
            zip(*x.decomposed_coordinates_and_features)):
                        
            if len(rois[i]) == 0:
                valid_roi_idxs.append(features.new_zeros(0, 
                                                         dtype=torch.bool))
                new_coordinates.append(coordinates.new_zeros((0, 4)))
                new_features.append(features.new_zeros((0, 
                                                        features.shape[1])))
            else:
                n_points = len(coordinates)
                points = coordinates * self.voxel_size
                points = points.unsqueeze(1).expand(n_points, len(rois[i]), 3)
                rois_exp = rois[i].unsqueeze(0).expand(n_points, len(rois[i]), 
                                                       rois[i].shape[1])
                face_distances = self.get_face_distances(points, rois_exp)
                inside_condition = face_distances.min(dim=-1).values > 0
                min_pts_condition = inside_condition.sum(dim=0) > self.min_pts_threshold
                inside_condition = inside_condition[:, min_pts_condition]
                nonzero = torch.nonzero(inside_condition)
                new_coordinate = coordinates[nonzero[:, 0]]
                new_index = nonzero[:, 1] + n_rois            
                
                new_coordinate = torch.cat((
                    new_index.unsqueeze(1), new_coordinate), dim=1)
                new_coordinates.append(new_coordinate)
                new_features.append(features[nonzero[:, 0]])
                valid_roi_idxs.append(min_pts_condition)
                n_valid_rois = min_pts_condition.sum()
                n_rois += n_valid_rois
                ids += [i] * n_valid_rois

        new_tensor = ME.SparseTensor(
            torch.cat(new_features),
            torch.cat(new_coordinates).float(),
            tensor_stride=x.tensor_stride)
        r2scene = x.coordinates.new_tensor(ids)
        
        return new_tensor, r2scene, valid_roi_idxs
 
    @staticmethod
    def get_face_distances(points: Tensor, boxes: Tensor) -> Tensor:
        """Compute distances from points to box faces.
        Args:
            points (Tensor): Locations.
            boxes (Tensor): Boxes of shape (..., 6).
        
        Returns:
            Tesnor: Distances from points to box faces of shape (..., 6).
        """
        dx_min = points[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - points[..., 0]
        dy_min = points[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - points[..., 1]
        dz_min = points[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - points[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)

    def _loss(self, cls_preds: Tensor, targets: Tensor, 
              v2r: Tensor, r2scene: Tensor, 
              rois: List[BaseInstance3DBoxes], gt_idxs: List[Tensor], 
              batch_data_samples: SampleList) -> dict:
        """Loss function about binary segmentation per mini-batch.
        
        Args:
            cls_preds (Tensor): Extracted feature maps for all valid proposals.
            targets (Tensor): Extracted voxel-wise instance and semantic markup 
                for all valid proposals.
            v2r (Tensor): voxel to roi mapping.
            r2scene (Tensor): roi to scene mapping.
            rois (List[BaseInstance3DBoxes]): valid boxes for all
                scenes in a mini-batch.
            gt_idxs (List[Tensor]): bbox targets for rois for all
                scenes in a mini-batch.
            batch_data_samples (SampleList): Batch of meta info.
        
        Returns:
            dict: instance and semantic losses. 
        """
        v2scene = r2scene[v2r]
        inst_losses = []
        seg_losses = []
        for i in range(len(batch_data_samples)):
            inst_loss, seg_loss = self._loss_single(
                cls_preds=cls_preds[v2scene == i],
                targets=targets[v2scene == i],
                v2r=v2r[v2scene == i],
                rois=rois[i],
                gt_idxs=gt_idxs[i],
                batch_data_sample=batch_data_samples[i])
            inst_losses.append(inst_loss)
            seg_losses.append(seg_loss)
        return dict(inst_loss=torch.mean(torch.stack(inst_losses)), 
                    seg_loss=torch.mean(torch.stack(seg_losses)))

    def _loss_single(self, cls_preds: Tensor, targets: Tensor, 
                     v2r: Tensor, rois: BaseInstance3DBoxes, 
                     gt_idxs: Tensor, 
                     batch_data_sample: SampleList) -> tuple[Tensor, ...]:
        """Loss function about binary segmentation per scene.
        
        Args:
            cls_preds (Tensor): Extracted feature maps for proposals.
            targets (Tensor): Extracted voxel-wise instance and semantic markup 
                for proposals.
            v2r (Tensor): voxel to roi mapping.
            rois (BaseInstance3DBoxes): valid boxes for a scene.
            gt_idxs (Tensor): bbox targets for rois for a scene.
            batch_data_sample (SampleList): Batch of meta info.
        
        Returns:
            tuple[Tensor, ...]: Instance and semantic loss values.
        """
        if len(rois) == 0 or cls_preds.shape[0] == 0:
            return cls_preds.sum(), cls_preds.sum() 
        v2r = v2r - v2r.min()
        assert len(torch.unique(v2r)) == len(rois)
        assert torch.all(torch.unique(v2r) == torch.arange(0, 
                                                    v2r.max() + 1).to(v2r.device))
        assert torch.max(gt_idxs) < len(batch_data_sample.gt_instances_3d.bboxes_3d)

        v2bbox = gt_idxs[v2r.long()]
        assert torch.unique(v2bbox)[0] != -1
        inst_targets = targets[:, 0]
        seg_targets = targets[:, 1]

        seg_preds = cls_preds[:, :-1]
        inst_preds = cls_preds[:, -1]

        labels = v2bbox == inst_targets

        seg_targets[seg_targets == -1] = self.n_classes
        seg_loss = self.multiclass_loss(seg_preds, seg_targets.long())

        inst_loss = self.binary_loss(inst_preds, labels)
        return inst_loss, seg_loss
  
    def predict(self,
                x: SparseTensor, 
                points: TensorField,
                proposals: List[InstanceData],
                batch_data_samples: SampleList,
                **kwargs) -> List[PointData]:
        """Generate masks for all scenes.

        Args:
            x (SparseTensor): Input feature map.
            points (TensorField): src points for inverse mapping 
            proposals (List[InstanceData]): Predicted bounding boxes for all
                scenes in a mini-batch.
            batch_data_samples (SampleList): Batch of meta info.

        Returns:
            results (List[PointData]): final predicted masks.
        """
        inv_mapping = points.inverse_mapping(x.coordinate_map_key).long()
        src_idxs = torch.arange(0, x.features.shape[0]).to(inv_mapping.device)
        src_idxs = src_idxs.unsqueeze(1).expand(src_idxs.shape[0], 2)
        bboxes = [(p.bboxes_3d, p.scores_3d, p.labels_3d) for p in proposals]
        cls_preds, idxs, v2r, r2scene, _ , scores, labels = self._forward(x, 
                                                                    src_idxs, 
                                                                    bboxes)
        cls_preds = cls_preds[:, -1]
        idxs = idxs[:, 0]
        v2scene = r2scene[v2r]
        
        results = []
        for i in range(len(batch_data_samples)):
            result = self._get_instances_single(
                cls_preds=cls_preds[v2scene == i],
                idxs=idxs[v2scene == i],
                v2r=v2r[v2scene == i],
                scores=scores[i],
                labels=labels[i],
                inverse_mapping=inv_mapping)
            results.append(result)

        return results

    # per scene
    def _get_instances_single(self, 
                              cls_preds: Tensor, idxs: Tensor, 
                              v2r: Tensor, scores: Tensor, 
                              labels: Tensor, 
                              inverse_mapping: Tensor) -> PointData:
        """Get instances masks for one scene.
        
        Args:
            cls_preds (Tensor): Masks predictions for each proposal.
            idxs (Tensor): Voxels idxs in source voxel space.
            v2r (Tensor): Voxel to roi mapping.
            scores (Tensor): Scores for proposals.
            labels (Tensor): Labels for proposals.
            inverse mapping (Tensor): voxel to point mapping.
        
        Returns:
            (PointData): final instances.

        """ 
        if scores.shape[0] == 0:
            return PointData(pts_instance_mask=inverse_mapping.new_zeros((1, 
                                    len(inverse_mapping)), dtype=torch.bool),
                    instance_labels=inverse_mapping.new_tensor([0], 
                                                dtype=torch.long),
                    instance_scores=inverse_mapping.new_tensor([0], 
                                                    dtype=torch.float32))
        v2r = v2r - v2r.min()
        assert len(torch.unique(v2r)) == scores.shape[0]
        assert torch.all(torch.unique(v2r) == torch.arange(0, 
                                            v2r.max() + 1).to(v2r.device))

        cls_preds = cls_preds.sigmoid()
        bin_cls_preds = cls_preds > self.test_cfg.seg_score_thr
        v2r_one_hot = torch.nn.functional.one_hot(v2r).bool()
        n_rois = v2r_one_hot.shape[1]
        idxs_expand = idxs.unsqueeze(-1).expand(idxs.shape[0], 
                                                n_rois).long()
        bin_preds_exp = bin_cls_preds.unsqueeze(-1).expand(bin_cls_preds.shape[0], 
                                                                        n_rois)
        cls_preds[cls_preds <= self.test_cfg.seg_score_thr] = 0
        cls_preds_expand = cls_preds.unsqueeze(-1).expand(cls_preds.shape[0],
                                                         n_rois)
        idxs_expand[~v2r_one_hot] = inverse_mapping.max() + 1

        voxels_masks = idxs.new_zeros(inverse_mapping.max() + 2, 
                                      n_rois, dtype=bool)
        voxels_preds = idxs.new_zeros(inverse_mapping.max() + 2, 
                                      n_rois)
        voxels_preds = voxels_preds.scatter_(0, idxs_expand, 
                                            cls_preds_expand)[:-1, :]
        voxels_masks = voxels_masks.scatter_(0, idxs_expand, 
                                                bin_preds_exp)[:-1, :]
        
        scores = scores * voxels_preds.sum(axis=0) / (voxels_masks.sum(axis=0) + 0.001)
        points_masks = voxels_masks[inverse_mapping].T.bool()
        
        return PointData(pts_instance_mask=points_masks, 
                         instance_labels=labels, 
                         instance_scores=scores)
