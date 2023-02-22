# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Optional, List

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
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class TD3DSegmentationHead(BaseModule):
    def __init__(self,
                 voxel_size: int,
                 assigner_iou_thr: float,
                 roi_extractor: Optional[dict],
                 unet: Optional[dict],
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None):
        super(TD3DSegmentationHead, self).__init__()
        self.roi_extractor = TASK_UTILS.build(roi_extractor)
        self.unet = MODELS.build(unet)
        self.voxel_size = voxel_size
        self.assigner_iou_thr = assigner_iou_thr 
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to install MinkowskiEngine.`'  # noqa: E501
            )

    def loss(self,
        x: SparseTensor,
        pts_targets: Tensor,
        bboxes_pred: List[BaseInstance3DBoxes], 
        assigned_det_idxs: List[Tensor],
        batch_data_samples: SampleList,
        **kwargs) -> dict:

        
        assigned_bboxes = []
        for i in range(len(batch_data_samples)):
            assigned_iou_idxs = self.get_targets(bboxes_pred[i], 
                                        batch_data_samples[i].gt_instances_3d.bboxes_3d)
            assigned_idxs = assigned_iou_idxs #???????????????????????????? assigned_det_idxs[i]
            assigned_idxs[assigned_idxs != assigned_iou_idxs] = -1

            boxes = bboxes_pred[i][assigned_idxs != -1]
            assigned_idxs = assigned_idxs[assigned_idxs != -1]

            if len(boxes) != 0:
                assigned_idxs_one_hot = torch.nn.functional.one_hot(assigned_idxs)
                mask, idxs = torch.topk(assigned_idxs_one_hot, min(self.train_cfg.num_rois, len(boxes)), 0)
                sampled_boxes = batch_data_samples[i].box_type_3d(boxes.tensor[idxs].view(-1, boxes.tensor.shape[1]))
                sampled_assigned_idxs = assigned_idxs[idxs].view(-1)
                mask = mask.view(-1).bool()
                assigned_bboxes.append((sampled_boxes[mask],
                                        sampled_assigned_idxs.new_zeros(len(sampled_boxes[mask])),
                                        sampled_assigned_idxs[mask]))
            else:
                assigned_bboxes.append((boxes, 
                                        assigned_idxs.new_zeros(0),                      
                                        assigned_idxs))


        cls_preds, targets, v2r, r2scene, rois, _ , assigned_idxs = self._forward(x, pts_targets, assigned_bboxes)
        loss = self._loss(cls_preds, targets, v2r, r2scene, rois, assigned_idxs, batch_data_samples)

        return loss

    def get_targets(self, rois: BaseInstance3DBoxes, 
                    gt_bboxes: BaseInstance3DBoxes) -> Tensor:
        ious = rois.overlaps(rois, gt_bboxes.to(rois.device))
        values, indices = ious.max(dim=1)
        indices = torch.where(values > self.assigner_iou_thr, indices, -1)
        return indices

    def _forward(self, x, targets, bboxes):
        rois = [b[0] for b in bboxes]
        scores = [b[1] for b in bboxes]
        labels = [b[2] for b in bboxes]
        levels = [torch.zeros(len(b[0])) for b in bboxes]
        
        feats_with_targets = ME.SparseTensor(torch.cat((x.features, targets), axis=1), x.coordinates)
        tensors, ids, rois, scores, labels = self.roi_extractor.extract([feats_with_targets], levels, rois, scores, labels)
        if tensors[0].features.shape[0] == 0:
            return (targets.new_zeros((0, 1)),
                    targets.new_zeros((0, 1)),
                    targets.new_zeros(0),
                    targets.new_zeros(0),
                    [targets.new_zeros((0, 7)) for i in range(len(bboxes))],
                    [targets.new_zeros(0) for i in range(len(bboxes))],
                    [targets.new_zeros(0) for i in range(len(bboxes))])

        feats = ME.SparseTensor(tensors[0].features[:, :-2], tensors[0].coordinates)
        targets = tensors[0].features[:, -2:]

        preds = self.unet(feats).features
        return preds, targets, feats.coordinates[:, 0].long(), ids[0], rois[0], scores[0], labels[0]

    def _loss(self, cls_preds, targets, v2r, r2scene, rois, gt_idxs, batch_data_samples):
        v2scene = r2scene[v2r]
        inst_losses = []
        seg_losses = []
        for i in range(len(batch_data_samples)):
            inst_loss, seg_loss = self._loss_second_single(
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

    def _loss_single(self, cls_preds, targets, v2r, rois, gt_idxs, batch_data_sample):
        if len(rois) == 0 or cls_preds.shape[0] == 0:
            return cls_preds.sum().float()
        v2r = v2r - v2r.min()
        assert len(torch.unique(v2r)) == len(rois)
        assert torch.all(torch.unique(v2r) == torch.arange(0, v2r.max() + 1).to(v2r.device))
        assert torch.max(gt_idxs) < len(batch_data_sample.gt_instances_3d.bboxes_3d)

        v2bbox = gt_idxs[v2r.long()]
        assert torch.unique(v2bbox)[0] != -1
        inst_targets = targets[:, 0]
        seg_targets = targets[:, 1]

        seg_preds = cls_preds[:, :-1]
        inst_preds = cls_preds[:, -1]

        labels = v2bbox == inst_targets

        seg_targets[seg_targets == -1] = self.n_classes
        seg_loss = self.cls_loss(seg_preds, seg_targets.long())

        inst_loss = self.inst_loss(inst_preds, labels)
        return inst_loss, seg_loss


























    def predict(self, 
        x: SparseTensor, 
        field: TensorField, 
        batch_data_samples: SampleList,
        **kwargs) -> Tuple:
        pass
