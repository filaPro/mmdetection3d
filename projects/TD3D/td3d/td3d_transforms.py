import numpy as np
import scipy
import torch
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class Elastic(BaseTransform):
    """Apply elastic augmentation to a 3D scene.

    Required Keys:

    - points (np.float32)

    Modified Keys:

    - points (np.float32)

    """

    def transform(self, input_dict):
        """Private function for elastic to a points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after elastic, 'points' is updated
            in the result dict.
        """

        coords = input_dict['points'].tensor[:, :3].numpy()
        coords = self.elastic(coords, 6, 40.)
        coords = self.elastic(coords, 20, 160.)
        input_dict['points'].tensor[:, :3] = torch.tensor(coords)
        return input_dict

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
            for n in noise
        ]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

@TRANSFORMS.register_module()
class BboxRecalculation(BaseTransform):
    """Re-calculation of bounding boxes after all points's augmentations.

    Required Keys:

    - points (np.float32)
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - gt_bboxes_3d   

    Modified Keys:
    
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - gt_bboxes_3d (np.float32)
    - gt_labels_3d (np.float32)

    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes: int) -> None:
        self.num_classes = num_classes

    def transform(self, input_dict):
        """Private function for re-calculation of bounding boxes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after re-calculation of bounding boxes, 
            'pts_instance_mask', 'pts_semantic_mask', 'gt_bboxes_3d',
            'gt_labels_3d' are updated in the result dict.
        """
        # because pts_instance_mask contains instances from non-instaces classes
        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        pts_semantic_mask[pts_semantic_mask == self.num_classes] = -1
        
        idxs = torch.unique(pts_instance_mask)
        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1
        pts_instance_mask = mapping[pts_instance_mask]

        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()
        input_dict['pts_semantic_mask'] = pts_semantic_mask.numpy()

        if torch.sum(pts_instance_mask == -1) != 0:
            pts_instance_mask[pts_instance_mask == -1] = torch.max(pts_instance_mask) + 1
            pts_instance_mask_one_hot = torch.nn.functional.one_hot(pts_instance_mask)[
                :, :-1
            ]
        else:
            pts_instance_mask_one_hot = torch.nn.functional.one_hot(pts_instance_mask)


        points = input_dict['points'][:, :3].tensor
        points_for_max = points.unsqueeze(1).expand(points.shape[0], 
                                                    pts_instance_mask_one_hot.shape[1], 
                                                    points.shape[1]).clone()
        points_for_min = points.unsqueeze(1).expand(points.shape[0], 
                                                    pts_instance_mask_one_hot.shape[1], 
                                                    points.shape[1]).clone()
        points_for_max[~pts_instance_mask_one_hot.bool()] = float('-inf')
        points_for_min[~pts_instance_mask_one_hot.bool()] = float('inf')
        bboxes_max = points_for_max.max(axis=0)[0]
        bboxes_min = points_for_min.min(axis=0)[0]
        bboxes_sizes = bboxes_max - bboxes_min
        bboxes_centers = (bboxes_max + bboxes_min) / 2
        bboxes = torch.hstack((bboxes_centers, bboxes_sizes, torch.zeros_like(bboxes_sizes[:, :1])))
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"].__class__(bboxes, with_yaw=False, 
                                                                          origin=(.5, .5, .5))

        pts_semantic_mask_expand = pts_semantic_mask.unsqueeze(1).expand(pts_semantic_mask.shape[0], 
                                                                         pts_instance_mask_one_hot.shape[1]).clone()
        pts_semantic_mask_expand[~pts_instance_mask_one_hot.bool()] = -1
        assert pts_semantic_mask_expand.max(axis=0)[0].shape[0] != 0
        input_dict['gt_labels_3d'] = pts_semantic_mask_expand.max(axis=0)[0].numpy()
        return input_dict