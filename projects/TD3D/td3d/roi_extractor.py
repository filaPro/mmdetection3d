from mmdet3d.registry import TASK_UTILS
import torch
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = TensorField = None
    pass

@TASK_UTILS.register_module()
class Mink3DRoIExtractor:
    def __init__(self, voxel_size, padding, min_pts_threshold):
        # min_pts_threshold: minimal number of points per roi
        self.voxel_size = voxel_size
        self.padding = padding
        self.min_pts_threshold = min_pts_threshold
    
    # per scene and per level
    def _extract_single(self, coordinates, features, rois):
        # coordinates: of shape (n_points, 3)
        # features: of shape (n_points, c)
        # voxel_size: float
        # rois: of shape (n_rois, 7) #todo: why 7?
        # -> new indices of shape n_new_points
        # -> new coordinates of shape (n_new_points, 3)
        # -> new features of shape (n_new_points, c + 3)
        # -> new rois of shape (n_new_rois, 7) #todo: why 7?
        # -> new scores of shape (n_new_rois)
        # -> new labels of shape (n_new_rois)
        n_points = len(coordinates)
        n_rois = len(rois)
        roi_size = rois.shape[1]
        if n_rois == 0:
            return (coordinates.new_zeros(0),
                    coordinates.new_zeros((0, 3)),
                    features.new_zeros((0, features.shape[1])),
                    features.new_zeros(0))

        points = coordinates * self.voxel_size
        points = points.unsqueeze(1).expand(n_points, n_rois, 3)
        rois = rois.unsqueeze(0).expand(n_points, n_rois, roi_size)
        face_distances = self.get_face_distances(points, rois)
        inside_condition = face_distances.min(dim=-1).values > 0
        min_pts_condition = inside_condition.sum(dim=0) > self.min_pts_threshold
        inside_condition = inside_condition[:, min_pts_condition]
        nonzero = torch.nonzero(inside_condition)
        new_coordinates = coordinates[nonzero[:, 0]]
        return nonzero[:, 1], new_coordinates, features[nonzero[:, 0]], min_pts_condition

    def extract(self, x, bboxes):
        # tensors: list[SparseTensor] of len n_levels
        # levels: list[Tensor] of len batch_size;
        #         each of shape n_rois_i
        # rois: list[BaseInstance3DBoxes] of len batch_size;
        #       each of len n_rois_i
        # -> list[SparseTensor] of len n_levels
        # -> list[Tensor] of len n_levels;
        #    contains scene id for each extracted roi
        # -> list[list[BaseInstance3DBoxes]] of len n_levels;
        #    each of len batch_size; just splitted rois
        #    per level and per scene
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

            new_index, new_coordinate, new_feature, valid_roi_idx = self._extract_single(
                coordinates, features, rois[i])
            new_index = new_index + n_rois            
            new_coordinate = torch.cat((
                new_index.unsqueeze(1), new_coordinate), dim=1)
            new_coordinates.append(new_coordinate)
            new_features.append(new_feature)
            valid_roi_idxs.append(valid_roi_idx)
            n_valid_rois = valid_roi_idx.sum()
            n_rois += n_valid_rois
            ids += [i] * n_valid_rois

        new_tensor = ME.SparseTensor(
            torch.cat(new_features),
            torch.cat(new_coordinates).float(),
            tensor_stride=x.tensor_stride)

        return new_tensor, x.coordinates.new_tensor(ids), valid_roi_idxs
 
    @staticmethod
    def get_face_distances(points, boxes):
        # points: of shape (..., 3)
        # boxes: of shape (..., 6)
        # -> of shape (..., 6): dx_min, dx_max, dy_min, dy_max, dz_min, dz_max
        dx_min = points[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - points[..., 0]
        dy_min = points[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - points[..., 1]
        dz_min = points[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - points[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
