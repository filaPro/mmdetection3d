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
    def _extract_single(self, coordinates, features, voxel_size, rois, scores, labels):
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
        n_boxes = len(rois)
        if n_boxes == 0:
            return (coordinates.new_zeros(0),
                    coordinates.new_zeros((0, 3)),
                    features.new_zeros((0, features.shape[1])),
                    features.new_zeros((0, 7)), #todo: why 7?
                    features.new_zeros(0),
                    coordinates.new_zeros(0))
        points = coordinates * self.voxel_size
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        rois = rois.unsqueeze(0).expand(n_points, n_boxes, 7) #todo: why 7?
        face_distances = self.get_face_distances(points, rois)
        inside_condition = face_distances.min(dim=-1).values > 0
        min_pts_condition = inside_condition.sum(dim=0) > self.min_pts_threshold
        inside_condition = inside_condition[:, min_pts_condition]
        rois = rois[0, min_pts_condition]
        scores = scores[min_pts_condition]
        labels = labels[min_pts_condition]
        nonzero = torch.nonzero(inside_condition)
        new_coordinates = coordinates[nonzero[:, 0]]
        return nonzero[:, 1], new_coordinates, features[nonzero[:, 0]], rois, scores, labels

    def extract(self, tensors, levels, rois, scores, labels):
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
        box_type = rois[0].__class__
        with_yaw = rois[0].with_yaw
        for i, roi in enumerate(rois):
            rois[i] = torch.cat((
                roi.gravity_center,
                roi.tensor[:,  3:6] + self.padding,
                roi.tensor[:, 6:]), dim=1)

        new_tensors, new_ids, new_rois, new_scores, new_labels = [], [], [], [], []
        for level, x in enumerate(tensors):
            voxel_size = self.voxel_size * x.tensor_stride[0]
            new_coordinates, new_features, new_roi, new_score, new_label, ids = [], [], [], [], [], []
            n_rois = 0
            for i, (coordinates, features) in enumerate(
                zip(*x.decomposed_coordinates_and_features)):
                roi = rois[i][levels[i] == level]
                score = scores[i][levels[i] == level]
                label = labels[i][levels[i] == level]
                new_index, new_coordinate, new_feature, roi, score, label = self._extract_single(
                    coordinates, features, voxel_size, roi, score, label)
                new_index = new_index + n_rois
                n_rois += len(roi)
                new_coordinate = torch.cat((
                    new_index.unsqueeze(1), new_coordinate), dim=1)
                new_coordinates.append(new_coordinate)
                new_features.append(new_feature)
                ids += [i] * len(roi)
                roi = torch.cat((roi[:, :3],
                            roi[:,  3:6] - self.padding,
                            roi[:, 6:]), dim=1)
                new_roi.append(box_type(roi, with_yaw=with_yaw, origin=(.5, .5, .5)))
                new_score.append(score)
                new_label.append(label)
            new_tensors.append(ME.SparseTensor(
                torch.cat(new_features),
                torch.cat(new_coordinates).float(),
                tensor_stride=x.tensor_stride))
            new_ids.append(x.coordinates.new_tensor(ids))
            new_rois.append(new_roi)
            new_scores.append(new_score)
            new_labels.append(new_label)

        return new_tensors, new_ids, new_rois, new_scores, new_labels
 
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
