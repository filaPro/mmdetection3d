from typing import Dict

from mmengine.logging import MMLogger
from mmdet3d.evaluation import InstanceSegMetric
from .instance_seg_eval import instance_seg_eval
from mmdet3d.registry import METRICS


@METRICS.register_module()
class TD3DInstanceSegMetric(InstanceSegMetric):
    """ The only difference with InstanceSegMetric is that following
    ScanNet evaluator we accept instance prediction as a boolean tensor
    of shape (n_points, n_instances) instead of integer tensor of shape
    (n_points, ). For this purpose we only replace instance_seg_eval call.
    """
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.classes = self.dataset_meta['classes']
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']

        gt_semantic_masks = []
        gt_instance_masks = []
        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []

        for eval_ann, sinlge_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            gt_instance_masks.append(eval_ann['pts_instance_mask'])
            pred_instance_masks.append(
                sinlge_pred_results['pts_instance_mask'])
            pred_instance_labels.append(sinlge_pred_results['instance_labels'])
            pred_instance_scores.append(sinlge_pred_results['instance_scores'])

        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            logger=logger)

        return ret_dict
