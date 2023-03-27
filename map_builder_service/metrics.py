from typing import List, Sequence, Tuple

import numpy as np
import torch
from torchmetrics import Metric

from .utils import lines_to_mask


class CULaneMetric(Metric):
    def __init__(
        self,
        line_width: int = 30,
    ):
        super().__init__()

        self.add_state(
            "tp",
            default=torch.tensor(0, dtype=torch.int32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.tensor(0, dtype=torch.int32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.tensor(0, dtype=torch.int32),
            dist_reduce_fx="sum",
        )

        self.line_width = line_width

    def _mask_pairwise_iou(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> List[Tuple[int, int, torch.Tensor]]:
        pred_cls_list = preds.unique().cpu().numpy().tolist()
        target_cls_list = target.unique().cpu().numpy().tolist()
        pirwise_iou = []
        for pred_cls in pred_cls_list:
            if pred_cls == 0:
                continue
            for target_cls in target_cls_list:
                if target_cls == 0:
                    continue
                pirwise_iou.append(
                    (
                        pred_cls,
                        target_cls,
                        self._mask_iou(
                            preds == pred_cls,
                            target == target_cls,
                        ),
                    )
                )
        return pirwise_iou

    def _mask_iou(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds_area = preds.count_nonzero()
        target_area = target.count_nonzero()
        intersection = (preds & target).count_nonzero()
        return intersection / (preds_area + target_area - intersection)

    def update(
        self,
        pred: Sequence[np.ndarray],
        target: Sequence[np.ndarray],
        size: Tuple[int, int],
    ):
        pred_mask = torch.tensor(
            lines_to_mask(pred, size, self.line_width),
            dtype=torch.int32,
        )
        target_mask = torch.tensor(
            lines_to_mask(target, size, self.line_width),
            dtype=torch.int32,
        )
        self.update_masks(pred_mask, target_mask)

    def update_masks(self, pred: torch.Tensor, target: torch.Tensor):
        assert not pred.is_floating_point()
        assert not target.is_floating_point()
        assert pred.shape == target.shape

        pred_detected_lines = set()
        target_detected_lines = set()
        target_classes = set(target.unique().detach().cpu().numpy().tolist()) - set([0])
        pairwise_iou = sorted(
            self._mask_pairwise_iou(
                pred,
                target,
            ),
            key=lambda x: x[2].item(),
            reverse=True,
        )
        for pred_cls, target_cls, iou in pairwise_iou:
            if pred_cls in pred_detected_lines or target_cls in target_detected_lines:
                continue
            if iou >= 0.5:
                self.tp += torch.tensor(1)
                target_detected_lines.add(target_cls)
            else:
                self.fp += torch.tensor(1)
            pred_detected_lines.add(pred_cls)
        self.fn += torch.tensor(len(target_classes - target_detected_lines))

    def compute_precision(self):
        return self.tp / (self.tp + self.fp)

    def compute_recall(self):
        return self.tp / (self.tp + self.fn)

    def compute_f1(self):
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    def compute(self):
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.compute_precision(),
            "recall": self.compute_recall(),
            "f1": self.compute_f1(),
        }
