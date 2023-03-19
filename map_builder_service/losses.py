from collections.abc import Sequence
from typing import Iterable

import torch
from torch import nn


class LaneNetEmbedingLoss(nn.Module):
    def __init__(self, sigma_v: float = 0.5, sigma_d: float = 3.0):
        super().__init__()
        self.sigma_v = sigma_v
        self.sigma_d = sigma_d

    def forward(self, emb: torch.Tensor, gt_seg: torch.Tensor):
        loss = [self.loss(batch_emb, batch_gt_seg) for batch_emb, batch_gt_seg in zip(emb, gt_seg)]
        return sum(loss) / len(loss)

    def loss(self, emb: torch.Tensor, gt_seg: torch.Tensor):
        classes = torch.unique(gt_seg)
        cluster_points = [emb[:, gt_seg == cls].transpose(0, 1) for cls in classes]
        cluster_means = [points.mean(dim=0) for points in cluster_points]
        loss_var = self.var_loss(cluster_points, cluster_means)
        loss_dist = self.dist_loss(cluster_means)
        return loss_var + loss_dist

    def _cluster_var_loss(self, points: torch.Tensor, cluster_mean: torch.Tensor):
        loss = (cluster_mean - points).norm() - self.sigma_v
        loss = loss * (loss > 0)
        return loss.mean()

    def var_loss(self, points: Iterable[torch.Tensor], cluster_means: Iterable[torch.Tensor]):
        cluster_losses = [
            self._cluster_var_loss(cls_points, cls_mean)
            for cls_points, cls_mean in zip(points, cluster_means)
        ]
        return sum(cluster_losses) / len(cluster_losses)

    def dist_loss(self, cluster_means: Sequence[torch.Tensor]):
        loss = 0.0
        C = len(cluster_means)
        for i, mu_i in enumerate(cluster_means):
            for j, mu_j in enumerate(cluster_means):
                if j == i:
                    continue
                loss = loss + max(0.0, self.sigma_d - (mu_i - mu_j).norm())
        return loss / (C * (C - 1))
