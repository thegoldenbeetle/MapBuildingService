import lightning as L
import numpy as np
import torch
import torchmetrics
from PIL import Image
from sklearn.cluster import MeanShift
from torch import Tensor, nn, optim
from torchvision import transforms
from torchvision.models import VGG16_BN_Weights, vgg16_bn
from torchvision.utils import draw_segmentation_masks

from .losses import LaneNetEmbedingLoss


def lanelet_clustering(emb, bin_seg, threshold: float = 1.5, min_points: int = 15):
    seg = np.zeros(bin_seg.shape, dtype=int)
    if not (bin_seg > 0).sum():
        return seg
    embeddings = emb[:, bin_seg > 0].transpose(0, 1)
    mean_shift = MeanShift(bandwidth=threshold, bin_seeding=True, n_jobs=-1)
    mean_shift.fit(embeddings)
    labels = mean_shift.labels_
    seg[bin_seg > 0] = labels + 1
    for label in np.unique(seg):
        label_seg = seg[seg == label]
        if len(seg[seg == label]) < min_points:
            label_seg[::] = 0
    return seg


class LaneNet(L.LightningModule):
    def __init__(self, pretrained: bool = True, embeding_size: int = 8):
        super().__init__()
        self.backbone = self._init_backbone(pretrained=pretrained)

        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
        )
        self.emb = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, embeding_size, 1),
        )
        self.seg = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.valid_acc = torchmetrics.Accuracy(task="binary")
        self.valid_recall = torchmetrics.Recall(task="binary")
        self.valid_precision = torchmetrics.Precision(task="binary")
        self.valid_f1 = torchmetrics.F1Score(task="binary")

    def setup(self, stage):
        del stage
        prototype_array = torch.randn(2, 3, 512, 288)
        self.logger.experiment.add_graph(self, prototype_array)

    @staticmethod
    def _init_backbone(pretrained: bool = True):
        weights = None
        if pretrained:
            weights = VGG16_BN_Weights.DEFAULT
        backbone = vgg16_bn(weights=weights).features
        backbone._modules.pop("33")
        backbone._modules.pop("43")
        return backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.layer1(x)
        emb = self.emb(x)
        seg = self.seg(x).squeeze(1)
        return emb, seg

    def training_step(self, batch, batch_idx):
        del batch_idx

        x, gt_seg = batch
        bin_gt_seg = (gt_seg > 0).float()
        pred_emb, pred_bin_seg = self(x)

        loss_emb = LaneNetEmbedingLoss()(pred_emb, gt_seg)
        loss_seg = nn.BCELoss()(pred_bin_seg, bin_gt_seg)
        loss = loss_emb + loss_seg

        self.log("train/loss", loss)
        self.log("train/loss_seg", loss_seg)
        self.log("train/loss_emb", loss_emb)

        self.train_acc(pred_bin_seg, bin_gt_seg)
        self.log("train/seg_accuracy", self.train_acc, on_step=True, on_epoch=False)

        self.train_recall(pred_bin_seg, bin_gt_seg)
        self.log("train/seg_recall", self.train_recall, on_step=True, on_epoch=False)

        self.train_precision(pred_bin_seg, bin_gt_seg)
        self.log("train/seg_precision", self.train_precision, on_step=True, on_epoch=False)

        self.train_f1(pred_bin_seg, bin_gt_seg)
        self.log("train/seg_f1", self.train_f1, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, gt_seg, images = batch
        bin_gt_seg = (gt_seg > 0).float()
        pred_emb, pred_bin_seg = self(x)

        loss_emb = LaneNetEmbedingLoss()(pred_emb, gt_seg)
        loss_seg = nn.BCELoss()(pred_bin_seg, bin_gt_seg)
        loss = loss_emb + loss_seg

        self.log("val/loss", loss)
        self.log("val/loss_seg", loss_seg)
        self.log("val/loss_emb", loss_emb)

        self.valid_acc(pred_bin_seg, bin_gt_seg)
        self.log("val/seg_accuracy", self.valid_acc, on_step=True, on_epoch=True)

        self.valid_recall(pred_bin_seg, bin_gt_seg)
        self.log("val/seg_recall", self.valid_recall, on_step=True, on_epoch=True)

        self.valid_precision(pred_bin_seg, bin_gt_seg)
        self.log("val/seg_precision", self.valid_precision, on_step=True, on_epoch=True)

        self.valid_f1(pred_bin_seg, bin_gt_seg)
        self.log("val/seg_f1", self.valid_f1, on_step=True, on_epoch=True)

        if batch_idx * x.shape[0] in [0, 100, 500]:
            img = (transforms.ToTensor()(Image.open(images[0][0])) * 255).to(torch.uint8)

            # GT mask
            gt_image = img
            masks = []
            for label in np.unique(gt_seg[0]):
                if label == 0:
                    continue
                mask = (
                    nn.functional.interpolate(
                        (gt_seg[0] == label).unsqueeze(0).unsqueeze(0).float(),
                        img.shape[1:],
                    )[0]
                    > 0
                )
                masks.append(mask)
            mask = torch.cat(masks, dim=0)
            gt_image = draw_segmentation_masks(gt_image, mask)

            # Pred mask
            pred_seg = torch.Tensor(
                lanelet_clustering(pred_emb[0], pred_bin_seg[0], threshold=1.5, min_points=15)
            )
            pred_image = img
            masks = []
            for label in np.unique(pred_seg):
                if label == 0:
                    continue
                mask = (
                    nn.functional.interpolate(
                        (pred_seg == label).unsqueeze(0).unsqueeze(0).float(),
                        img.shape[1:],
                    )[0]
                    > 0
                )
                masks.append(mask)
            mask = torch.cat(masks, dim=0)
            pred_image = draw_segmentation_masks(pred_image, mask)

            # Pred bin mask
            mask = (
                nn.functional.interpolate(
                    pred_bin_seg[0].unsqueeze(0).unsqueeze(0).float(),
                    img.shape[1:],
                )[0, 0]
                > 0
            )
            bin_pred_image = draw_segmentation_masks(img, mask)

            self.logger.experiment.add_embedding(
                pred_emb[0][:, pred_bin_seg[0] > 0].transpose(0, 1), global_step=self.current_epoch
            )
            self.logger.experiment.add_image(
                f"image_{batch_idx}/gt",
                gt_image,
                self.current_epoch,
            )
            self.logger.experiment.add_image(
                f"image_{batch_idx}/pred",
                pred_image,
                self.current_epoch,
            )
            self.logger.experiment.add_image(
                f"image_{batch_idx}/bin_pred",
                bin_pred_image,
                self.current_epoch,
            )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
