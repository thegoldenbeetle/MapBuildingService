import cv2
import lightning as L
import numpy as np
import torch
import torchmetrics
from PIL import Image
from torch import nn, optim
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)
from torchvision.utils import draw_segmentation_masks

from .metrics import CULaneMetric
from .utils import interpolate_lines, lane_ransac_clustering


class SegModel(L.LightningModule):
    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = self._get_model(pretrained=pretrained)

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.valid_acc = torchmetrics.Accuracy(task="binary")
        self.valid_recall = torchmetrics.Recall(task="binary")
        self.valid_precision = torchmetrics.Precision(task="binary")
        self.valid_f1 = torchmetrics.F1Score(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        self.test_culane = CULaneMetric()

    def setup(self, stage):
        del stage
        prototype_array = torch.randn(2, 3, 288, 512)
        self.logger.experiment.add_graph(self, prototype_array)

        self.loggers[1]._version = self.loggers[0].version  # noqa: SLF001

    @staticmethod
    def _get_model(pretrained: bool = True):
        model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            if pretrained
            else None,
        )
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        if pretrained:
            model.aux_classifier[4] = nn.Conv2d(
                10, 1, kernel_size=(1, 1), stride=(1, 1)
            )
        return model

    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        del batch_idx

        x, gt_seg = batch
        bin_gt_seg = (gt_seg > 0).float()
        pred_bin_seg_logit = self(x)[:, 0]
        pred_bin_seg = nn.Sigmoid()(pred_bin_seg_logit)

        pos_weight = torch.full(bin_gt_seg.shape[1:], 0.9527 / 0.0473).to(x.device)
        loss_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
            pred_bin_seg_logit, bin_gt_seg
        )
        loss = loss_seg

        self.log("train/loss", loss)
        self.log("train/loss_seg", loss_seg)

        self.train_acc(pred_bin_seg, bin_gt_seg)
        self.log("train/seg_accuracy", self.train_acc, on_step=True, on_epoch=False)

        self.train_recall(pred_bin_seg, bin_gt_seg)
        self.log("train/seg_recall", self.train_recall, on_step=True, on_epoch=False)

        self.train_precision(pred_bin_seg, bin_gt_seg)
        self.log(
            "train/seg_precision",
            self.train_precision,
            on_step=True,
            on_epoch=False,
        )

        self.train_f1(pred_bin_seg, bin_gt_seg)
        self.log("train/seg_f1", self.train_f1, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, gt_seg, images, gt_lines = batch
        bin_gt_seg = (gt_seg > 0).float()
        pred_bin_seg_logit = self(x)[:, 0]
        pred_bin_seg = nn.Sigmoid()(pred_bin_seg_logit)

        pos_weight = torch.full(bin_gt_seg.shape[1:], 0.9527 / 0.0473).to(x.device)
        loss_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
            pred_bin_seg_logit, bin_gt_seg
        )
        loss = loss_seg

        self.log("val/loss", loss)
        self.log("val/loss_seg", loss_seg)

        self.valid_acc(pred_bin_seg, bin_gt_seg)
        self.log("val/seg_accuracy", self.valid_acc, on_step=True, on_epoch=True)

        self.valid_recall(pred_bin_seg, bin_gt_seg)
        self.log("val/seg_recall", self.valid_recall, on_step=True, on_epoch=True)

        self.valid_precision(pred_bin_seg, bin_gt_seg)
        self.log("val/seg_precision", self.valid_precision, on_step=True, on_epoch=True)

        self.valid_f1(pred_bin_seg, bin_gt_seg)
        self.log("val/seg_f1", self.valid_f1, on_step=True, on_epoch=True)

        if batch_idx == 0:
            img = (
                (transforms.ToTensor()(Image.open(images[0])) * 255)
                .to(torch.uint8)
                .cpu()
            )

            # GT mask
            gt_image = img
            masks = []
            for label in np.unique(gt_seg[0].cpu()):
                if label == 0:
                    continue
                mask = (
                    nn.functional.interpolate(
                        (gt_seg[0].cpu() == label).unsqueeze(0).unsqueeze(0).float(),
                        img.shape[1:],
                    )[0]
                    > 0
                )
                masks.append(mask)
            mask = torch.cat(masks, dim=0)
            gt_image = draw_segmentation_masks(gt_image, mask)

            # Pred mask
            pred_seg = torch.Tensor(
                lane_ransac_clustering(
                    pred_bin_seg[0].cpu().numpy() > 0.5,
                )
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
            if masks:
                mask = torch.cat(masks, dim=0)
                pred_image = draw_segmentation_masks(pred_image, mask)
                self.logger.experiment.add_image(
                    f"image_{batch_idx}/pred",
                    pred_image,
                    self.current_epoch,
                )

            # Pred bin mask
            mask = (
                nn.functional.interpolate(
                    (pred_bin_seg[0] > 0.5).cpu().unsqueeze(0).unsqueeze(0).float(),
                    img.shape[1:],
                )[0, 0]
                > 0
            )
            bin_pred_image = draw_segmentation_masks(img, mask)

            self.logger.experiment.add_image(
                f"image_{batch_idx}/gt",
                gt_image,
                self.current_epoch,
            )
            self.logger.experiment.add_image(
                f"image_{batch_idx}/bin_pred",
                bin_pred_image,
                self.current_epoch,
            )

        return loss

    def test_step(self, batch, batch_idx):
        x, gt_seg, images, gt_lines = batch
        bin_gt_seg = (gt_seg > 0).float()
        pred_bin_seg_logit = self(x)[:, 0]
        pred_bin_seg = nn.Sigmoid()(pred_bin_seg_logit)

        pos_weight = torch.full(bin_gt_seg.shape[1:], 0.9527 / 0.0473).to(x.device)
        loss_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
            pred_bin_seg_logit, bin_gt_seg
        )
        loss = loss_seg

        self.log("test/loss", loss, prog_bar=True)
        self.log("test/loss_seg", loss_seg)

        self.test_acc(pred_bin_seg, bin_gt_seg)
        self.log("test/seg_accuracy", self.test_acc)

        self.test_recall(pred_bin_seg, bin_gt_seg)
        self.log("test/seg_recall", self.test_recall)

        self.test_precision(pred_bin_seg, bin_gt_seg)
        self.log("test/seg_precision", self.test_precision)

        self.test_f1(pred_bin_seg, bin_gt_seg)
        self.log("test/seg_f1", self.test_f1)

        # Line metrics
        for bin_seg, lines, image_file in zip(pred_bin_seg, gt_lines, images):
            pred_seg = lane_ransac_clustering(
                bin_seg.cpu().numpy() >= 0.5,
            )
            size = Image.open(image_file).size
            upscale_pred_seg = cv2.resize(
                pred_seg,
                size,
                interpolation=cv2.INTER_NEAREST,
            )
            pred_lines = interpolate_lines(upscale_pred_seg)
            self.test_culane.update(pred_lines, lines, size=size)

    def on_test_epoch_end(self):
        culane_metrics = self.test_culane.compute()
        for name, value in culane_metrics.items():
            self.log(f"test/culane_{name}", value, logger=True)
        self.test_culane.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
