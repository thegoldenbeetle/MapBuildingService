import itertools
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .utils import lines_to_mask


class LaneDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform=None,
        line_transform=None,
        line_radius: float = 5.0,
        sub_datasets: Optional[Sequence[str]] = None,
        is_test: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.is_test = is_test
        self.line_radius = line_radius
        self.transform = transform
        self.line_transform = line_transform
        self.sub_datasets = set(
            filter(
                lambda x: x.is_dir(),
                self.data_dir.glob("*"),
            )
        )
        if sub_datasets is not None:
            self.sub_datasets = self.sub_datasets & set(sub_datasets)
        self.dataset_info = list(
            itertools.chain.from_iterable(
                map(
                    self._load_subdataset,
                    self.sub_datasets,
                )
            )
        )

    @staticmethod
    def _load_subdataset(subdataset: Path) -> list:
        with (subdataset / "labels.json").open("r") as stream:
            items = json.load(stream)
        for item in items:
            item["raw_file"] = subdataset / item["raw_file"]
        return items

    def __len__(self):
        return len(self.dataset_info)

    def _item_to_lines(self, item: dict) -> Sequence[np.ndarray]:
        lines = []
        for i, line_xs in enumerate(item["lanes"]):
            line = np.array(
                [point for point in zip(line_xs, item["h_samples"]) if point[0] > 0],
                dtype=int,
            )
            lines.append(line)
        return lines

    def __getitem__(self, idx):
        item = self.dataset_info[idx]
        image_file = item["raw_file"]
        image = Image.open(image_file)
        gt_lines = self._item_to_lines(item)
        gt_mask = lines_to_mask(gt_lines, image.size, int(self.line_radius * 2))
        if self.transform is not None:
            image = self.transform(image)
        if self.line_transform is not None:
            gt_mask = self.line_transform(gt_mask)
        gt_mask = gt_mask[0]
        if self.is_test:
            return image, gt_mask, str(image_file), gt_lines
        return image, gt_mask

    @staticmethod
    def collate_fn(data):
        is_test = len(data[0]) != 2
        if is_test:
            image, gt_mask, image_file, gt_lines = zip(*data)
        else:
            image, gt_mask = zip(*data)
        image = torch.stack(image)
        gt_mask = torch.stack(gt_mask)
        if is_test:
            return image, gt_mask, image_file, gt_lines
        else:
            return image, gt_mask


class LaneDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        train_transform=None,
        train_line_transform=None,
        test_transform=None,
        test_line_transform=None,
        sub_datasets: Optional[Sequence[str]] = None,
        train_size: float = 0.8,
        line_radius: float = 30.0,
        batch_size: int = 64,
        image_size: Tuple[int, int] = (512, 288),
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_size = train_size
        self.line_radius = line_radius
        self.image_size = image_size
        self.num_workers = num_workers
        if num_workers is None:
            self.num_workers = os.cpu_count()

        self.train_transform = train_transform
        if train_transform is None:
            self.train_transform = self.default_train_transform()
        self.train_line_transform = train_line_transform
        if train_line_transform is None:
            self.train_line_transform = self.default_train_line_transform()

        self.test_transform = test_transform
        if test_transform is None:
            self.test_transform = self.default_test_transform()
        self.test_line_transform = test_line_transform
        if test_line_transform is None:
            self.test_line_transform = self.default_test_line_transform()

        self.sub_datasets = sub_datasets

    def default_train_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(),
                transforms.RandomGrayscale(),
                transforms.GaussianBlur(3),
                transforms.RandomErasing(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize(self.image_size, antialias=True),
            ]
        )

    def default_train_line_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    self.image_size, interpolation=Image.NEAREST, antialias=True
                ),
            ]
        )

    def default_test_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize(self.image_size, antialias=True),
            ]
        )

    def default_test_line_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    self.image_size, interpolation=Image.NEAREST, antialias=True
                ),
            ]
        )

    def setup(self, stage=None):
        del stage
        self.train_data = LaneDataset(
            self.data_dir / "train",
            transform=self.train_transform,
            line_transform=self.train_line_transform,
            line_radius=self.line_radius,
            sub_datasets=self.sub_datasets,
        )
        self.val_data = LaneDataset(
            self.data_dir / "test",
            transform=self.test_transform,
            line_transform=self.test_line_transform,
            line_radius=self.line_radius,
            sub_datasets=self.sub_datasets,
            is_test=True,
        )
        self.test_data = LaneDataset(
            self.data_dir / "test",
            transform=self.test_transform,
            line_transform=self.test_line_transform,
            line_radius=self.line_radius,
            sub_datasets=self.sub_datasets,
            is_test=True,
        )

    def train_dataloader(self):
        return DataLoader(
            Subset(self.train_data, list(range(16))),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=LaneDataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.val_data, list(range(16))),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=LaneDataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            Subset(self.test_data, list(range(16))),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=LaneDataset.collate_fn,
        )
