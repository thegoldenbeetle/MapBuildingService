#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import fire
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _extract_zip(
    zip_file_path: Union[str, Path], out_dir: Union[str, Path], verbose: bool = True
):
    out_dir = Path(out_dir)
    with ZipFile(zip_file_path) as zfile:
        if verbose:
            logger.info(f"Unzip {zip_file_path}...")
        out_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            uncompress_size = sum((f.file_size for f in zfile.infolist()))
            pbar = tqdm(
                total=uncompress_size, unit="B", unit_scale=True, dynamic_ncols=True
            )
            for f in zfile.infolist():
                zfile.extract(member=f, path=out_dir)
                pbar.update(f.file_size)
            pbar.close()
        else:
            zfile.extractall(out_dir)


def preprocess_tusimple(
    path: Union[str, Path],
    out_dir: Union[str, Path],
    name: str = "TuSimple",
    verbose: bool = True,
):
    path = Path(path)
    out_dir = Path(out_dir)
    train_set = path / "train_set.zip"
    test_set = path / "test_set.zip"
    test_label_file = path / "test_label.json"

    train_out_path = out_dir / "train" / "TuSimple"
    _extract_zip(train_set, train_out_path, verbose=verbose)

    test_out_path = out_dir / "test" / "TuSimple"
    _extract_zip(test_set, test_out_path, verbose=verbose)

    test_labels = []
    with open(test_label_file, "r") as stream:
        if verbose:
            logger.info("Create test gt...")
        for line in stream:
            label = json.loads(line)
            test_labels.append(label)

    train_labels = []
    if verbose:
        logger.info("Create train gt...")
    for train_label_file in train_out_path.glob("label_data_*.json"):
        with open(train_label_file, "r") as stream:
            for line in stream:
                label = json.loads(line)
                train_labels.append(label)

    out_train_labels_file = out_dir / "train" / "TuSimple" / "labels.json"
    with out_train_labels_file.open("w") as stream:
        json.dump(train_labels, stream)

    out_test_labels_file = out_dir / "test" / "TuSimple" / "labels.json"
    with out_test_labels_file.open("w") as stream:
        json.dump(test_labels, stream)


if __name__ == "__main__":
    fire.Fire(
        {
            "tusimple": preprocess_tusimple,
        }
    )
