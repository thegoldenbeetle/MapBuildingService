#!/usr/bin/env python3
import json
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import fire


def preprocess_tusimple(
    path: Union[str, Path], out_dir: Union[str, Path], name: str = "TuSimple", verbose: bool = True
):
    path = Path(path)
    out_dir = Path(out_dir)
    train_set = path / "train_set.zip"
    test_set = path / "test_set.zip"
    test_label_file = path / "test_label.json"

    with ZipFile(train_set) as zfile:
        if verbose:
            print(f"Unzip train {train_set}...")
        train_out_path = out_dir / "train" / "TuSimple"
        train_out_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            uncompress_size = sum((f.file_size for f in zfile.infolist()))
            pbar = tqdm(total=uncompress_size, unit = 'B', unit_scale = True, dynamic_ncols=True)
            for f in zfile.infolist():
                zfile.extract(member=f, path=train_out_path)
                pbar.update(f.file_size)
            pbar.close()
        else:
            zfile.extractall(train_out_path)

    with ZipFile(test_set) as zfile:
        if verbose:
            print(f"Unzip test {test_set}...")
        test_out_path = out_dir / "test" / "TuSimple"
        test_out_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            uncompress_size = sum((f.file_size for f in zfile.infolist()))
            pbar = tqdm(total=uncompress_size, unit = 'B', unit_scale = True, dynamic_ncols=True)
            for f in zfile.infolist():
                zfile.extract(member=f, path=test_out_path)
                pbar.update(f.file_size)
            pbar.close()
        else:
            zfile.extractall(test_out_path)

    train_files = set(map(str, train_out_path.glob("**/*.jpg")))
    train_labels = []
    test_labels = []
    with open(test_label_file, "r") as stream:
        if verbose:
            print(f"Create gt...")
        for line in tqdm(stream):
            label = json.loads(line)
            if label["raw_file"] in train_files:
                train_labels.append(label)
            else:
                test_labels.append(label)

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
