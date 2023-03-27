import warnings
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import MeanShift


def lines_to_mask(
    lines: Iterable[np.ndarray],
    size: Tuple[int, int],
    line_width: int,
) -> np.ndarray:
    mask_img = Image.new("I", size, color=0)
    draw = ImageDraw.Draw(mask_img)
    for i, line in enumerate(lines):
        cls = i + 1
        draw.line(
            list(map(tuple, line.astype(int))),
            fill=cls,
            width=line_width,
        )
    return np.array(mask_img)


def lanelet_clustering(emb, bin_seg, threshold: float = 1.5, min_points: int = 15):
    seg = np.zeros(bin_seg.shape, dtype=int)
    if not (bin_seg > 0.5).sum():
        return seg
    embeddings = emb[:, bin_seg > 0.5].transpose(0, 1)
    mean_shift = MeanShift(bandwidth=threshold, bin_seeding=True, n_jobs=-1)
    mean_shift.fit(embeddings)
    labels = mean_shift.labels_
    seg[bin_seg > 0 / 5] = labels + 1
    for label in np.unique(seg):
        label_seg = seg[seg == label]
        if len(seg[seg == label]) < min_points:
            label_seg[::] = 0
    return seg


def interpolate_lines(
    mask: np.ndarray,
    points: int = 100,
) -> Sequence[Sequence[np.ndarray]]:
    lines = []
    for label in np.unique(mask):
        if label == 0:
            continue
        y, x = np.where(mask == label)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            p = np.poly1d(np.polyfit(y, x, 2))
        new_y = np.linspace(y.min(), y.max(), points)
        new_x = p(new_y)
        line = np.stack((new_x, new_y), axis=1)
        lines.append(line)
    return lines
