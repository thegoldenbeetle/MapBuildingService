import warnings
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import MeanShift
from sklearn.linear_model import RANSACRegressor


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
    if not (bin_seg > 0).sum():
        return seg
    embeddings = emb[:, bin_seg > 0].T
    mean_shift = MeanShift(bandwidth=threshold, bin_seeding=True, n_jobs=-1)
    mean_shift.fit(embeddings)
    labels = mean_shift.labels_
    seg[bin_seg] = labels + 1
    for label in np.unique(seg):
        label_seg = seg[seg == label]
        if len(seg[seg == label]) < min_points:
            label_seg[::] = 0
    return seg


def lane_ransac_clustering(
    bin_seg: np.ndarray,
    residual_threshold: float = 30,
    min_samples: int = 3,
    min_points: int = 100,
) -> np.ndarray:
    points = np.stack(np.where(bin_seg > 0.5), axis=1)
    line_seg = np.zeros(bin_seg.shape, dtype=int)
    i = 0
    while len(points) > min_points:
        x = np.ones((len(points), 2))
        x[:, 1] = points[:, 0]
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold, min_samples=min_samples
        )
        _ = ransac.fit(x, points[:, 1])
        inlier_mask = ransac.inlier_mask_
        if inlier_mask.sum() > min_points:
            i += 1
            mask_points = points[inlier_mask]
            line_seg[mask_points[:, 0], mask_points[:, 1]] = i
        points = points[~inlier_mask]
    return line_seg


def interpolate_lines(
    mask: np.ndarray,
    points: int = 100,
) -> Sequence[np.ndarray]:
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


def interpolate_lines_ransac(
    mask: np.ndarray,
    points: int = 100,
    residual_threshold: float = 30,
    min_samples: int = 15,
    min_x: Optional[int] = None,
    max_x: Optional[int] = None,
    min_y: Optional[int] = None,
    max_y: Optional[int] = None,
) -> Sequence[np.ndarray]:
    lines = []
    for label in np.unique(mask):
        if label == 0:
            continue
        y, x = np.where(mask == label)
        Y = np.ones((len(y), 3))
        Y[:, 1] = y
        Y[:, 2] = y**2
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold, min_samples=min_samples
        )
        ransac.fit(Y, x)

        new_y = np.linspace(y.min(), y.max(), points)
        new_Y = np.ones((len(new_y), 3))
        new_Y[:, 1] = new_y
        new_Y[:, 2] = new_y**2
        new_x = ransac.predict(new_Y)
        line = np.stack((new_x, new_y), axis=1)
        line = line[line[:, 0] >= 1]
        line = line[line[:, 0] < mask.shape[1] - 1]
        line = line[line[:, 1] >= 1]
        line = line[line[:, 1] < mask.shape[0] - 1]
        if min_x:
            line = line[line[:, 0] >= min_x]
        if max_x:
            line = line[line[:, 0] <= max_x]
        if min_y:
            line = line[line[:, 1] >= min_y]
        if max_y:
            line = line[line[:, 1] <= max_y]
        lines.append(line)
    return lines
