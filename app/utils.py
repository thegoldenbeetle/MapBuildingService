import hashlib
import io
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_segmentation_masks

from .config import settings


def get_mask_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    img_tensor = pil_to_tensor(image)
    seg_masks = []
    for label in np.unique(mask):
        if label == 0:
            continue
        seg_masks.append(torch.tensor(mask == label, dtype=torch.bool))
    seg_mask = torch.stack(seg_masks)
    return to_pil_image(draw_segmentation_masks(img_tensor, seg_mask))


def draw_lines(image: Image.Image, lines: Sequence[np.ndarray]) -> Image.Image:
    # Draw lines
    fig, ax = plt.subplots(1)
    # Hide grid lines
    ax.grid(False)
    ax.set_axis_off()
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ## Show image
    ax.imshow(np.array(image))
    for line in lines:
        ax.plot(*line.T)

    # Save image
    imgdata = io.BytesIO()
    fig.savefig(
        imgdata,
        format="jpg",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    imgdata.seek(0)
    return Image.open(imgdata)


def save_image(image: Image.Image) -> str:
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    img_out_name = image_hash + ".jpg"
    image.save(settings.storage_path / img_out_name)
    return settings.image_path + "/" + img_out_name
