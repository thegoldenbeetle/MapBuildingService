from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from map_builder_service.lanenet import LaneNet
from map_builder_service.utils import interpolate_lines_ransac


@dataclass
class PredictResult:
    mask: np.ndarray
    lines_2d: List[np.ndarray]
    lines_3d: Optional[List[np.ndarray]]


class DetectionModel:
    def __init__(self):
        self.image_size = (288, 512)
        self.model = self.load_model()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize(self.image_size, antialias=True),
            ]
        )

    def load_model(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = LaneNet.load_from_checkpoint("data/model.ckpt")
        model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def detect_line(self, img: Image.Image) -> PredictResult:
        line_mask = self.model.lines_segmentation(self.transform(img).unsqueeze(0))[0]
        mask = cv2.resize(
            line_mask,
            img.size,
            interpolation=cv2.INTER_NEAREST,
        )
        lines = interpolate_lines_ransac(mask, points=1000, min_y=350)
        return PredictResult(
            mask=mask,
            lines_2d=list(lines),
            lines_3d=None,
        )
