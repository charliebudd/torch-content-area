import torch
from typing import Sequence, Optional

from .get_points import get_points_handcrafted, get_points_learned
from .fit_area import fit_area

def estimate_area_handcrafted(image: torch.Tensor, strip_count: int=16, feature_thresholds: Sequence[float]=(20, 30, 25), confidence_thresholds: Sequence[float]=(0.03, 0.06)) -> torch.Tensor:
    """
    Estimates the content area for the given endoscopic image(s) using handcrafted feature extraction.
    """
    points = get_points_handcrafted(image, strip_count, feature_thresholds)
    area = fit_area(points, image.shape[-2:], confidence_thresholds)
    return area


def estimate_area_learned(image: torch.Tensor, strip_count: int=16, model: Optional[torch.jit.ScriptModule]=None,  model_patch_size: int=7, confidence_thresholds: Sequence[float]=(0.03, 0.06)) -> torch.Tensor:
    """
    Estimates the content area for the given endoscopic image(s) using learned feature extraction.
    """
    points = get_points_learned(image, strip_count, model, model_patch_size)
    area = fit_area(points, image.shape[-2:], confidence_thresholds)
    return area
