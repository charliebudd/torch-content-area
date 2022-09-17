import torch
import torchcontentareaext as ext
from typing import Sequence, Optional


models = {}

def load_default_model(device):
    dir = "/".join(__file__.split("/")[:-1])
    return torch.jit.load(f"{dir}/models/kernel_3_8.pt", map_location=device)

def get_default_model(device):
    global models
    model = models.get(device)
    if model == None:
        model = load_default_model(device)
        models.update({device: model})
    return model


def estimate_area_handcrafted(image: torch.Tensor, strip_count: int=16, feature_thresholds: Sequence[float]=(20, 30, 25), confidence_thresholds: Sequence[float]=(0.03, 0.06)) -> torch.Tensor:
    """
    Estimates the content area for the given endoscopic image(s) using handcrafted feature extraction.
    """
    return ext.estimate_area_handcrafted(image, strip_count, feature_thresholds, confidence_thresholds)


def estimate_area_learned(image: torch.Tensor, strip_count: int=16, model: Optional[torch.jit.ScriptModule]=None,  model_patch_size: int=7, confidence_thresholds: Sequence[float]=(0.03, 0.06)) -> torch.Tensor:
    """
    Estimates the content area for the given endoscopic image(s) using learned feature extraction.
    """
    if model == None:
        model = get_default_model(image.device)
    return ext.estimate_area_learned(image, strip_count, model._c, model_patch_size, confidence_thresholds)


def get_points_handcrafted(image: torch.Tensor, strip_count: int=16, feature_thresholds: Sequence[float]=(20, 30, 25)) -> torch.Tensor:
    """
    Finds candidate edge points and corresponding scores in the given image(s) using handcrafted feature extraction.
    """
    return ext.get_points_handcrafted(image, strip_count, feature_thresholds)


def get_points_learned(image: torch.Tensor, strip_count: int=16, model: Optional[torch.jit.ScriptModule]=None,  model_patch_size: int=7) -> torch.Tensor:
    """
    Finds candidate edge points and corresponding scores in the given image(s) using learned feature extraction.
    """
    if model == None:
        model = get_default_model(image.device)
    return ext.get_points_learned(image, strip_count, model._c, model_patch_size)


def fit_area(points: torch.Tensor, image_size: Sequence[int], confidence_thresholds: Sequence[float]=(0.03, 0.06)) -> torch.Tensor:
    """
    Finds candidate edge points and corresponding scores in the given image(s) using handcrafted feature extraction.
    """
    return ext.fit_area(points, image_size, confidence_thresholds)
