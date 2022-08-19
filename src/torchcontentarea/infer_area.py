import torch
import torchcontentareaext as ext
from typing import Sequence

def load_default_model():
    dir = "/".join(__file__.split("/")[:-1])
    return torch.jit.load(f"{dir}/models/kernel_3_8.pt")

def infer_area_handcrafted(image: torch.Tensor, strip_count: int=16, feature_thresholds: Sequence[float]=(20, 30, 25), confidence_thresholds: Sequence[float]=(0.03, 0.06)) -> torch.Tensor:
    """
    Infers the content area for a batch of endoscopic images using handcrafted feature extraction.
    Feature thresholds for edge strength, edge angle, and border intensity. 
    """
    return ext.infer_area_handcrafted(image, strip_count, feature_thresholds, confidence_thresholds)

def infer_area_learned(image: torch.Tensor, strip_count: int=16, model: torch.jit.ScriptModule=load_default_model(),  model_patch_size: int=7, confidence_thresholds: Sequence[float]=(0.03, 0.06)) -> torch.Tensor:
    """
    Infers the content area for a batch of endoscopic images using learned feature extraction.
    """
    return ext.infer_area_learned(image, strip_count, model._c, model_patch_size, confidence_thresholds)
