import torch
from typing import Sequence, Optional

def get_points_handcrafted(image: torch.Tensor, strip_count: int=16, feature_thresholds: Sequence[float]=(20, 30, 25)) -> torch.Tensor:
    """
    Finds candidate edge points and corresponding scores in the given image(s) using handcrafted feature extraction.
    """
    raise NotImplementedError()

def get_points_learned(image: torch.Tensor, strip_count: int=16, model: Optional[torch.jit.ScriptModule]=None,  model_patch_size: int=7) -> torch.Tensor:
    """
    Finds candidate edge points and corresponding scores in the given image(s) using learned feature extraction.
    """
    raise NotImplementedError()
