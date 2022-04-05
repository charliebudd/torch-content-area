import torch
from enum import IntEnum
from typing import Sequence

import __torchcontentareaext as __ext

class InterpolationMode(IntEnum):
    """A tag to specify the type of interpolation when cropping"""
    NEAREST = 0
    BILINEAR = 1

class ContentAreaInference(__ext.ContentAreaInference):
    """Entry point for content area inference"""
    def __init__(self) -> None:
        super().__init__()

    def infer_mask(self, image: torch.Tensor) -> torch.Tensor:
        """Infers the content area for a given endoscopic image and returns a binary mask"""
        return self.__infer_mask(image)

    def infer_area(self, image: torch.Tensor) -> Sequence[int]:
        """Infers the content area for a given endoscopic image and returns the parameters of the content area (None if no area found)"""
        return self.__infer_area(image)

    def draw_mask(self, image: torch.Tensor, area: Sequence[int]) -> torch.Tensor:
        """Returns a binary mask for the provided content area parameters"""
        return self.__draw_mask(image, area)

    def crop_area(self, image: torch.Tensor, area: Sequence[int], size: Sequence[int], interpolation_mode: InterpolationMode=InterpolationMode.BILINEAR) -> torch.Tensor:
        """Crops and resizes the image to within the provided content area"""
        return self.__crop_area(image, area, size, int(interpolation_mode))

    def get_points(self, image: torch.Tensor) -> Sequence[Sequence[int]]:
        """Returns a list of candidate border points, mainly for debugging purposes"""
        return self.__get_points(image)
