import torch
from enum import IntEnum
from typing import Sequence

import __torchcontentareaext as __ext

class InterpolationMode(IntEnum):
    """A tag to specify the type of interpolation when cropping"""
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2


class ContentAreaType(IntEnum):
    """A tag to specify the type of content area"""
    NONE = 0
    CIRCLE = 1


class ContentArea(__ext.ContentArea):
    """A description of an images content area"""
    def __init__(self, x, y, r) -> None:
        super().__init__(x, y, r)

    @property
    def type(self): return ContentAreaType(self.__type)

    @property 
    def x(self): return self.__x
    @property 
    def y(self): return self.__y
    @property 
    def r(self): return self.__r
    
    @x.setter 
    def x(self, v): self.__x = v
    @y.setter 
    def y(self, v): self.__y = v
    @r.setter 
    def r(self, v): self.__r = v


class ContentAreaInference(__ext.ContentAreaInference):
    """Entry point for content area inference"""
    def __init__(self) -> None:
        super().__init__()

    def infer_mask(self, image: torch.Tensor) -> torch.Tensor:
        """Infers the content area for a given endoscopic image and returns a binary mask"""
        return self.__infer_mask(image)

    def infer_area(self, image: torch.Tensor) -> ContentArea:
        """Infers the content area for a given endoscopic image and returns an content area discription"""
        return self.__infer_area(image)

    def draw_mask(self, image: torch.Tensor, area: ContentArea) -> torch.Tensor:
        """Returns a binary mask for the provided content area discription"""
        return self.__draw_mask(image, area)

    def crop_area(self, image: torch.Tensor, area: ContentArea, size: Sequence[int], interpolation_mode: InterpolationMode=InterpolationMode.BILINEAR) -> torch.Tensor:
        """Crops and resizes the image to within the provided content area discription"""
        return self.__crop_area(image, area, size, int(interpolation_mode))

    def get_points(self, image: torch.Tensor) -> Sequence[Sequence[int]]:
        """Returns a list of candidate border points, mainly for debugging purposes"""
        return self.__get_points(image)
