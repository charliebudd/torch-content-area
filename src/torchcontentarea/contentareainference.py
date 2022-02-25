import torch
import _torchcontentareaext as ext

class ContentAreaInference:
    """Entry point for content area inference"""

    def __init__(self) -> None:
        self._ext = ext.ContentAreaInference()

    def infer_mask(self, image: torch.Tensor) -> torch.Tensor:
        """Infers the content area for a given endoscopic image and returns a binary mask"""
        return self._ext.infer_mask(image)

    def infer_area(self, image: torch.Tensor) -> tuple:
        """Infers the content area for a given endoscopic image and returns an content area discription"""
        return self._ext.infer_mask(image)

    def draw_mask(self, image: torch.Tensor, area: tuple) -> torch.Tensor:
        """Returns a binary mask for the provided content area discription"""
        return self._ext.infer_mask(image, area)

    # def crop_to_area(self, image: torch.Tensor, area: tuple, size: tuple) -> torch.Tensor:
    #     """Crops and resizes the image to within the provided content area discription"""
    #     return self._ext.crop_to_area(image, area, size)
