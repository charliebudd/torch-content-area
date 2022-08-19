"""A PyTorch tool kit for segmenting the endoscopic content area in laparoscopy footage."""

from .infer_area import infer_area_handcrafted as infer_area, infer_area_handcrafted, infer_area_learned

from . import _version
__version__ = _version.get_versions()['version']
