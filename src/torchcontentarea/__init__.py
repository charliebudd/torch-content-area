"""A PyTorch tool kit for segmenting the endoscopic content area in laparoscopy footage."""

from .extension_wrapper import estimate_area_handcrafted as estimate_area, estimate_area_handcrafted, estimate_area_learned
from .extension_wrapper import get_points_handcrafted as get_points, get_points_handcrafted, get_points_learned
from .extension_wrapper import fit_area

from . import _version
__version__ = _version.get_versions()['version']
