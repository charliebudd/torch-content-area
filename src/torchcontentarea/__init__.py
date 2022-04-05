"""A PyTorch tool kit for segmenting the endoscopic content area in laparoscopy footage."""

from .contentareainference import ContentAreaInference, InterpolationMode

from . import _version
__version__ = _version.get_versions()['version']
