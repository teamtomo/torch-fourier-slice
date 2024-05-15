"""Fourier slice slice_extraction/slice_insertion from 2D images and 3D volumes in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-fourier-slice")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .project import project_3d_to_2d
from .backproject import backproject_2d_to_3d
from .slice_insertion import insert_central_slices_rfft_3d
from .slice_extraction import extract_central_slices_rfft_3d
