"""Fourier slice extraction/insertion from 2D images and 3D volumes in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-fourier-slice")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

__all__ = [
    "backproject_2d_to_3d",
    "project_3d_to_2d",
    "extract_central_slices_rfft_3d",
    "insert_central_slices_rfft_3d",
]

from torch_fourier_slice.backproject import backproject_2d_to_3d
from torch_fourier_slice.project import project_3d_to_2d
from torch_fourier_slice.slice_extraction._extract_central_slices_rfft_3d import (
    extract_central_slices_rfft_3d,
)
from torch_fourier_slice.slice_insertion._insert_central_slices_rfft_3d import (
    insert_central_slices_rfft_3d,
)
