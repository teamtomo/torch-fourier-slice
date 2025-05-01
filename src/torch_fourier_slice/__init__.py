"""Fourier slicing on images and volumes.

Fourier slice slice_extraction/slice_insertion from 2D images and 3D
volumes in PyTorch.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-fourier-slice")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .backproject import backproject_2d_to_3d, backproject_2d_to_3d_batched
from .project import project_2d_to_1d, project_3d_to_2d
from .slice_extraction import (
    extract_central_slices_rfft_2d,
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_batched,
)
from .slice_insertion import (
    insert_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_batched,
)

__all__ = [
    "backproject_2d_to_3d",
    "backproject_2d_to_3d_batched",
    "project_3d_to_2d",
    "project_2d_to_1d",
    "extract_central_slices_rfft_3d",
    "extract_central_slices_rfft_3d_batched",
    "extract_central_slices_rfft_2d",
    "insert_central_slices_rfft_3d",
    "insert_central_slices_rfft_3d_batched",
]
