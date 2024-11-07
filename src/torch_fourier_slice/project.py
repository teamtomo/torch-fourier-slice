"""Module for 3D to 2D projection operations using Fourier slice theorem.

This module provides functionality to project 3D volumes into 2D images using
the Fourier slice theorem, which states that the 2D Fourier transform of a
projection is equal to a central slice through the 3D Fourier transform of
the volume.
"""

import torch
import torch.nn.functional as F
from torch_grid_utils import fftfreq_grid

from torch_fourier_slice.slice_extraction._extract_central_slices_rfft_3d import (
    extract_central_slices_rfft_3d,
)


def project_3d_to_2d(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad: bool = True,
    fftfreq_max: float | None = None,
    edge_pad_fraction: float = 0.0,
) -> torch.Tensor:
    """Project a cubic volume by sampling a central slice through its DFT.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, d, d)` volume.
    rotation_matrices: torch.Tensor
        `(..., 3, 3)` array of rotation matrices for insertion of `images`.
        Rotation matrices left-multiply column vectors containing xyz coordinates.
    pad: bool
        Whether to pad the volume 2x with zeros to increase sampling rate in the DFT.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection.
    edge_pad_fraction: float
        Fraction of the dft edge to pad with reflective padding.
        This is useful for reducing artifacts from the DFT edge.
        Recommended values are 0.1 - 0.25.

    Returns
    -------
    projections: torch.Tensor
        `(..., d, d)` array of projection images.
    """
    # padding
    if pad is True:
        pad_length = volume.shape[-1] // 2
        volume = F.pad(volume, pad=[pad_length] * 6, mode="constant", value=0)

    # premultiply by sinc2
    grid = fftfreq_grid(
        image_shape=volume.shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=volume.device,
    )
    volume = volume * torch.sinc(grid) ** 2

    # calculate DFT
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.fftshift(
        dft,
        dim=(
            -3,
            -2,
        ),
    )  # actual fftshift of 3D rfft

    # make projections by taking central slices
    projections = extract_central_slices_rfft_3d(
        volume_rfft=dft,
        image_shape=volume.shape,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
    )  # (..., h, w) rfft stack

    # edge padding
    if edge_pad_fraction > 0:
        # Use the longer dimension (height) for calculating pad size
        edge_pad = int(projections.shape[-2] * edge_pad_fraction)
        # For rfft, pad only left, right, and top (not bottom)
        projections = torch.nn.functional.pad(
            projections,
            pad=[
                edge_pad,  # left
                0,  # right (no pad as rfft is symmetric)
                edge_pad,  # top
                edge_pad,  # bottom
            ],
            mode="reflect",
        )

    # transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of 2D rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(
        projections, dim=(-2, -1)
    )  # recenter 2D image in real space

    # unpad if required
    if pad is True:
        projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]
    if edge_pad_fraction > 0:
        projections = projections[..., edge_pad:-edge_pad, edge_pad:-edge_pad]

    return torch.real(projections)
