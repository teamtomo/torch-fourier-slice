import torch
import torch.nn.functional as F
from torch_grid_utils import fftfreq_grid

from .slice_extraction import (
    extract_central_slices_rfft_2d,
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multichannel,
)


def project_3d_to_2d(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad: bool = True,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:
    """Project a cubic volume by sampling a central slice through its DFT.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, d, d)` volume.
    rotation_matrices: torch.Tensor
        `(..., 3, 3)` array of rotation matrices for extraction of `images`.
        Rotation matrices left-multiply column vectors containing xyz coordinates.
    pad: bool
        Whether to pad the volume 2x with zeros to increase sampling rate in the DFT.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection.
    zyx_matrices: bool
        Set to True if the provided matrices left multiply zyx column vectors
        instead of xyz column vectors.

    Returns
    -------
    projections: torch.Tensor
        `(..., d, d)` array of projection images.
    """
    d, h, w = volume.shape[-3:]
    if len({d, h, w}) != 1:  # use set to remove duplicates
        raise ValueError("all dimensions of volume must be equal.")
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
    # volume center to array origin
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    # actual fftshift of 3D rfft
    dft = torch.fft.fftshift(dft, dim=(-3, -2))

    # make projections by taking central slices
    projections = extract_central_slices_rfft_3d(
        volume_rfft=dft,
        image_shape=volume.shape,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        zyx_matrices=zyx_matrices,
    )  # (..., h, w) rfft stack

    # transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of 2D rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(
        projections, dim=(-2, -1)
    )  # recenter 2D image in real space

    # unpad if required
    if pad is True:
        projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]
    return torch.real(projections)


def project_3d_to_2d_multichannel(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad: bool = True,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:
    """Project a multichannel cubic volume with the same rotations.

    Parameters
    ----------
    volume: torch.Tensor
        `(c, d, d, d)` volume.
    rotation_matrices: torch.Tensor
        `(..., 3, 3)` array of rotation matrices for extraction of `images`.
        Rotation matrices left-multiply column vectors containing xyz coordinates.
    pad: bool
        Whether to pad the volume 2x with zeros to increase sampling rate in the DFT.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection.
    zyx_matrices: bool
        Set to True if the provided matrices left multiply zyx column vectors
        instead of xyz column vectors.

    Returns
    -------
    projections: torch.Tensor
        `(..., c, d, d)` tensor of multichannel projection images.
    """
    d, h, w = volume.shape[-3:]
    if len({d, h, w}) != 1:  # use set to remove duplicates
        raise ValueError("all dimensions of volume must be equal.")
    # padding
    if pad is True:
        pad_length = volume.shape[-1] // 2
        volume = F.pad(volume, pad=[pad_length] * 6, mode="constant", value=0)

    volume_shape = tuple(volume.shape[-3:])

    # premultiply by sinc2
    grid = fftfreq_grid(
        image_shape=volume_shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=volume.device,
    )
    volume = volume * torch.sinc(grid) ** 2

    # calculate DFT
    # volume center to array origin
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    # actual fftshift of 3D rfft
    dft = torch.fft.fftshift(dft, dim=(-3, -2))

    # make projections by taking central slices
    projections = extract_central_slices_rfft_3d_multichannel(
        volume_rfft=dft,
        image_shape=volume_shape,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        zyx_matrices=zyx_matrices,
    )  # (..., h, w) rfft stack

    # transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of 2D rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(
        projections, dim=(-2, -1)
    )  # recenter 2D image in real space

    # unpad if required
    if pad is True:
        projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]
    return torch.real(projections)


def project_2d_to_1d(
    image: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad: bool = True,
    fftfreq_max: float | None = None,
    yx_matrices: bool = False,
) -> torch.Tensor:
    """Project a square image by sampling a central line through its DFT.

    Parameters
    ----------
    image: torch.Tensor
        `(d, d)` image.
    rotation_matrices: torch.Tensor
        `(..., 2, 2)` array of rotation matrices for extraction of `lines`.
        Rotation matrices left-multiply column vectors containing xy coordinates.
    pad: bool
        Whether to pad the volume 2x with zeros to increase sampling rate in the DFT.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection.
    yx_matrices: bool
        Set to True if the provided matrices left multiply yx column vectors
        instead of xy column vectors.

    Returns
    -------
    projections: torch.Tensor
        `(..., d)` array of projected lines.
    """
    h, w = image.shape[-3:]
    if len({h, w}) != 1:  # use set to remove duplicates
        raise ValueError("all dimensions of image must be equal.")
    # padding
    if pad is True:
        pad_length = image.shape[-1] // 2
        image = F.pad(image, pad=[pad_length] * 4, mode="constant", value=0)

    # premultiply by sinc2
    grid = fftfreq_grid(
        image_shape=image.shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=image.device,
    )
    image = image * torch.sinc(grid) ** 2

    # calculate DFT
    dft = torch.fft.fftshift(image, dim=(-2, -1))  # image center to array origin
    dft = torch.fft.rfftn(dft, dim=(-2, -1))
    dft = torch.fft.fftshift(dft, dim=(-2,))  # actual fftshift of 2D rfft

    # make projections by taking central slices
    projections = extract_central_slices_rfft_2d(
        image_rfft=dft,
        image_shape=image.shape,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        yx_matrices=yx_matrices,
    )  # (..., w) rfft stack

    # transform back to real space
    # not needed for 1D: torch.fft.ifftshift(projections, dim=(-2,))
    projections = torch.fft.irfftn(projections, dim=(-1))
    projections = torch.fft.ifftshift(
        projections, dim=(-1)
    )  # recenter line in real space

    # unpad if required
    if pad is True:
        projections = projections[..., pad_length:-pad_length]
    return torch.real(projections)
