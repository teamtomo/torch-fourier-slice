import torch
import torch.nn.functional as F
from torch_grid_utils import fftfreq_grid

from .slice_insertion import (
    insert_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multichannel,
)


def backproject_2d_to_3d(
    images: torch.Tensor,  # (b, d, d)
    rotation_matrices: torch.Tensor,  # (b, 3, 3)
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:
    """Perform a 3D reconstruction from a set of 2D projection images.

    Parameters
    ----------
    images: torch.Tensor
        `(..., d, d)` array of 2D projection images.
    rotation_matrices: torch.Tensor
        `(..., 3, 3)` array of rotation matrices for insert of `images`.
        Rotation matrices left-multiply column vectors containing xyz coordinates.
    pad_factor: float
        Factor determining the size after padding relative to the original size.
        A pad_factor of 2.0 doubles the box size, 3.0 triples it, etc.
        The default value of 2.0 should suffice in most cases. See issue #24
        for more info.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection.
    zyx_matrices: bool
        Set to True if the provided matrices left multiply zyx column vectors
        instead of xyz column vectors.

    Returns
    -------
    reconstruction: torch.Tensor
        `(d, d, d)` cubic volume containing the 3D reconstruction from
        `images`.
    """
    h, w = images.shape[-2:]
    if h != w:
        raise ValueError("images must be square.")

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")
    if pad_factor > 1.0:
        p = int((images.shape[-1] * (pad_factor - 1.0)) // 2)
        images = F.pad(images, pad=[p] * 4)

    # construct shapes
    h, w = images.shape[-2:]
    volume_shape = (w, w, w)

    # calculate DFTs of images
    images = torch.fft.fftshift(images, dim=(-2, -1))  # volume center to array origin
    images = torch.fft.rfftn(images, dim=(-2, -1))
    images = torch.fft.fftshift(images, dim=(-2,))  # actual fftshift

    # insert image DFTs into a 3D rfft as central slices
    dft, weights = insert_central_slices_rfft_3d(
        image_rfft=images,
        volume_shape=volume_shape,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        zyx_matrices=zyx_matrices,
    )

    # reweight reconstruction
    valid_weights = weights > 1e-3
    dft[valid_weights] /= weights[valid_weights]

    # back to real space
    dft = torch.fft.ifftshift(dft, dim=(-3, -2))  # actual ifftshift
    dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))  # center in real space

    # correct for convolution with linear interpolation kernel
    grid = fftfreq_grid(
        image_shape=dft.shape, rfft=False, fftshift=True, norm=True, device=dft.device
    )
    dft = dft / torch.sinc(grid) ** 2

    # unpad
    if pad_factor > 1.0:
        dft = F.pad(dft, pad=[-p] * 6)
    return torch.real(dft)


def backproject_2d_to_3d_multichannel(
    images: torch.Tensor,  # (..., c, d, d)
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:
    """Perform a 3D reconstruction from multichannel 2D projection images.

    Parameters
    ----------
    images: torch.Tensor
        `(..., c, d, d)` multichannel tensors of 2D projection images.
    rotation_matrices: torch.Tensor
        `(..., 3, 3)` array of rotation matrices to insert each channel of
        `images`. Rotation matrices left-multiply column vectors containing
        xyz coordinates.
    pad_factor: float
        Factor determining the size after padding relative to the original size.
        A pad_factor of 2.0 doubles the box size, 3.0 triples it, etc.
        The default value of 2.0 should suffice in most cases. See issue #24
        for more info.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection.
    zyx_matrices: bool
        Set to True if the provided matrices left multiply zyx column vectors
        instead of xyz column vectors.

    Returns
    -------
    reconstruction: torch.Tensor
        `(c, d, d, d)` multichannel cubic volume containing the 3D
        reconstructions.
    """
    h, w = images.shape[-2:]
    if h != w:
        raise ValueError("images must be square.")

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")
    if pad_factor > 1.0:
        p = int((images.shape[-1] * (pad_factor - 1.0)) // 2)
        images = F.pad(images, pad=[p] * 4)

    # construct shapes
    h, w = images.shape[-2:]
    volume_shape = (w, w, w)

    # calculate DFTs of images
    images = torch.fft.fftshift(images, dim=(-2, -1))  # volume center to array origin
    images = torch.fft.rfftn(images, dim=(-2, -1))
    images = torch.fft.fftshift(images, dim=(-2,))  # actual fftshift

    # insert image DFTs into a 3D rfft as central slices
    dft, weights = insert_central_slices_rfft_3d_multichannel(
        image_rfft=images,
        volume_shape=volume_shape,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        zyx_matrices=zyx_matrices,
    )

    # reweight reconstruction
    valid_weights = weights > 1e-3
    dft[..., valid_weights] /= weights[..., valid_weights]

    # back to real space
    dft = torch.fft.ifftshift(dft, dim=(-3, -2))  # actual ifftshift
    dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))  # center in real space

    # correct for convolution with linear interpolation kernel
    grid = fftfreq_grid(
        image_shape=dft.shape[-3:],
        rfft=False,
        fftshift=True,
        norm=True,
        device=dft.device,
    )
    dft = dft / torch.sinc(grid) ** 2

    # unpad
    if pad_factor > 1.0:
        dft = F.pad(dft, pad=[-p] * 6)
    return torch.real(dft)
