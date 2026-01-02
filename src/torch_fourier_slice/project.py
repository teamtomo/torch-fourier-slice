import torch
import torch.nn.functional as F
from torch_grid_utils import fftfreq_grid

from .slice_extraction import (
    extract_central_slices_rfft_2d,
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multichannel,
)


def compute_cube_face_averages(volume: torch.Tensor, n: int = 1) -> float:
    """Get the average value of all voxels within n-voxels of the cube faces.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, d, d)` volume.
    n: int
        Number of voxels from the cube faces to include in the average.

    Returns
    -------
    float
        Average value of all voxels within n-voxels of the cube faces.
    """
    d, h, w = volume.shape[-3:]

    assert n >= 1, "n must be >= 1"
    assert len({d, h, w}) == 1, "all dimensions of volume must be equal."
    assert n < d // 2, "n must be less than half the volume size."

    total_sum = volume.sum()
    total_voxels = volume.numel()

    # Get the sum of the interior of the cube to avoid double counting
    interior = volume[n:-n, n:-n, n:-n]
    interior_sum = interior.sum()
    interior_voxels = interior.numel()

    face_sum = total_sum - interior_sum
    face_voxels = total_voxels - interior_voxels

    return float((face_sum / face_voxels).item())


def project_3d_to_2d(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad_factor: float = 2.0,
    oversample_factor: int = 2,
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
    pad_factor: float
        Factor determining how much padding to apply to each side of the volume.
        Using `pad_factor=2.0` means going from `(d, d, d)` to `(2d, 2d, 2d)` and
        and likewise `pad_factor=3.0` means going to `(3d, 3d, 3d)`, etc.
        The default value of 2.0 should suffice in most cases. See issue #24
        for more info.
    oversample_factor: int
        Integer factor determining oversampling in Fourier space. Values beyond 1
        can help mitigate aliasing artifacts in the generated projections. The default
        value of 2 should suffice in most cases.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection. If None,
        no masking is applied to limit the frequency content.
    zyx_matrices: bool
        Set to True if the provided matrices left multiply zyx coordinates rather than
        the default xyz coordinates.

    Returns
    -------
    projections: torch.Tensor
        `(..., d, d)` array of projected images.
    """
    d, h, w = volume.shape[-3:]
    if len({d, h, w}) != 1:  # use set to remove duplicates
        raise ValueError("all dimensions of volume must be equal.")

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")

    if oversample_factor < 1 or oversample_factor != int(oversample_factor):
        raise ValueError("oversample_factor must be an integer >= 1")

    # Apply edge padding to the nearest integer matching the desired pad_factor
    if pad_factor > 1.0:
        p = int((w * (pad_factor - 1.0)) // 2)
        edge_value = compute_cube_face_averages(volume, n=4)  # 4 is arbitrary
        volume = F.pad(volume, pad=[p] * 6, mode="constant", value=edge_value)

    # set the shape as a variable
    volume_shape = tuple(volume.shape[-3:])

    # calculate DFT
    # volume center to array origin
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))

    # Track and zero out the DC component
    dft_dc = dft[..., 0, 0, 0].real.item()
    dft[..., 0, 0, 0] = 0.0

    # fftshift the transformed volume so DC is at center
    dft = torch.fft.fftshift(dft, dim=(-3, -2))

    if oversample_factor > 1:
        image_shape = (
            volume_shape[0] * oversample_factor,
            volume_shape[1] * oversample_factor,
            volume_shape[2] * oversample_factor,
        )
    else:
        image_shape = volume_shape

    # make projections by taking central slices
    projections = extract_central_slices_rfft_3d(
        volume_rfft=dft,
        image_shape=image_shape,
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

    # donwmsample if oversampled
    if oversample_factor > 1:
        # Extract out center square based on original size
        w, h = projections.shape[-1], projections.shape[-2]
        start_x = (w - (w // oversample_factor)) // 2
        start_y = (h - (h // oversample_factor)) // 2
        projections = projections[
            ...,
            start_y : start_y + (h // oversample_factor),
            start_x : start_x + (w // oversample_factor),
        ]

    # Account for the subtracted off mean value for the DFT
    mean_value = dft_dc / dft.numel()
    projections += mean_value * d

    # unpad
    if pad_factor > 1.0:
        projections = F.pad(projections, pad=[-p] * 4)

    return projections


def project_3d_to_2d_multichannel(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad_factor: float = 2.0,
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
    projections: torch.Tensor
        `(..., c, d, d)` tensor of multichannel projection images.
    """
    d, h, w = volume.shape[-3:]
    if len({d, h, w}) != 1:  # use set to remove duplicates
        raise ValueError("all dimensions of volume must be equal.")

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")
    if pad_factor > 1.0:
        p = int((volume.shape[-1] * (pad_factor - 1.0)) // 2)
        volume = F.pad(volume, pad=[p] * 6)

    # set the shape as a variable
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

    # unpad
    if pad_factor > 1.0:
        projections = F.pad(projections, pad=[-p] * 4)
    return projections


def project_2d_to_1d(
    image: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad_factor: float = 2.0,
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
    pad_factor: float
        Factor determining the size after padding relative to the original size.
        A pad_factor of 2.0 doubles the box size, 3.0 triples it, etc.
        The default value of 2.0 should suffice in most cases. See issue #24
        for more info.
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
    h, w = image.shape[-2:]
    if h != w:
        raise ValueError("all dimensions of image must be equal.")

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")
    if pad_factor > 1.0:
        p = int((image.shape[-1] * (pad_factor - 1.0)) // 2)
        image = F.pad(image, pad=[p] * 4)

    # set the shape as a variable
    image_shape = tuple(image.shape[-2:])

    # premultiply by sinc2
    grid = fftfreq_grid(
        image_shape=image_shape,
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
        image_shape=image_shape,
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

    # unpad
    if pad_factor > 1.0:
        projections = F.pad(projections, pad=[-p] * 2)

    return projections
