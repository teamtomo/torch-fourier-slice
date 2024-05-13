import torch
import torch.nn.functional as F
import einops

from ..grids import fftfreq_grid
from ..dft_utils import fftfreq_to_dft_coordinates
from ..grids.central_slice_grid import central_slice_grid
from ..interpolation import sample_dft_3d


def project_fourier(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad: bool = True,
    fftfreq_max: float | None = None,

) -> torch.Tensor:
    """Project a cubic volume by sampling a central slice through its DFT.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, d, d)` volume.
    rotation_matrices: torch.Tensor
        `(..., 3, 3)` array of matrices which rotate coordinates (zyx) of the
        central slice to be sampled.
    pad: bool
        Whether to pad the volume with zeros to increase sampling in the DFT.

    Returns
    -------
    projections: torch.Tensor
        `(..., d, d)` array of projection images.
    """
    # padding
    if pad is True:
        pad_length = volume.shape[-1] // 2
        volume = F.pad(volume, pad=[pad_length] * 6, mode='constant', value=0)

    # premultiply by sinc2
    grid = fftfreq_grid(
        image_shape=volume.shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=volume.device
    )
    volume = volume * torch.sinc(grid) ** 2

    # calculate DFT
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.fftshift(dft, dim=(-3, -2,))  # actual fftshift of 3D rfft

    # make projections by taking central slices
    projections = extract_central_slices_rfft(
        volume_rfft=dft,
        image_shape=volume.shape,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max
    )  # (..., h, w) rfft stack

    # transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of 2D rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter 2D image in real space

    # unpad if required
    if pad is True:
        projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]

    return torch.real(projections)


def extract_central_slices_rfft(
    volume_rfft: torch.Tensor,
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
):
    """Extract central slice from an fftshifted rfft."""
    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = central_slice_grid(
        volume_shape=image_shape,
        rfft=True,
        fftshift=True,
        device=volume_rfft.device,
    )  # (h, w, 3)

    # keep track of some shapes
    stack_shape = tuple(rotation_matrices.shape[:-2])
    rfft_shape = freq_grid.shape[-3], freq_grid.shape[-2]
    output_shape = (*stack_shape, *rfft_shape)

    # get (b, 3, 1) array of zyx coordinates to rotate
    if fftfreq_max is not None:
        normed_grid = einops.reduce(freq_grid ** 2, 'h w zyx -> h w', reduction='sum') ** 0.5
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        valid_coords = einops.rearrange(freq_grid, 'h w zyx -> (h w) zyx')
    valid_coords = einops.rearrange(valid_coords, 'b zyx -> b zyx 1')

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, '... i j -> ... 1 i j')

    # rotate all valid coordinates by each rotation matrix
    rotated_coords = rotation_matrices @ valid_coords  # (..., b, zyx, 1)

    # remove last dim of size 1
    rotated_coords = einops.rearrange(rotated_coords, '... b zyx 1 -> ... b zyx')

    # flip coordinates that ended up in redundant half transform after rotation
    conjugate_mask = rotated_coords[..., 2] < 0
    rotated_coords[conjugate_mask, ...] *= -1

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = fftfreq_to_dft_coordinates(
        frequencies=rotated_coords,
        image_shape=image_shape,
        rfft=True
    )
    samples = sample_dft_3d(dft=volume_rfft, coordinates=rotated_coords)  # (...) rfft

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype)
    if fftfreq_max is None:
        freq_grid_mask = torch.ones(size=rfft_shape, dtype=torch.bool, device=volume_rfft.device)

    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts
