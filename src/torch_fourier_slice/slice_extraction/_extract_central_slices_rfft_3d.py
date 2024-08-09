import torch
import einops
from torch_image_lerp import sample_image_3d

from ..dft_utils import fftfreq_to_dft_coordinates
from ..grids.central_slice_fftfreq_grid import central_slice_fftfreq_grid


def extract_central_slices_rfft_3d(
    volume_rfft: torch.Tensor,
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
):
    """Extract central slice from an fftshifted rfft."""
    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = central_slice_fftfreq_grid(
        volume_shape=image_shape,
        rfft=True,
        fftshift=True,
        device=volume_rfft.device,
    )  # (h, w, 3) zyx coords

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

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]    [ax + by + cz]
    # [d e f] [y]  = [dx + ey + fz]
    # [g h i] [z]    [gx + hy + iz]
    #
    # zyx:
    # [i h g] [z]    [gx + hy + iz]
    # [f e d] [y]  = [dx + ey + fz]
    # [c b a] [x]    [ax + by + cz]
    rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

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
    samples = sample_image_3d(image=volume_rfft, coordinates=rotated_coords)  # (...) rfft

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype)
    if fftfreq_max is None:
        freq_grid_mask = torch.ones(size=rfft_shape, dtype=torch.bool, device=volume_rfft.device)

    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts
