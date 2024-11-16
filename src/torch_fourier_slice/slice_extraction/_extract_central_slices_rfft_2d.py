import einops
import torch
from torch_image_lerp import sample_image_2d

from ..dft_utils import fftfreq_to_dft_coordinates
from ..grids.central_line_fftfreq_grid import central_line_fftfreq_grid


def extract_central_slices_rfft_2d(
    image_rfft: torch.Tensor,
    image_shape: tuple[int, int],
    rotation_matrices: torch.Tensor,  # (..., 2, 2)
    fftfreq_max: float | None = None,
) -> torch.Tensor:
    """Extract central slice from an fftshifted rfft."""
    # generate grid of DFT sample frequencies for a central slice spanning the x-plane
    freq_grid = central_line_fftfreq_grid(
        image_shape=image_shape,
        rfft=True,
        fftshift=True,
        device=image_rfft.device,
    )  # (w, 2) yx coords

    # keep track of some shapes
    stack_shape = tuple(rotation_matrices.shape[:-2])
    rfft_shape = (freq_grid.shape[-2],)
    output_shape = (*stack_shape, *rfft_shape)

    # get (b, 2, 1) array of yx coordinates to rotate
    if fftfreq_max is not None:
        freq_grid_mask = freq_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]
    else:
        valid_coords = freq_grid
    valid_coords = einops.rearrange(valid_coords, "b yx -> b yx 1")

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
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    # rotate all valid coordinates by each rotation matrix
    rotated_coords = rotation_matrices @ valid_coords  # (..., b, yx, 1)

    # remove last dim of size 1
    rotated_coords = einops.rearrange(rotated_coords, "... b yx 1 -> ... b yx")

    # flip coordinates that ended up in redundant half transform after rotation
    conjugate_mask = rotated_coords[..., 1] < 0
    rotated_coords[conjugate_mask, ...] *= -1

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = fftfreq_to_dft_coordinates(
        frequencies=rotated_coords, image_shape=image_shape, rfft=True
    )  # (...) rfft
    samples = sample_image_2d(image=image_rfft, coordinates=rotated_coords)

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(
        output_shape, device=image_rfft.device, dtype=image_rfft.dtype
    )
    if fftfreq_max is None:
        freq_grid_mask = torch.ones(
            size=rfft_shape, dtype=torch.bool, device=image_rfft.device
        )

    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts
