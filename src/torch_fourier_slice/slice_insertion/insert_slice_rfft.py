import einops
import torch

from ..dft_utils import fftfreq_to_dft_coordinates, rfft_shape
from ..grids.central_slice_grid import central_slice_grid
from ..interpolation import insert_into_dft_3d


def insert_central_slices_rfft_3d(
    image_rfft: torch.Tensor,  # fftshifted rfft of (..., h, w) 2d image
    volume_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,
    fftfreq_max: float | None = None
):
    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = central_slice_grid(
        volume_shape=volume_shape,
        rfft=True,
        fftshift=True,
        device=image_rfft.device,
    )  # (h, w, 3)

    # get (b, 3, 1) array of zyx coordinates to rotate (up to fftfreq_max)
    if fftfreq_max is not None:
        normed_grid = einops.reduce(freq_grid ** 2, 'h w zyx -> h w', reduction='sum') ** 0.5
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        valid_coords = einops.rearrange(freq_grid, 'h w zyx -> (h w) zyx')
    valid_coords = einops.rearrange(valid_coords, 'b zyx -> b zyx 1')

    # get (..., b) array of data at each coordinate from image rffts
    valid_data = image_rfft[..., freq_grid_mask]

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, '... i j -> ... 1 i j')

    # rotate all valid coordinates by each rotation matrix and remove last dim
    rotated_coords = einops.rearrange(
        rotation_matrices @ valid_coords, pattern='... b zyx 1 -> ... b zyx'
    )

    # flip coordinates in redundant half transform and take conjugate value
    conjugate_mask = rotated_coords[..., 2] < 0
    rotated_coords[conjugate_mask] *= -1
    valid_data[conjugate_mask] = torch.conj(valid_data[conjugate_mask])

    # calculate actual coordinates in DFT array from fftfreq coordinates
    valid_coords = fftfreq_to_dft_coordinates(
        valid_coords, image_shape=volume_shape, rfft=True
    )

    # initialise output volume and volume for keeping track of weights
    dft_3d = torch.zeros(
        size=rfft_shape(volume_shape), dtype=torch.complex64, device=image_rfft.device
    )
    weights = torch.zeros_like(dft_3d, dtype=torch.float64)

    # insert data into 3D DFT
    dft_3d, weights = insert_into_dft_3d(
        data=valid_data,
        coordinates=valid_coords,
        dft=dft_3d,
        weights=weights
    )
    return dft_3d, weights
