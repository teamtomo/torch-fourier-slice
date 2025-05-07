import einops
import torch
from torch_image_interpolation import insert_into_image_3d

from .._dft_utils import _fftfreq_to_dft_coordinates, _rfft_shape
from .._grids import _central_slice_fftfreq_grid


def insert_central_slices_rfft_3d(
    image_rfft: torch.Tensor,  # fftshifted rfft of (..., h, w) 2d image
    volume_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_matrices = rotation_matrices.to(torch.float32)

    ft_dtype = image_rfft.dtype
    device = image_rfft.device

    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=volume_shape,
        rfft=True,
        fftshift=True,
        device=device,
    )  # (h, w, 3)

    # get (b, 3, 1) array of zyx coordinates to rotate (up to fftfreq_max)
    if fftfreq_max is not None:
        normed_grid = (
            einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        freq_grid_mask = torch.ones(
            size=image_rfft.shape[-2:], dtype=torch.bool, device=device
        )
        valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")
    valid_coords = einops.rearrange(valid_coords, "b zyx -> b zyx 1")

    # get (..., b) array of data at each coordinate from image rffts
    valid_data = image_rfft[..., freq_grid_mask]

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
    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    # rotate all valid coordinates by each rotation matrix and remove last dim
    rotated_coordinates = einops.rearrange(
        rotation_matrices @ valid_coords, pattern="... b zyx 1 -> ... b zyx"
    )

    # flip coordinates in redundant half transform and take conjugate value
    conjugate_mask = rotated_coordinates[..., 2] < 0
    rotated_coordinates[conjugate_mask] *= -1
    valid_data[conjugate_mask] = torch.conj(valid_data[conjugate_mask])

    # calculate positions to sample in DFT array from fftfreq coordinates
    rotated_coordinates = _fftfreq_to_dft_coordinates(
        rotated_coordinates, image_shape=volume_shape, rfft=True
    )

    # initialise output volume and volume for keeping track of weights
    dft_3d = torch.zeros(
        size=_rfft_shape(volume_shape),
        dtype=ft_dtype,
        device=device,
    )
    weights = torch.zeros_like(
        dft_3d,
        dtype=torch.float32 if ft_dtype == torch.complex64 else torch.float64,
        device=device,
    )

    # insert data into 3D DFT
    dft_3d, weights = insert_into_image_3d(
        values=valid_data,
        coordinates=rotated_coordinates,
        image=dft_3d,
        weights=weights,
        interpolation="trilinear",
    )
    return dft_3d, weights


def insert_central_slices_rfft_3d_multichannel(
    image_rfft: torch.Tensor,  # fftshifted rfft of (..., c, d, d) 2d image
    volume_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3) dims need to match rfft
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_matrices = rotation_matrices.to(torch.float32)
    ft_dtype = image_rfft.dtype
    device = image_rfft.device
    channels = image_rfft.shape[-3]

    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=volume_shape,
        rfft=True,
        fftshift=True,
        device=device,
    )  # (d, d, d, 3)

    # get (b, 3, 1) array of zyx coordinates to rotate (up to fftfreq_max)
    if fftfreq_max is not None:
        normed_grid = (
            einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        freq_grid_mask = torch.ones(
            size=image_rfft.shape[-2:], dtype=torch.bool, device=device
        )
        valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")

    valid_coords = einops.rearrange(valid_coords, "hw zyx -> hw zyx 1")

    # get (..., b) array of data at each coordinate from image rffts
    valid_data = image_rfft[..., freq_grid_mask]

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
    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    # rotate all valid coordinates by each rotation matrix and remove last dim
    rotated_coordinates = einops.rearrange(
        rotation_matrices @ valid_coords, pattern="... hw zyx 1 -> ... hw zyx"
    )

    # flip coordinates in redundant half transform and take conjugate value
    conjugate_mask = rotated_coordinates[..., 2] < 0
    rotated_coordinates[conjugate_mask] *= -1
    # switch channel to end for torch-image-interpolation
    valid_data = einops.rearrange(valid_data, "... c hw -> ... hw c")
    valid_data[conjugate_mask] = torch.conj(valid_data[conjugate_mask])

    # calculate positions to sample in DFT array from fftfreq coordinates
    rotated_coordinates = _fftfreq_to_dft_coordinates(
        rotated_coordinates, image_shape=volume_shape, rfft=True
    )

    # initialise output volume and volume for keeping track of weights
    volume_dft_shape = _rfft_shape(volume_shape)
    dft_3d = torch.zeros(
        size=(channels, *volume_dft_shape),
        dtype=ft_dtype,
        device=device,
    )
    weights = torch.zeros(
        size=volume_dft_shape,
        dtype=torch.float32 if ft_dtype == torch.complex64 else torch.float64,
        device=device,
    )

    # insert data into 3D DFT
    dft_3d, weights = insert_into_image_3d(
        values=valid_data,
        coordinates=rotated_coordinates,
        image=dft_3d,
        weights=weights,
        interpolation="trilinear",
    )
    return dft_3d, weights
