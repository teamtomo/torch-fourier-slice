import einops
import torch
from torch_image_interpolation import sample_image_2d, sample_image_3d
from torch_grid_utils.fftfreq_grid import fftfreq_grid, transform_fftfreq_grid

from .._dft_utils import _fftfreq_to_dft_coordinates
from .._grids import _central_slice_fftfreq_grid


def extract_central_slices_rfft_3d(
    volume_rfft: torch.Tensor,
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:
    """Extract central slice from an fftshifted rfft."""
    rotation_matrices = rotation_matrices.to(torch.float32)

    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = _central_slice_fftfreq_grid(
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
        normed_grid = (
            einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        freq_grid_mask = torch.ones(
            size=rfft_shape, dtype=torch.bool, device=volume_rfft.device
        )
        valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")
    valid_coords = einops.rearrange(valid_coords, "b zyx -> b zyx 1")

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]   [ax + by + cz]   [x']
    # [d e f] [y]   [dx + ey + fz]   [y']
    # [g h i] [z] = [gx + hy + iz] = [z']
    #
    # zyx:
    # [i h g] [z]   [gx + hy + iz]   [z']
    # [f e d] [y]   [dx + ey + fz]   [y']
    # [c b a] [x] = [ax + by + cz] = [x']
    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    # rotate all valid coordinates by each rotation matrix
    rotated_coords = rotation_matrices @ valid_coords  # (..., b, zyx, 1)

    # remove last dim of size 1
    rotated_coords = einops.rearrange(rotated_coords, "... b zyx 1 -> ... b zyx")

    # flip coordinates that ended up in redundant half transform after rotation
    conjugate_mask = rotated_coords[..., 2] < 0
    rotated_coords[conjugate_mask, ...] *= -1

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = _fftfreq_to_dft_coordinates(
        frequencies=rotated_coords, image_shape=image_shape, rfft=True
    )
    samples = sample_image_3d(
        image=volume_rfft, coordinates=rotated_coords, interpolation="trilinear"
    )  # (...) rfft

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(
        output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype
    )
    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts


def extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,  # (c, d, d, d)
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:  # (..., c, h, w)
    """Extract central slice from an fftshifted rfft."""
    rotation_matrices = rotation_matrices.to(torch.float32)

    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=image_shape,
        rfft=True,
        fftshift=True,
        device=volume_rfft.device,
    )  # (h, w, 3) zyx coords

    # keep track of some shapes
    channels = volume_rfft.shape[0]
    stack_shape = tuple(rotation_matrices.shape[:-2])
    rfft_shape = freq_grid.shape[-3], freq_grid.shape[-2]
    output_shape = (*stack_shape, channels, *rfft_shape)

    # get (b, 3, 1) array of zyx coordinates to rotate
    if fftfreq_max is not None:
        normed_grid = (
            einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        freq_grid_mask = torch.ones(
            size=rfft_shape, dtype=torch.bool, device=volume_rfft.device
        )
        valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")
    valid_coords = einops.rearrange(valid_coords, "hw zyx -> hw zyx 1")

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]   [ax + by + cz]   [x']
    # [d e f] [y]   [dx + ey + fz]   [y']
    # [g h i] [z] = [gx + hy + iz] = [z']
    #
    # zyx:
    # [i h g] [z]   [gx + hy + iz]   [z']
    # [f e d] [y]   [dx + ey + fz]   [y']
    # [c b a] [x] = [ax + by + cz] = [x']
    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    # rotate all valid coordinates by each rotation matrix
    rotated_coords = rotation_matrices @ valid_coords  # (..., b, zyx, 1)

    # remove last dim of size 1
    rotated_coords = einops.rearrange(rotated_coords, "... hw zyx 1 -> ... hw zyx")

    # flip coordinates that ended up in redundant half transform after rotation
    conjugate_mask = rotated_coords[..., 2] < 0
    rotated_coords[conjugate_mask, ...] *= -1

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = _fftfreq_to_dft_coordinates(
        frequencies=rotated_coords, image_shape=image_shape, rfft=True
    )
    samples = sample_image_3d(
        image=volume_rfft, coordinates=rotated_coords, interpolation="trilinear"
    )  # shape is (..., c)

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])
    samples = einops.rearrange(samples, "... hw c -> ... c hw")

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(
        output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype
    )
    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts


def transform_slice_2d(
    projection_image_dfts: torch.Tensor,
    rfft_shape: tuple[int, int],
    stack_shape: tuple[int, ...],
    transform_matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply anisotropic magnification transformation to a 2D slice in Fourier space.

    Parameters
    ----------
    projection_image_dfts: torch.Tensor
        Projection image DFTs with shape `(*stack_shape, h, w)` where `(h, w)`
        is the rfft shape.
    rfft_shape: tuple[int, int]
        Shape of the rfft `(h, w_rfft)` where `w_rfft = n//2 + 1`.
    stack_shape: tuple[int, ...]
        Shape of batch dimensions (can be empty for single image).
    transform_matrix: torch.Tensor
        `(2, 2)` anisotropic magnification matrix in real space (yx ordering).

    Returns
    -------
    transformed_dfts: torch.Tensor
        Transformed projection image DFTs with same shape as input.
    """
    # Calculate full 2D image shape from rfft shape
    # rfft_shape is (h, w_rfft) where w_rfft = n//2 + 1
    # Full width is w_full = 2 * (w_rfft - 1) = n
    h_full = rfft_shape[0]
    w_full = 2 * (rfft_shape[1] - 1)
    image_shape_2d = (h_full, w_full)

    # Generate 2D frequency grid
    freq_grid = fftfreq_grid(
        image_shape=image_shape_2d,
        rfft=True,
        fftshift=True,
        device=projection_image_dfts.device,
    )  # (h, w, 2) yx coords

    # Apply transformation matrix
    transformed_freqs = transform_fftfreq_grid(
        frequency_grid=freq_grid,
        real_space_matrix=transform_matrix,
        device=projection_image_dfts.device,
    )  # (h, w, 2) yx coords

    # Convert transformed frequencies to DFT array coordinates
    transformed_coords = _fftfreq_to_dft_coordinates(
        frequencies=transformed_freqs,
        image_shape=image_shape_2d,
        rfft=True
    )

    # Flatten coordinates: (h, w, 2) -> (h*w, 2)
    transformed_coords_flat = einops.rearrange(
        transformed_coords, "h w yx -> (h w) yx"
    )

    # Resample each 2D DFT at the transformed frequencies
    # Need to handle batch dimensions in stack_shape
    if len(stack_shape) > 0:
        # Flatten batch dimensions for processing
        projection_image_dfts_flat = einops.rearrange(
            projection_image_dfts, "... h w -> (...) h w"
        )

        # Use channel dimension as batch: pass (b, h, w) as (c=b, h, w)
        resampled = sample_image_2d(
            image=projection_image_dfts_flat,  # (b, h, w) treated as (c=b, h, w)
            coordinates=transformed_coords_flat,  # (h*w, 2)
            interpolation="bilinear"
        )  # Returns (h*w, b) - samples for each "channel" (batch)

        # Transpose to get (b, h*w)
        n_coords = transformed_coords_flat.shape[0]
        resampled = einops.rearrange(
            resampled, "n_coords b -> b n_coords", n_coords=n_coords
        )
        # Reshape back to (b, h, w) - coordinates now match rfft_shape
        resampled = einops.rearrange(
            resampled, "b (h w) -> b h w", h=rfft_shape[0], w=rfft_shape[1]
        )
        # Reshape to original stack_shape
        projection_image_dfts = resampled.reshape(*stack_shape, *rfft_shape)
    else:
        # Single image case
        resampled = sample_image_2d(
            image=projection_image_dfts[None, ...],  # Add batch dim -> (1, h, w)
            coordinates=transformed_coords_flat[None, ...],  # Add batch dim
            interpolation="bilinear"
        )[0]  # Remove batch dim -> (h*w,)
        projection_image_dfts = einops.rearrange(
            resampled, "(h w) -> h w", h=rfft_shape[0], w=rfft_shape[1]
        )

    # Apply the 1/|det A| scaling to preserve total intensity
    det_A = torch.linalg.det(transform_matrix)
    projection_image_dfts = projection_image_dfts / det_A

    return projection_image_dfts


def transform_slice_2d_multichannel(
    projection_image_dfts: torch.Tensor,
    rfft_shape: tuple[int, int],
    stack_shape: tuple[int, ...],
    channels: int,
    transform_matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply anisotropic magnification transformation to a multichannel 2D slice.

    Parameters
    ----------
    projection_image_dfts: torch.Tensor
        Projection image DFTs with shape `(*stack_shape, c, h, w)` where `(h, w)`
        is the rfft shape and `c` is the number of channels.
    rfft_shape: tuple[int, int]
        Shape of the rfft `(h, w_rfft)` where `w_rfft = n//2 + 1`.
    stack_shape: tuple[int, ...]
        Shape of batch dimensions (can be empty for single image).
    channels: int
        Number of channels.
    transform_matrix: torch.Tensor
        `(2, 2)` anisotropic magnification matrix in real space (yx ordering).

    Returns
    -------
    transformed_dfts: torch.Tensor
        Transformed projection image DFTs with same shape as input.
    """
    # Calculate full 2D image shape from rfft shape
    h_full = rfft_shape[0]
    w_full = 2 * (rfft_shape[1] - 1)
    image_shape_2d = (h_full, w_full)

    # Generate 2D frequency grid
    freq_grid = fftfreq_grid(
        image_shape=image_shape_2d,
        rfft=True,
        fftshift=True,
        device=projection_image_dfts.device,
    )  # (h, w, 2) yx coords

    # Apply transformation matrix
    transformed_freqs = transform_fftfreq_grid(
        frequency_grid=freq_grid,
        real_space_matrix=transform_matrix,
        device=projection_image_dfts.device,
    )  # (h, w, 2) yx coords with A^{-T} applied

    # Convert transformed frequencies to DFT array coordinates
    transformed_coords = _fftfreq_to_dft_coordinates(
        frequencies=transformed_freqs,
        image_shape=image_shape_2d,
        rfft=True
    )

    # Flatten coordinates: (h, w, 2) -> (h*w, 2)
    transformed_coords_flat = einops.rearrange(
        transformed_coords, "h w yx -> (h w) yx"
    )

    # Resample each 2D DFT at the transformed frequencies
    # Handle batch and channel dimensions
    if len(stack_shape) > 0:
        # Flatten batch dimensions but keep channels separate
        projection_image_dfts_flat = einops.rearrange(
            projection_image_dfts, "... c h w -> (...) c h w"
        )

        # Process each channel separately
        batch_size = projection_image_dfts_flat.shape[0]
        transformed_coords_expanded = einops.repeat(
            transformed_coords_flat, "n yx -> (b c) n yx",
            b=batch_size, c=channels
        )

        # Flatten channels into batch for sampling
        projection_image_dfts_for_sampling = einops.rearrange(
            projection_image_dfts_flat, "b c h w -> (b c) h w"
        )

        # Resample all images
        resampled = sample_image_2d(
            image=projection_image_dfts_for_sampling,
            coordinates=transformed_coords_expanded,
            interpolation="bilinear"
        )  # (b*c, h*w)

        # Reshape back to (b, c, h, w) then to original stack shape
        resampled = einops.rearrange(
            resampled, "(b c) (h w) -> b c h w",
            b=batch_size, c=channels, h=rfft_shape[0], w=rfft_shape[1]
        )
        projection_image_dfts = resampled.reshape(
            *stack_shape, channels, *rfft_shape
        )
    else:
        # Single image case with channels
        transformed_coords_expanded = einops.repeat(
            transformed_coords_flat, "n yx -> c n yx", c=channels
        )
        resampled = sample_image_2d(
            image=projection_image_dfts,
            coordinates=transformed_coords_expanded,
            interpolation="bilinear"
        )  # (c, h*w)
        projection_image_dfts = einops.rearrange(
            resampled, "c (h w) -> c h w", h=rfft_shape[0], w=rfft_shape[1]
        )

    # Apply the 1/|det A| scaling to preserve total intensity
    det_A = torch.linalg.det(transform_matrix)
    projection_image_dfts = projection_image_dfts / det_A

    return projection_image_dfts
