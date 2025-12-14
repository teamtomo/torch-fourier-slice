import einops
import torch
from torch_image_interpolation import sample_image_2d, sample_image_3d
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from .._dft_utils import _fftfreq_to_dft_coordinates
from .._grids import _central_slice_fftfreq_grid


def extract_central_slices_rfft_3d(
    volume_rfft: torch.Tensor,
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    transform_matrix: torch.Tensor | None = None,  # (2, 2)
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

    # --- apply anisotropic magnification in Fourier space ---
    if transform_matrix is not None:
        # Generate 2D frequency grid with transformation applied
        # torch_grid_utils.fftfreq_grid now handles A^{-T} internally
        transformed_freqs = fftfreq_grid(
            image_shape=(rfft_shape[0], rfft_shape[1]),
            rfft=True,
            fftshift=True,
            device=volume_rfft.device,
            transform_matrix=transform_matrix,
        )  # (h, w, 2) yx coords with A^{-T} applied

        # Convert transformed frequencies to DFT array coordinates
        transformed_coords = _fftfreq_to_dft_coordinates(
            frequencies=transformed_freqs,
            image_shape=(rfft_shape[0], rfft_shape[1]),
            rfft=True
        )

        # Resample each 2D DFT at the transformed frequencies
        # Need to handle batch dimensions in stack_shape
        if len(stack_shape) > 0:
            # Flatten batch dimensions for processing
            projection_image_dfts_flat = einops.rearrange(
                projection_image_dfts, "... h w -> (...) h w"
            )

            # Expand coordinates for all images in the stack
            batch_size = projection_image_dfts_flat.shape[0]
            transformed_coords_expanded = einops.repeat(
                transformed_coords, "h w yx -> b (h w) yx", b=batch_size
            )

            # Resample all images
            resampled = sample_image_2d(
                image=projection_image_dfts_flat,
                coordinates=transformed_coords_expanded,
                interpolation="bilinear"
            )  # (b, h*w)

            # Reshape back to (b, h, w) then to original stack shape
            resampled = einops.rearrange(
                resampled, "b (h w) -> b h w", h=rfft_shape[0], w=rfft_shape[1]
            )
            projection_image_dfts = einops.rearrange(
                resampled, "(batch) h w -> batch h w", batch=batch_size
            ).reshape(*stack_shape, *rfft_shape)
        else:
            # Single image case
            transformed_coords_flat = einops.rearrange(
                transformed_coords, "h w yx -> (h w) yx"
            )
            resampled = sample_image_2d(
                image=projection_image_dfts[None, ...],  # Add batch dim
                coordinates=transformed_coords_flat[None, ...],  # Add batch dim
                interpolation="bilinear"
            )[0]  # Remove batch dim
            projection_image_dfts = einops.rearrange(
                resampled, "(h w) -> h w", h=rfft_shape[0], w=rfft_shape[1]
            )

        # Apply the 1/|det A| scaling to preserve total intensity
        det_A = torch.linalg.det(transform_matrix)
        projection_image_dfts = projection_image_dfts / det_A

    return projection_image_dfts


def extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,  # (c, d, d, d)
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    transform_matrix: torch.Tensor | None = None,  # (2, 2)
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

    # --- apply anisotropic magnification in Fourier space ---
    if transform_matrix is not None:
        # Generate 2D frequency grid with transformation applied
        # torch_grid_utils.fftfreq_grid now handles A^{-T} internally
        transformed_freqs = fftfreq_grid(
            image_shape=(rfft_shape[0], rfft_shape[1]),
            rfft=True,
            fftshift=True,
            device=volume_rfft.device,
            transform_matrix=transform_matrix,
        )  # (h, w, 2) yx coords with A^{-T} applied

        # Convert transformed frequencies to DFT array coordinates
        transformed_coords = _fftfreq_to_dft_coordinates(
            frequencies=transformed_freqs,
            image_shape=(rfft_shape[0], rfft_shape[1]),
            rfft=True
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
                transformed_coords, "h w yx -> b (h w) yx", b=batch_size * channels
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
            projection_image_dfts = resampled.reshape(*stack_shape, channels, *rfft_shape)
        else:
            # Single image case with channels
            transformed_coords_expanded = einops.repeat(
                transformed_coords, "h w yx -> c (h w) yx", c=channels
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
