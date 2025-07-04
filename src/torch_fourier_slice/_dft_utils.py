from collections.abc import Sequence

import torch


def _rfft_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def _fftshift_1d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-1))
    else:
        output = input
    return output


def _fftshift_2d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-2,))
    return output


def _ifftshift_2d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-2,))
    return output


def _fftshift_3d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-3, -2))
    return output


def _ifftshift_3d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-3, -2))
    return output


def _fftfreq_to_dft_coordinates(
    frequencies: torch.Tensor, image_shape: tuple[int, ...], rfft: bool
) -> torch.Tensor:
    """Convert DFT sample frequencies into array coordinates in a fftshifted DFT.

    Specifically:
    - `delta_fftfreq` is the step size in frequency space between adjacent DFT samples.
    - normalized DFT sample frequencies span `[-0.5, 0.5 - delta_fftfreq]`
    - we specify continuous coordinates in the DFT array coordinate system `[0, n - 1]`
      for subsequent sampling

    This functions converts from normalized DFT sample frequencies to DFT array
    coordinates, accounting for arbitrary dimensionaly and whether the DFT is the
    non-redundant half transform from `rfft`.


    Parameters
    ----------
    frequencies: torch.Tensor
        `(..., d)` array of multidimensional DFT sample frequencies
    image_shape: tuple[int, ...]
        Length `d` array of image dimensions.
    rfft: bool
        Whether output should be compatible with an rfft (`True`) or a
        full DFT (`False`)

    Returns
    -------
    coordinates: torch.Tensor
        `(..., d)` array of coordinates into a fftshifted DFT.
    """
    # grab relevant dimension lengths (number of samples per dimension)
    image_shape = torch.as_tensor(
        image_shape, device=frequencies.device, dtype=frequencies.dtype
    )
    rfft_shape = torch.as_tensor(
        _rfft_shape(image_shape), device=frequencies.device, dtype=frequencies.dtype
    )

    # define step size in each dimension
    delta_fftfreq = 1 / image_shape

    # calculate total width of DFT interval in cycles/sample per dimension
    # last dim is only non-redundant half in rfft case
    fftfreq_interval_width = 1 - delta_fftfreq
    if rfft is True:
        fftfreq_interval_width[-1] = 0.5

    # allocate for continuous output dft sample coordinates
    coordinates = torch.empty_like(frequencies)

    # transform frequency coordinates into array coordinates
    if rfft is True:
        # full dimensions span `[-0.5, 0.5 - delta_fftfreq]`
        coordinates[..., :-1] = (frequencies[..., :-1] + 0.5) / fftfreq_interval_width[:-1]
        coordinates[..., :-1] = coordinates[..., :-1] * (image_shape[:-1] - 1)

        # half transform dimension (interval width 0.5)
        coordinates[..., -1] = (frequencies[..., -1] * 2) * (rfft_shape[-1] - 1)
    else:
        # all dims are full and span `[-0.5, 0.5 - delta_fftfreq]`
        coordinates[..., :] = (frequencies[..., :] + 0.5) / fftfreq_interval_width
        coordinates[..., :] = coordinates[..., :] * (image_shape - 1)
    return coordinates


def _dft_center(
    image_shape: tuple[int, ...],
    rfft: bool,
    fftshifted: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Return the position of the DFT center for a given input shape."""
    fft_center = torch.zeros(size=(len(image_shape),), device=device)
    image_shape = torch.as_tensor(image_shape).float()
    if rfft is True:
        image_shape = torch.tensor(_rfft_shape(image_shape), device=device)
    if fftshifted is True:
        fft_center = torch.divide(image_shape, 2, rounding_mode="floor")
    if rfft is True:
        fft_center[-1] = 0
    return fft_center.long()
