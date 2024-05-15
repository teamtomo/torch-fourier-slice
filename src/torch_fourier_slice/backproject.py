import torch
import torch.nn.functional as F

from .grids import fftfreq_grid
from .slice_insertion import insert_central_slices_rfft_3d


def backproject_2d_to_3d(
    images: torch.Tensor,  # (b, h, w)
    rotation_matrices: torch.Tensor,  # (b, 3, 3)
    pad: bool = True,
    do_gridding_correction: bool = True,
    fftfreq_max: float | None = None,
):
    """Perform a 3D reconstruction from a set of 2D projection images.

    Parameters
    ----------
    images: torch.Tensor
        `(batch, h, w)` array of 2D projection images.
    rotation_matrices: torch.Tensor
        `(batch, 3, 3)` array of rotation matrices for insert of `images`.
        Rotation matrices left-multiply column vectors containing coordinates.
    pad: bool
        Whether to pad the input images 2x (`True`) or not (`False`).
    do_gridding_correction: bool
        Each 2D image pixel contributes to the nearest eight voxels in 3D and weights are set
        according to a linear interpolation kernel. The effects of this trilinear interpolation in
        Fourier space can be 'undone' through division by a sinc^2 function in real space.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection.

    Returns
    -------
    reconstruction: torch.Tensor
        `(d, h, w)` cubic volume containing the 3D reconstruction from `images`.
    """
    b, h, w = images.shape
    if h != w:
        raise ValueError('images must be square.')
    if pad is True:
        p = images.shape[-1] // 4
        images = F.pad(images, pad=[p] * 4)

    # construct shapes
    b, h, w = images.shape
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
        fftfreq_max=fftfreq_max
    )

    # reweight reconstruction
    valid_weights = weights > 1e-3
    dft[valid_weights] /= weights[valid_weights]

    # back to real space
    dft = torch.fft.ifftshift(dft, dim=(-3, -2,))  # actual ifftshift
    dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))  # center in real space

    # correct for convolution with linear interpolation kernel
    if do_gridding_correction is True:
        grid = fftfreq_grid(
            image_shape=dft.shape,
            rfft=False,
            fftshift=True,
            norm=True,
            device=dft.device
        )
        dft = dft / torch.sinc(grid) ** 2

    # unpad
    if pad is True:
        dft = F.pad(dft, pad=[-p] * 6)
    return torch.real(dft)
