import pytest
import torch
from scipy.stats import special_ortho_group
from torch_fourier_shell_correlation import fsc

from torch_fourier_slice import (
    backproject_2d_to_3d,
    backproject_2d_to_3d_multichannel,
    project_2d_to_1d,
    project_3d_to_2d,
    project_3d_to_2d_multichannel,
)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_project_3d_to_2d_rotation_center(device):
    # rotation center should be at position of DC in DFT
    volume = torch.zeros((32, 32, 32), device=device)
    volume[16, 16, 16] = 1

    # make projections
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=100),
        device=device,
    )
    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
    )

    assert device in str(projections.device)
    # check max is always at (16, 16), implying point (16, 16) never moves
    for image in projections:
        max_idx = torch.argmax(image)
        i, j = divmod(max_idx.item(), 32)
        assert (i, j) == (16, 16)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_project_2d_to_1d_rotation_center(device):
    # rotation center should be at position of DC in DFT
    image = torch.zeros((32, 32), device=device)
    image[16, 16] = 1

    # make projections
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=2, size=100),
        device=device,
    )
    projections = project_2d_to_1d(
        image=image,
        rotation_matrices=rotation_matrices,
    )

    assert device in str(projections.device)
    # check max is always at (16), implying point (16) never moves
    for image in projections:
        i = torch.argmax(image)
        assert i == 16


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_3d_2d_projection_backprojection_cycle(cube, device):
    # make projections
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=1500),
        device=device,
    )
    projections = project_3d_to_2d(
        volume=cube.to(device),
        rotation_matrices=rotation_matrices,
    )

    # reconstruct
    volume = backproject_2d_to_3d(
        images=projections,
        rotation_matrices=rotation_matrices,
    )

    assert device in str(projections.device)
    assert device in str(volume.device)

    # calculate FSC between the ground truth volume and the reconstruction
    _fsc = fsc(cube.to("cpu"), volume.float().to("cpu"))
    assert torch.all(_fsc[-10:] > 0.99)  # few low res shells at 0.98...


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_3d_2d_projection_backprojection_cycle_leading_dims(cube, device):
    # make projections
    size = cube.shape[-1]
    rotation_matrices = torch.rand((4, 5, 3, 3), device=device)
    projections = project_3d_to_2d(
        volume=cube.to(device),
        rotation_matrices=rotation_matrices,
    )

    assert device in str(projections.device)
    assert projections.shape == (4, 5, size, size)

    # reconstruct
    volume = backproject_2d_to_3d(
        images=projections,
        rotation_matrices=rotation_matrices,
    )

    assert device in str(volume.device)
    assert volume.shape == (size,) * 3


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_3d_to_2d_projection_backprojection_cycle_multichannel(device):
    channels, slices, size = 4, 8, 10
    volumes_shape = (channels, size, size, size)
    projections_shape = (slices, channels, size, size)
    # a volume with 4 channels
    volumes = torch.rand(volumes_shape, device=device)  # (c, d, d, d)
    # a rotation matrix for each tilt -> (n, 3, 3)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=slices),
        device=device,
    )

    # run batched back projection
    projections = project_3d_to_2d_multichannel(volumes, rotation_matrices)
    assert projections.shape == projections_shape
    assert device in str(projections.device)

    # run batched back projection
    result = backproject_2d_to_3d_multichannel(projections, rotation_matrices)
    assert result.shape == volumes_shape
    assert device in str(result.device)


@pytest.mark.parametrize(
    "dtype, device",
    ((p0, p1) for p0, p1 in zip([torch.float32, torch.float64], DEVICES)),
)
def test_dtypes_slice_insertion(dtype, device):
    images = torch.rand((10, 28, 28), dtype=dtype, device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=10),
        dtype=dtype,
        device=device,
    )
    result = backproject_2d_to_3d(images, rotation_matrices)
    assert result.dtype == dtype
    assert device in str(result.device)
