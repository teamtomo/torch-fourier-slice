import pytest
import torch
from scipy.stats import special_ortho_group
from torch_fourier_shell_correlation import fsc

from torch_fourier_slice import backproject_2d_to_3d, project_2d_to_1d, project_3d_to_2d


def test_project_3d_to_2d_rotation_center():
    # rotation center should be at position of DC in DFT
    volume = torch.zeros((32, 32, 32))
    volume[16, 16, 16] = 1

    # make projections
    rotation_matrices = torch.tensor(special_ortho_group.rvs(dim=3, size=100)).float()
    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
    )

    # check max is always at (16, 16), implying point (16, 16) never moves
    for image in projections:
        max_idx = torch.argmax(image)
        i, j = divmod(max_idx.item(), 32)
        assert (i, j) == (16, 16)


def test_project_2d_to_1d_rotation_center():
    # rotation center should be at position of DC in DFT
    image = torch.zeros((32, 32))
    image[16, 16] = 1

    # make projections
    rotation_matrices = torch.tensor(special_ortho_group.rvs(dim=2, size=100)).float()
    projections = project_2d_to_1d(
        image=image,
        rotation_matrices=rotation_matrices,
    )

    # check max is always at (16), implying point (16) never moves
    for image in projections:
        i = torch.argmax(image)
        assert i == 16


def test_3d_2d_projection_backprojection_cycle(cube):
    # make projections
    rotation_matrices = torch.tensor(special_ortho_group.rvs(dim=3, size=1500)).float()
    projections = project_3d_to_2d(
        volume=cube,
        rotation_matrices=rotation_matrices,
    )

    # reconstruct
    volume = backproject_2d_to_3d(
        images=projections,
        rotation_matrices=rotation_matrices,
    )

    # calculate FSC between the projections and the reconstructions
    _fsc = fsc(cube, volume.float())

    assert torch.all(_fsc[-5:] > 0.99)  # few low res shells at 0.98...


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
)
def test_dtypes_slice_insertion(dtype):
    images = torch.rand((10, 28, 28), dtype=dtype)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=10),
        dtype=dtype,
    )
    result = backproject_2d_to_3d(images, rotation_matrices)
    assert result.dtype == dtype
