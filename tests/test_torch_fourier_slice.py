import pytest
import torch

from torch_fourier_slice import project_3d_to_2d, backproject_2d_to_3d
from torch_fourier_shell_correlation import fsc
from scipy.stats import special_ortho_group


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
        max = torch.argmax(image)
        i, j = divmod(max.item(), 32)
        assert (i, j) == (16, 16)


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
    "images, rotation_matrices",
    [
        (
            torch.rand((10, 28, 28)).float(),
            torch.tensor(special_ortho_group.rvs(dim=3, size=10)).float()
        ),
    ]
)
def test_dtypes_slice_insertion(images, rotation_matrices):
    result = backproject_2d_to_3d(images, rotation_matrices)
    assert result.dtype == torch.float64
