import torch

from torch_fourier_slice.interpolation import insert_into_image_3d, sample_image_3d


def test_single_3d_insertion():
    # insert a value of 2.2 at [3.25, 7.25, 5.5]
    data = torch.tensor([2.2, ])
    coordinates = torch.tensor([3.25, 7.25, 5.5]).reshape((1, 3))
    image = torch.zeros(size=(10, 10, 10))
    weights = torch.zeros(size=(10, 10, 10))
    image, weights = insert_into_image_3d(data=data, coordinates=coordinates, image=image, weights=weights)

    # should sum to 2.2 in 7 voxels around point at which 2.2 was inserted
    tolerance = 1e-6
    assert image.sum() - 2.2 < tolerance
    assert image[3:5, 7:9, 5:7].sum() - 2.2 < tolerance
    assert weights[3:5, 7:9, 5:7].sum() == 1

    # check specific values
    assert image[3, 7, 5] - (0.75 * 0.75 * 0.5 * 2.2) < tolerance
    assert image[3, 8, 5] - (0.75 * 0.25 * 0.5 * 2.2) < tolerance
    assert image[3, 7, 6] - (0.75 * 0.75 * 0.5 * 2.2) < tolerance
    assert image[3, 8, 6] - (0.75 * 0.25 * 0.5 * 2.2) < tolerance
    assert image[4, 7, 5] - (0.25 * 0.75 * 0.5 * 2.2) < tolerance
    assert image[4, 8, 5] - (0.25 * 0.25 * 0.5 * 2.2) < tolerance
    assert image[4, 7, 6] - (0.25 * 0.75 * 0.5 * 2.2) < tolerance
    assert image[4, 8, 6] - (0.25 * 0.25 * 0.5 * 2.2) < tolerance


def test_multiple_3d_insertion():
    # insert a value of 1 at both [3.25, 7.25, 1.5] and [1.75, 3.75, 8.5]
    data = torch.ones(size=(2,))
    coordinates = torch.tensor(
        [[3.25, 7.25, 1.5],
         [1.75, 3.75, 8.5]]
    ).reshape((2, 3))
    image = torch.zeros(size=(10, 10, 10))
    weights = torch.zeros(size=(10, 10, 10))
    image, weights = insert_into_image_3d(data=data, coordinates=coordinates, image=image, weights=weights)

    # should sum to 1 in 8 voxels around points at which 1 was inserted
    assert image.sum() == 2
    assert image[3:5, 7:9, 1:3].sum() == 1
    assert image[1:3, 3:5, 8:10].sum() == 1

    # check specific values
    assert image[3, 7, 1] == 0.75 * 0.75 * 0.5
    assert image[3, 8, 1] == 0.75 * 0.25 * 0.5
    assert image[3, 7, 2] == 0.75 * 0.75 * 0.5
    assert image[3, 8, 2] == 0.75 * 0.25 * 0.5
    assert image[4, 7, 1] == 0.25 * 0.75 * 0.5
    assert image[4, 8, 1] == 0.25 * 0.25 * 0.5
    assert image[4, 7, 2] == 0.25 * 0.75 * 0.5
    assert image[4, 8, 2] == 0.25 * 0.25 * 0.5

    # check specific values
    assert image[1, 3, 8] == 0.25 * 0.25 * 0.5
    assert image[1, 3, 9] == 0.25 * 0.25 * 0.5
    assert image[1, 4, 8] == 0.25 * 0.75 * 0.5
    assert image[1, 4, 9] == 0.25 * 0.75 * 0.5
    assert image[2, 3, 8] == 0.75 * 0.25 * 0.5
    assert image[2, 3, 9] == 0.75 * 0.25 * 0.5
    assert image[2, 4, 8] == 0.75 * 0.75 * 0.5
    assert image[2, 4, 9] == 0.75 * 0.75 * 0.5

    # weights should be identical to image
    assert torch.allclose(weights, image)