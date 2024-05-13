import torch

from torch_fourier_slice.interpolation import insert_into_image_2d, sample_image_2d


def test_single_2d_insertion():
    # insert a value of 2.2 at [3.25, 7.25]
    data = torch.tensor([2.2, ])
    coordinates = torch.tensor([3.25, 7.25]).reshape((1, 2))
    image = torch.zeros(size=(10, 10))
    weights = torch.zeros(size=(10, 10))
    image, weights = insert_into_image_2d(data=data, coordinates=coordinates, image=image, weights=weights)

    # should sum to 2.2 in 4 pixels around point at which 2.2 was inserted
    tolerance = 1e-6
    assert image.sum() - 2.2 < tolerance
    assert image[3:5, 7:9].sum() - 2.2 < tolerance
    assert weights[3:5, 7:9].sum() == 1

    # check specific values
    assert image[3, 7] - (0.75 * 0.75 * 2.2) < tolerance
    assert image[3, 8] - (0.75 * 0.25 * 2.2) < tolerance
    assert image[4, 7] - (0.25 * 0.75 * 2.2) < tolerance
    assert image[4, 8] - (0.25 * 0.25 * 2.2) < tolerance


def test_multiple_2d_insertion():
    # insert a value of 1 at both [3.25, 7.25] and [1.75, 3.75]
    data = torch.ones(size=(2,))
    coordinates = torch.tensor(
        [[3.25, 7.25],
         [1.75, 3.75]]
    ).reshape((2, 2))
    image = torch.zeros(size=(10, 10))
    weights = torch.zeros(size=(10, 10))
    image, weights = insert_into_image_2d(data=data, coordinates=coordinates, image=image, weights=weights)

    # should sum to 1 in 4 pixels around points at which 1 was inserted
    assert image.sum() == 2
    assert image[3:5, 7:9].sum() == 1
    assert image[1:3, 3:5].sum() == 1

    # check specific values
    assert image[3, 7] == 0.75 * 0.75
    assert image[3, 8] == 0.75 * 0.25
    assert image[4, 7] == 0.25 * 0.75
    assert image[4, 8] == 0.25 * 0.25

    # check specific values
    assert image[1, 3] == 0.25 * 0.25
    assert image[1, 4] == 0.25 * 0.75
    assert image[2, 3] == 0.75 * 0.25
    assert image[2, 4] == 0.75 * 0.75

    # weights should be identical to image
    assert torch.allclose(weights, image)
