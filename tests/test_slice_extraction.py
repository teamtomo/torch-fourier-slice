import torch

from torch_fourier_slice.slice_extraction._extract_central_slices_rfft_3d import extract_central_slices_rfft_3d, project_fourier
from torch_fourier_slice.dft_utils import fftshift_3d

from scipy.spatial.transform import Rotation as R


def test_slice_extraction():
    volume = torch.rand((10, 10, 10))
    dft = torch.fft.rfftn(volume, dim=(-3, -2, -1))
    dft = fftshift_3d(dft, rfft=True)
    rotation_matrices = torch.tensor(R.random(num=5, random_state=42).as_matrix()).float()
    results = extract_central_slices_rfft_3d(
        dft,
        image_shape=(10, 10, 10),
        rotation_matrices=rotation_matrices,
        fftfreq_max=0.25
    )
    print(results)


def test_project_volume():
    from datetime import datetime
    volume = torch.rand((128, 128, 128))

    rotation_matrices = torch.tensor(R.random(num=500, random_state=42).as_matrix()).float()
    t0 = datetime.now()
    project_fourier(volume, rotation_matrices=rotation_matrices, fftfreq_max=None)
    t1 = datetime.now()
    print(f'full slice: {t1 - t0}')

    t0 = datetime.now()
    project_fourier(volume, rotation_matrices=rotation_matrices, fftfreq_max=0.50)
    t1 = datetime.now()
    print(f'fftfrex max = 0.50:{t1 - t0}')

    t0 = datetime.now()
    project_fourier(volume, rotation_matrices=rotation_matrices, fftfreq_max=0.25)
    t1 = datetime.now()
    print(f'fftfrex max = 0.25:{t1 - t0}')
