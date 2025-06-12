"""A simple projection/backprojection cycle implementation."""

import matplotlib.pyplot as plt
import mrcfile
import napari
import torch
from scipy.stats import special_ortho_group
from torch_fourier_shell_correlation import fsc

from torch_fourier_slice import backproject_2d_to_3d, project_3d_to_2d

N_IMAGES = 50
torch.manual_seed(42)

# load a volume and normalise
volume = torch.tensor(
    mrcfile.read("/home/marten/data/datasets/emdb/test/emd_48372_10A.mrc")
)
volume -= torch.mean(volume)
volume /= torch.std(volume)

# rotation matrices for projection (operate on xyz column vectors)
rotations = torch.tensor(
    special_ortho_group.rvs(dim=3, size=N_IMAGES, random_state=42)
).float()

fig, ax = plt.subplots()

for x in [1.0, 1.5, 2.0, 3.0, 4.0]:
    # make projections
    projections = project_3d_to_2d(
        volume,
        rotation_matrices=rotations,
        pad_factor=x,
    )  # (b, h, w)

    # reconstruct volume from projections
    reconstruction = backproject_2d_to_3d(
        images=projections,
        rotation_matrices=rotations,
        pad_factor=x,
    )
    reconstruction -= torch.mean(reconstruction)
    reconstruction = reconstruction / torch.std(reconstruction)

    _fsc = fsc(volume, reconstruction)[1:]
    ax.plot(_fsc, label=f"{x}")

ax.legend()
plt.show()

# visualise
viewer = napari.Viewer()
# viewer.add_image(projections.numpy(), name="projections")
viewer.add_image(volume.numpy(), name="ground truth")
viewer.add_image(reconstruction.numpy(), name="reconstruction")
napari.run()
