"""A simple projection/backprojection cycle implementation."""

import mrcfile
import napari
import torch
from scipy.stats import special_ortho_group

from torch_fourier_slice import backproject_2d_to_3d, project_3d_to_2d

N_IMAGES = 1000
torch.manual_seed(42)

# load a volume and normalise
volume = torch.tensor(mrcfile.read("/Users/burta2/data/4v6x_bin4.mrc"))
volume -= torch.mean(volume)
volume /= torch.std(volume)

# rotation matrices for projection (operate on xyz column vectors)
rotations = torch.tensor(
    special_ortho_group.rvs(dim=3, size=N_IMAGES, random_state=42)
).float()

# make projections
projections = project_3d_to_2d(
    volume,
    rotation_matrices=rotations,
    pad=True,
)  # (b, h, w)

# reconstruct volume from projections
reconstruction = backproject_2d_to_3d(
    images=projections,
    rotation_matrices=rotations,
    pad=True,
)
reconstruction -= torch.mean(reconstruction)
reconstruction = reconstruction / torch.std(reconstruction)

# visualise
viewer = napari.Viewer()
viewer.add_image(projections.numpy(), name="projections")

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(volume.numpy(), name="ground truth")
viewer.add_image(reconstruction.numpy(), name="reconstruction")
napari.run()
