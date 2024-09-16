from torch_fourier_shell_correlation import fsc



def test_central_slice():

    import os
    import mrcfile
    import requests
    import tempfile
    import torch
    from scipy.spatial.transform import Rotation as R
    from torch_fourier_slice import project_3d_to_2d, backproject_2d_to_3d

    tmpdir= tempfile.gettempdir()
    fname = os.path.join(tmpdir, "emd_17129.map.gz")
    if not os.path.isfile(fname):
        response = requests.get("https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-17129/map/emd_17129.map.gz", stream=True)
        if response.status_code == 200:
            # Open a file in write-binary mode

            with open(fname, "wb") as f:
                # Write the content of the response to the file in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download completed successfully!")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            raise RuntimeError()
    volume = torch.as_tensor(mrcfile.read(fname), dtype=torch.float32)
    
    eulersDegs =  [[0,0,0],[0,0,90], [0,90,0], [0,45,45]]
    rot_mats =  torch.as_tensor(R.from_euler("ZYZ", eulersDegs, degrees=True
                                            ).as_matrix(), dtype=torch.float32)
    projs = project_3d_to_2d(
                    volume=volume,
                    rotation_matrices=rot_mats,
                    pad=False,
                    fftfreq_max=None)
    
    
    from torch_fourier_slice.grids.fftfreq_grid import fftfreq_grid
    from torch_fourier_slice.dft_utils import fftfreq_to_dft_coordinates

    freq_grid = fftfreq_grid(image_shape = volume.shape, rfft = False, fftshift = True, spacing= 1, norm = False, device = "cpu")


    rotation_matrices = torch.flip(rot_mats, dims=(-2, -1))

    rotated_coords = torch.einsum("b q p, ... p -> b  ... q", rotation_matrices, freq_grid)
    _rotated_coords = fftfreq_to_dft_coordinates(frequencies=rotated_coords, image_shape=volume.shape, rfft=False)
    
    from torch_image_lerp import sample_image_3d
    rot_vols = sample_image_3d(image=volume, coordinates=_rotated_coords)

        
    projs_sum = rot_vols.sum(1)
       
    diff = torch.abs(projs - projs_sum)
    from matplotlib import pyplot as plt
    
    
    for i in range(projs.shape[0]):
         _fsc = fsc(projs[i,...], projs_sum[i,...])
#         print(_fsc)
         print(diff[i,...].mean(-1).mean(-1))
#        assert torch.isclose(projs[i], projs_sum[i], atol=1e-1).all(), f"Error, disagreement in projections {i}"
#        breakpoint()
         plt.plot(_fsc, label="euler degs %s"%(eulersDegs[i]))
    plt.legend()
    plt.show()
        

    f, axes = plt.subplots(3,len(diff))
    for i in range(len(diff)):
        axes[0,i].imshow(projs[i])
        axes[1,i].imshow(diff[i])
        axes[2,i].imshow(projs_sum[i])
    plt.show()


if __name__ == "__main__":
    test_central_slice()
    """
PYTHONPATH=../torch-image-lerp/src:src/:$PYTHONPATH python tests/test_torch_fourier_slice.py
    """
