# temporary stub

def test_something():
    pass



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

    rot_mats =  torch.as_tensor(R.from_euler("ZYZ", [[0,0,0],[0,0,90], [0,90,0]], degrees=True
                                            ).as_matrix(), dtype=torch.float32)
    projs = project_3d_to_2d(
                    volume=volume,
                    rotation_matrices=rot_mats,
                    pad=False,
                    fftfreq_max=None)
    

    affine_mats = torch.zeros(rot_mats.shape[0], 3, 4)
    affine_mats[:,:3,:3] = rot_mats
#    affine_mats[:,:3,-1] += 1./volume.shape[-1] #TODO: It seems that the projections may be off by half a pixel


    volume = volume[None,None,...].expand(rot_mats.shape[0], -1, -1, -1, -1)
    rot_vols = torch.nn.functional.grid_sample(volume, torch.nn.functional.affine_grid(affine_mats, size=volume.shape), align_corners=False)
    projs_sum = rot_vols.sum(2).squeeze(1)
    
    for i in range(projs.shape[0]):
        assert torch.isclose(projs[i], projs_sum[i], atol=1e-1).all(), f"Error, disagreement in projections {i}"
        
    diff = torch.abs(projs - projs_sum)
    print(diff.mean(-1).mean(-1))

#    from matplotlib import pyplot as plt
#    f, axes = plt.subplots(3,3)
#    axes[0,0].imshow(projs[0])
#    axes[0,1].imshow(projs[1])
#    axes[0,2].imshow(projs[2])
#    
#    axes[1,0].imshow(diff[0])
#    axes[1,1].imshow(diff[1])
#    axes[1,2].imshow(diff[2])
#    
#    axes[2,0].imshow(projs_sum[0])
#    axes[2,1].imshow(projs_sum[1])
#    axes[2,2].imshow(projs_sum[2])
#    plt.show()

"""
 PYTHONPATH=../torch-image-lerp/src:$PYTHONPATH  python -m src.torch_fourier_slice.project
"""
if __name__ == "__main__":
    test_central_slice()
    """
PYTHONPATH=../torch-image-lerp/src:src/:$PYTHONPATH python tests/test_torch_fourier_slice.py
    """
