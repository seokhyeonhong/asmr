import os
from os.path import join as pjoin
import numpy as np
import torch
from diff3f import get_features_per_vertex
from utils import convert_mesh_container_to_torch_mesh
from dataloaders.mesh_container import MeshContainer
from diffusion import init_pipe
from dino import init_dino

"""
Data structure:

- DATA_ROOT
    - SMPL
        - smpl_0000.obj
        - smpl_0001.obj
        - ...
    - Mixamo
        - mixamo_0000.obj
        - mixamo_0001.obj
        - ...
    - etc.

For each .obj file, we will save a .npz file with the following
- vert: (V, 3) vertices
- face: (F, 3) faces
- feat: (V, 2048) features, where 2048 = 1280 + 768, 1280 from diffusion unet, 768 from DINO
"""

DATA_ROOT = "./ToSH_CharVtx"
SAVE_ROOT = "./ToSJ_diff3f_new"

if __name__ == "__main__":
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    num_views = 100
    H = 512
    W = 512
    num_images_per_prompt = 1
    tolerance = 0.004
    random_seed = 42
    use_normal_map = True

    def compute_features(device, pipe, dino_model, m, prompt):
        mesh = convert_mesh_container_to_torch_mesh(m, device=device, is_tosca=False)
        mesh_vertices = mesh.verts_list()[0]
        features = get_features_per_vertex(
            device=device,
            pipe=pipe, 
            dino_model=dino_model,
            mesh=mesh,
            prompt=prompt,
            mesh_vertices=mesh_vertices,
            num_views=num_views,
            H=H,
            W=W,
            tolerance=tolerance,
            num_images_per_prompt=num_images_per_prompt,
            use_normal_map=use_normal_map,
        )
        return features.cpu()

    pipe = init_pipe(device)
    dino_model = init_dino(device)

    for char_name in sorted(os.listdir(DATA_ROOT)): # SMPL, Mixamo, etc.
        tgt_dir = pjoin(SAVE_ROOT, char_name)
        os.makedirs(tgt_dir, exist_ok=True)
        if os.path.exists(pjoin(tgt_dir, "diff3f.npy")):
            print(f"{char_name} has already been processed")
            continue
        print(f"Processing {char_name}...")

        try:
            npz = np.load(pjoin(DATA_ROOT, char_name, "mesh", "tpose_mesh_notscaled.npz"))
            vert = npz["c_rest_pos"]
            face = npz["fid_to_cids"]
            mesh = MeshContainer(vert, face)
    
            feat = compute_features(device, pipe, dino_model, mesh, prompt="a humanoid character")
    
            np.save(pjoin(tgt_dir, "diff3f.npy"), feat.cpu().numpy())
            
            print(f"Saved {char_name} features to {tgt_dir}")
        except Exception as e:
            print(e)
            print(f"Exception at {char_name}")