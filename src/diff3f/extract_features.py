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

DATA_ROOT = "/data/shong/nrf/"

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

    for char_type in sorted(os.listdir(DATA_ROOT)): # SMPL, Mixamo, etc.
        if char_type != "ASMR-Revision":
            continue
        for char_name in sorted(os.listdir(pjoin(DATA_ROOT, char_type))):
            if not char_name.endswith(".obj"):
                continue
            if char_name.replace(".obj", ".npz") in os.listdir(pjoin(DATA_ROOT, char_type)):
                print(f"Skipping {char_name}")
                continue

            mesh = MeshContainer().load_from_file(pjoin(DATA_ROOT, char_type, char_name))
            feat = compute_features(device, pipe, dino_model, mesh, prompt="humanoid character")

            np.savez(pjoin(DATA_ROOT, char_type, char_name.replace(".obj", ".npz")),
                     vert=mesh.vert,
                     face=mesh.face,
                     feat=feat)
            print(f"Saved {char_name.replace('.obj', '.npz')}")
            print(f"Vertices: {mesh.vert.shape}, Faces: {mesh.face.shape}, Features: {feat.shape}")