import torch
import numpy as np

import os
import sys
root_dir = os.path.join(os.getcwd().split('VML-SAUS')[0],'VML-SAUS')
sys.path.append(f'{root_dir}/diffusion-net/src')
import diffusion_net

def get_dfn_info(mesh, cache_dir=None, map_location='cuda'):
    verts_list = torch.from_numpy(mesh.v).unsqueeze(0).float()
    face_list = torch.from_numpy(mesh.f).unsqueeze(0).long()
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = \
        diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)

    dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], torch.from_numpy(mesh.f)]
    dfn_info = [_.to(map_location).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    return dfn_info

def get_dfn_info2(vertices, faces, cache_dir=None, map_location='cuda'):
    verts_list = vertices.unsqueeze(0).float()
    face_list = faces.unsqueeze(0).long()
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = \
        diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)

    dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], faces]
    dfn_info = [_.to(map_location).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    return dfn_info