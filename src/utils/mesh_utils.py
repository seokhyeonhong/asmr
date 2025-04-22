# Code borrowed from: https://github.com/chacorp/matplotlib_render/blob/main/matplotlib_render.py
import os
from glob import glob
import trimesh
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as matclrs
from matplotlib.collections import PolyCollection

from scipy.spatial import cKDTree

import torch.nn.functional as F

def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M

def ortho(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 3] = 1.0
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(zfar + znear) / (zfar - znear)
    return M

def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c,-s, 0, 0],
                      [ s, c, 0, 0],
                      [ 0, 0, 1, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ 1, 0, 0, 0],
                      [ 0, c,-s, 0],
                      [ 0, s, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def transform_vertices(frame_v, MVP, F, norm=True, no_parsing=False):
    V = frame_v
    if norm:
        V = (V - (V.max(0) + V.min(0)) *0.5) / max(V.max(0) - V.min(0))
    V = np.c_[V, np.ones(len(V))]
    V = V @ MVP.T
    V /= V[:, 3].reshape(-1, 1)
    if no_parsing:
        return V
    VF = V[F]
    return VF

def calc_face_norm(vertices, faces, mode='faces'):
    """
    Args
        vertices (np.ndarray): vertices
        faces (np.ndarray): face indices
    """

    fv = vertices[faces]
    span = fv[:, 1:, :] - fv[:, :1, :]
    norm = np.cross(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[:, np.newaxis] + 1e-12)
    
    if mode=='faces':
        return norm
    
    # Compute mean vertex normals manually
    vertex_normals = np.zeros(vertices.shape, dtype=np.float64)
    for i, face in enumerate(faces):
        for vertex in face:
            vertex_normals[vertex] += norm[i]

    # Normalize the vertex normals
    norm_v = vertex_normals / (np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis] + 1e-12)
    return norm_v

def colors_to_cmap(colors):
    '''
    colors_to_cmap(nx3_or_nx4_rgba_array) yields a matplotlib colormap object that, when
    that will reproduce the colors in the given array when passed a list of n evenly
    spaced numbers between 0 and 1 (inclusive), where n is the length of the argument.

    Example:
      cmap = colors_to_cmap(colors)
      zs = np.asarray(range(len(colors)), dtype=np.float) / (len(colors)-1)
      # cmap(zs) should reproduce colors; cmap[zs[i]] == colors[i]
    '''
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        colors = np.hstack((colors, np.ones((len(colors),1))))
    steps = (0.5 + np.asarray(range(len(colors)-1), dtype=float))/(len(colors) - 1)
    return matclrs.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: ([(0, col[0], col[0])] + 
                   [(step, c0, c1) for (step,c0,c1) in zip(steps, col[:-1], col[1:])] + 
                   [(1, col[-1], col[-1])])
         for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:,clridx]]},
        N=len(colors)
    )

def get_new_mesh(vertices, faces, v_idx, invert=False):
    """Calculate standardized mesh
    Args:
        vertices (np.ndarray): [V, 3] array of vertices 
        faces (np.ndarray): [F, 3] array of face indices 
        v_idx (np.ndarray): [N] list of vertex index to remove from mesh
    Return:
        updated_verts (np.ndarray): [V', 3] new array of vertices 
        updated_faces (np.ndarray): [F', 3] new array of face indices  
        updated_verts_idx (np.ndarray): [N] list of vertex index to remove from mesh (fixed)
    """
    max_index = vertices.shape[0]
    new_vertex_indices = np.arange(max_index)

    if invert:
        mask = np.zeros(max_index, dtype=bool)
        mask[v_idx] = True
    else:
        mask = np.ones(max_index, dtype=bool)
        mask[v_idx] = False

    updated_verts = vertices[mask]
    updated_verts_idx = new_vertex_indices[mask]

    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(updated_verts_idx)}

    updated_faces = np.array([
                    [index_mapping.get(idx, -1) for idx in face]
                    for face in faces
                ])

    valid_faces = ~np.any(updated_faces == -1, axis=1)
    updated_faces = updated_faces[valid_faces]
    
    return updated_verts, updated_faces, updated_verts_idx

def plot_image_array(Vs, 
                     Fs, 
                     rot_list=None, 
                     size=6, 
                     norm=False, 
                     mode='mesh', 
                     linewidth=1, 
                     linestyle='solid', 
                     light_dir=np.array([0,0,1]),
                     trans=np.array([0,0,-5]),
                     bg_black = True,
                     logdir='.', 
                     name='000', 
                     save=False,
                    ):
    """
    Args:
        Vs (list): list of vertices [V, V, V, ...]
        Fs (list): list of face indices [F, F, F, ...]
        rot_list (list): list of euler angle [ [x,y,z], [x,y,z], ...]
        size (int): size of figure
        norm (bool): if True, normalize vertices
        mode (str): mode for rendering [mesh(wireframe), shade, normal]
        linewidth (float): line width for wireframe (kwargs for matplotlib)
        linestyle (str): line style for wireframe (kwargs for matplotlib)
        light_dir (np.array): light direction
        bg_black (bool): if True, use dark_background for plt.style
        logdir (str): directory for saved image
        name (str): name for saved image
        save (bool): if True, save the plot as image
    """
    if mode=='gouraud':
        print("currently WIP!: need to curl by z")
        
    num_meshes = len(Vs)
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)

        #xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        # model = translate(0, 0, -3) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        model = translate(trans[0], trans[1], trans[2]) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        # proj  = perspective(55, 1, 1, 100)
        proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
        MVP   = proj @ model # view is identity
        
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        if mode=='normal':
            C = calc_face_norm(V, F) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            C = np.clip(C, 0, 1) if False else C * 0.5 + 0.5
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        elif mode=='shade':
            C = calc_face_norm(V, F) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            
            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            C = C*0.5+0.25
            collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        elif mode=='gouraud':
            # I = np.argsort(Z)
            # V, F, vidx = get_new_mesh(V, F, I, invert=True)
            
            ### curling by normal
            C = calc_face_norm(V, F, mode='v') #@ model[:3,:3].T
            NI = np.argwhere(C[:,2] > 0.0).squeeze()
            V, F, vidx = get_new_mesh(V, F, NI, invert=True)
            
            C = calc_face_norm(V, F,mode='v') #@ model[:3,:3].T
            
            #VV = (V-V.min()) / (V.max()-V.min())# world coordinate
            V = transform_vertices(V, MVP, F, norm, no_parsing=True)
            triangle_ = tri.Triangulation(V[:,0], V[:,1], triangles=F)
            
            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            C = C*0.5+0.25
            #VV = (V-V.min()) / (V.max()-V.min()) #screen coordinate
            #cmap = colors_to_cmap(VV)
            cmap = colors_to_cmap(C)
            zs = np.linspace(0.0, 1.0, num=V.shape[0])
            plt.tripcolor(triangle_, zs, cmap=cmap, shading='gouraud')
            
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor='black')
            
        if mode!='gouraud':
            ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
    
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def get_batch_rotation(P, Q):
    """[P->Q] Find the optimal rotation for a batch of normal vectors.
    Args:
        P (np.array): [N, 3] a set of source normals.
        Q (np.array): [N, 3] a set of corresponding target normals.
    Return:
        R (np.array): [N, 3, 3] Rotation matrices (R) for each pair.
    """
    assert P.shape == Q.shape, "P and Q must have the same shape"
    assert P.shape[1] == 3, "Each normal vector should be of size 3"

    # Compute the covariance matrices for all pairs
    H = np.einsum('ij,ik->ijk', P, Q)

    # Perform Singular Value Decomposition in batch
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrices in batch
    R = np.einsum('ijk,ilk->ijl', Vt, U)

    return R

def get_rotation(P, Q):
    """[P->Q] Find the optimal rotation of normal vectors.
    Args:
        P (np.array): [1, 3] a set of source normal.
        Q (np.array): [1, 3] a set of corresponding target normal.
    Return: # Q = P @ R.T
        R (np.array): Rotation matrix (R)
    """
    assert P.shape == Q.shape, "P and Q must have the same shape"
    assert P.shape[1] == 3, "The normal vector should be of size 3"
    
    # Compute the covariance matrix
    H = P.T @ Q

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R = Vt.T @ U.T
    return R

def apply_batch_rotations(vertices, rotations):
    """Apply rotation matrices to a batch of vertices.
    Args:
        vertices (np.array): [N, 3] a set of vertices.
        rotations (np.array): [N, 3, 3] Rotation matrices (R) for each vertex.
    Return:
        rotated_vertices (np.array): [N, 3] Rotated vertices.
    """
    assert vertices.shape[0] == rotations.shape[0], "Vertices and rotations must have the same batch size"
    assert vertices.shape[1] == 3, "Each vertex should be of size 3"
    assert rotations.shape[1] == 3 and rotations.shape[2] == 3, "Each rotation matrix should be of size 3x3"

    # Apply the rotation matrices to the vertices
    rotated_vertices = np.einsum('nij,nj->ni', rotations, vertices)

    return rotated_vertices

def get_mean_scale(vertices, cache_dir=None):
    """
    Normalize vertices to be in range of -1 ~ 1
    Args:
        vertices (torch.tensor)
    Return:
        mean (torch.tensor)
        scale (torch.tensor)
    """
    V_max = vertices.max(0).values
    V_min = vertices.min(0).values
    
    mean_V = (V_max + V_min) * 0.5
    scale_V = 2 / max(V_max - V_min)
    
    # n_vertices = (vertices - mean_V) * scale_V
    
    if cache_dir is not None:
        np.savez(cache_dir, mean=mean_V, scale=scale_V)
        
    return mean_V, scale_V
    
def decimate_mesh(mesh, target_faces):
    """decimate mesh
    Args:
        mesh (trimesh.Trimesh)
    Return
        decimated_mesh (trimesh.Trimesh)
    """
    decimated_mesh = mesh.simplify_quadric_decimation(target_faces)
    return decimated_mesh

def map_vertices(original_mesh, decimated_mesh):
    """
    Get nearest neighbor vertex on original mesh
    Args:
        original_mesh (trimesh.Trimesh)
        decimated_mesh (trimesh.Trimesh)
    Return
        distances (np.ndarray): distance with the nearest neighbor vertex on original mesh
        vertex_map (np.ndarray): index of the nearest neighbor vertex on original mesh
    """
    tree = cKDTree(original_mesh.vertices)
    distances, vertex_map = tree.query(decimated_mesh.vertices, k=1)
    return distances, vertex_map

def apply_decimation_to_animation(animation_frames, vertex_map, decimated_mesh):
    """
    Args:
        animation_frames (list(trimesh.Trimesh))
        vertex_map (np.ndarray)
        decimated_mesh (list(trimesh.Trimesh))
    Return
        decimated_animation_frames (list(trimesh.Trimesh))
    """
    decimated_animation_frames = []
    for frame in animation_frames:
        decimated_vertices = frame.vertices[vertex_map]
        decimated_animation_frames.append(trimesh.Trimesh(vertices=decimated_vertices, faces=decimated_mesh.faces, process=False))
    return decimated_animation_frames

def get_new_mesh(vertices, faces, v_idx, invert=False):
    """calculate standardized mesh
    Args:
        vertices (torch.tensor): [V, 3] array of vertices 
        faces (torch.tensor): [F, 3] array of face indices 
        v_idx (torch.tensor): [N] list of vertex index to remove from mesh
    Return:
        updated_verts (torch.tensor): [V', 3] new array of vertices 
        updated_faces (torch.tensor): [F', 3] new array of face indices  
        updated_verts_idx (torch.tensor): [N] list of vertex index to remove from mesh (fixed)
    """
    max_index = vertices.shape[0]
    new_vertex_indices = torch.arange(max_index)

    if invert:
        mask = torch.zeros(max_index, dtype=torch.bool)
        mask[v_idx] = 1
    else:
        mask = torch.ones(max_index, dtype=torch.bool)
        mask[v_idx] = 0

    updated_verts     = vertices[mask]
    updated_verts_idx = new_vertex_indices[mask]

    index_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(updated_verts_idx)}

    updated_faces = torch.tensor([
                    [index_mapping.get(idx.item(), -1) for idx in face]
                    for face in faces
                ])

    valid_faces = ~torch.any(updated_faces == -1, dim=1)
    updated_faces = updated_faces[valid_faces]
    return updated_verts, updated_faces, updated_verts_idx

def decimate_mesh_vertex(mesh, num_vertex, tolerance=2, verbose=False):
    """
    Decimate the mesh to have approximately the target number of vertices.
    Args:
        mesh (trimesh.Trimesh): Mesh to decimate.
        num_vertex (int): Target vertex number.
    Return:
        mesh (trimesh.Trimesh): Decimated mesh.
    """
    
    #NOTE Euler Characteristic: V - E + F = 2
    num_faces = 100 + 2 * num_vertex
    prev_num_faces = mesh.faces.shape[0]
    
    while abs(mesh.vertices.shape[0] - num_vertex) > tolerance:
        if num_faces == prev_num_faces:
            num_faces = num_faces -1
        mesh = mesh.simplify_quadric_decimation(num_faces)
        if verbose:
            print("Decimated to", num_faces, "faces, mesh has", mesh.vertices.shape[0], "vertices")
        num_faces -= (mesh.vertices.shape[0] - num_vertex) // 2
        prev_num_faces = num_faces
    if verbose:
        print('Output mesh has', mesh.vertices.shape[0], 'vertices and', mesh.faces.shape[0], 'faces')
    return mesh

# pip install gdown potpourri3d trimesh open3d transforms3d libigl robust_laplacian vedo

def compute_sdf(vertices, faces, points):
    """
    Args:
        - vtx: torch.Tensor, shape=(V, 3)
        - face: torch.Tensor, shape=(F, 3)
        - point: torch.Tensor, shape=(P, 3)
    Return:
        - sdf: torch.Tensor, shape=(P,), signed distance from the surface
    """

    # face vertices
    face_verts = vertices[faces] # [F, 3, 3]

    # normals
    v0v1 = face_verts[:, 1] - face_verts[:, 0] # [F, 3]
    v0v2 = face_verts[:, 2] - face_verts[:, 0] # [F, 3]
    normals = torch.cross(v0v1, v0v2, dim=-1) # [F, 3]
    normals = F.normalize(normals, p=2, dim=-1) # [F, 3]

    # point-vertex vectors
    points_expanded = points[:, None, None, :]  # [P, 1, 1, 3]
    face_verts_expanded = face_verts[None]      # [1, F, 3, 3]
    point2vert = face_verts_expanded - points_expanded # [P, F, 3, 3]
    
    # point-face vectors using closest point-vertex for each face
    point2vert_dist = torch.norm(point2vert, p=2, dim=-1) # [P, F, 3]
    min_dist, min_idx = torch.min(point2vert_dist, dim=-1) # [P, F]
    point2face = torch.gather(point2vert, dim=-2, index=min_idx[..., None, None].expand(-1, -1, -1, 3)) # [P, F, 1, 3]
    point2face = point2face.squeeze(-2) # [P, F, 3]

    # sign: positive for outside, negative for inside
    sign = torch.sum(-point2face * normals[None], dim=-1) # [P, F]
    sign = torch.sign(sign)

    # distance
    dist = torch.norm(point2face, p=2, dim=-1) # [P, F]
    min_dist, min_idx = torch.min(dist, dim=-1) # [P]
    sign = torch.gather(sign, dim=-1, index=min_idx[..., None].expand(-1, -1)) # [P]
    sign = sign.squeeze(-1) # [P]
    dist = min_dist # [P]

    return sign * dist


    
# if __name__ == "__main__":
#     mesh = np.load("/data/saus/saus-train-v3/Amy/Mesh/tpose_mesh.npz")
#     vtx = torch.from_numpy(mesh["c_rest_pos"]).float().cuda()
#     face = torch.from_numpy(mesh["fid_to_cids"]).long().cuda()
#     # point = torch.randn(10, 3).cuda()
#     # mesh2 = np.load("/data/saus/saus-test-v3/Ortiz/Mesh/tpose_mesh.npz")
#     # point = torch.from_numpy(mesh2["c_rest_pos"]).float().cuda()
#     point = torch.tensor([[-4, 126, 6.5]]).float().cuda(); sdf = compute_sdf(vtx, face, point)
#     print(sdf)
#     breakpoint()