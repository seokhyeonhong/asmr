import trimesh
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import copy

def get_char_mesh(vtx, faces,
                  add_axes=False, add_plane=False, return_mesh=True):
    
    mesh = [ trimesh.Trimesh(vertices=vtx, faces=faces, process=False) ]

    if add_axes:
        x_axis = _get_axis([0, 0, 0], [100, 0, 0], color=[255, 0, 0])
        y_axis = _get_axis([0, 0, 0], [0, 100, 0], color=[0, 255, 0])
        z_axis = _get_axis([0, 0, 0], [0, 0, 100], color=[0, 0, 255])
        mesh.extend([x_axis, y_axis, z_axis])
    
    if add_plane:
        plane = _get_plane(radius=100, color=[128, 128, 128])
        mesh.append(plane)
    
    return trimesh.util.concatenate(mesh) if return_mesh else mesh

def get_pose_mesh(xform, jnt_color=[128, 128, 128],
                  add_axes=False, add_plane=False, return_mesh=True):
    if xform.shape[-1] == 3:
        eye = np.eye(4).reshape(1, 4, 4)
        eye = eye.repeat(xform.shape[0], axis=0)
        eye[..., :3, -1] = xform
        xform = eye
    assert xform.shape[-2:] == (4, 4)
    
    joint_boxes = []
    for i in range(xform.shape[0]):
        joint_boxes.append(_get_joint_box(xform[i], color=jnt_color))

    if add_axes:
        x_axis = _get_axis([0, 0, 0], [100, 0, 0], color=[255, 0, 0])
        y_axis = _get_axis([0, 0, 0], [0, 100, 0], color=[0, 255, 0])
        z_axis = _get_axis([0, 0, 0], [0, 0, 100], color=[0, 0, 255])
        joint_boxes.extend([x_axis, y_axis, z_axis])
    
    if add_plane:
        plane = _get_plane(radius=100, color=[128, 128, 128])
        joint_boxes.append(plane)

    return trimesh.util.concatenate(joint_boxes) if return_mesh else joint_boxes

def get_skinning_weights_mesh(vtx, faces, skinning_weights,
                              perjoint=True, tpose_xform=None, merge_skel_mesh=True):
    """
    Args:
    - vtx: (V, 3) numpy array of vertex positions
    - faces: (F, 3) numpy array of face indices
    - skinning_weights: (V, J) numpy array of skinning weights
    - perjoint: if True, visualize skinning weights per joint, otherwise argmax joint per each vertex
    - return_mesh: if True, return a trimesh object, otherwise a list of trimesh objects
    """
    nverts, njoint = skinning_weights.shape
    mesh = get_char_mesh(vtx, faces, add_axes=False, add_plane=False, return_mesh=True)
    if perjoint:
        # color map for continuous color
        cmap = plt.get_cmap("viridis")
        
        # min and max color of cmap
        min_color = np.array(cmap.colors[0]) * 255
        max_color = np.array(cmap.colors[-1]) * 255

        # joint mesh
        if tpose_xform is None:
            joint_mesh = [None] * njoint
        else:
            joint_mesh = get_pose_mesh(tpose_xform, jnt_color=min_color, add_axes=False, add_plane=False, return_mesh=False)
            joint_mesh = [copy.deepcopy(joint_mesh) for _ in range(njoint)]

        # colorize vertices
        mesh = [ mesh.copy() for _ in range(njoint) ]
        for j in range(njoint):
            color = cmap(skinning_weights[:, j])[:, :3] * 255
            color = np.concatenate([color, np.ones((nverts, 1)) * 255], axis=1).astype(np.uint8)
            mesh[j].visual.vertex_colors = color
            if tpose_xform is not None:
                # color the corresponding joint with the max color of cmap
                joint_mesh[j][j].visual.face_colors = max_color
        
        # concatenate meshes
        if tpose_xform is not None:
            joint_mesh_ = []
            for j in range(njoint):
                joint_mesh_.append(trimesh.util.concatenate(joint_mesh[j]))
                joint_mesh_[-1].apply_translation([0, 0, -100])
            joint_mesh = joint_mesh_

            # result
            if merge_skel_mesh:
                mesh = [ trimesh.util.concatenate([mesh[j], joint_mesh[j]]) for j in range(njoint) ]
            else:
                mesh = [ (mesh[j], joint_mesh[j]) for j in range(njoint) ]
        
    else:
        # color map
        cmap = plt.get_cmap("hsv")(np.linspace(0, 1, njoint))[:, :3]
        cmap = cmap[np.random.permutation(njoint)]
        cmap = ListedColormap(cmap)
        
        # joint mesh
        if tpose_xform is None:
            joint_mesh = [None] * njoint
        else:
            joint_mesh = get_pose_mesh(tpose_xform, jnt_color=[0, 0, 0], add_axes=False, add_plane=False, return_mesh=False)

        max_joint = np.argmax(skinning_weights, axis=1)
        color = cmap(max_joint / max_joint.max()) * 255
        # color = np.concatenate([color, np.ones((nverts, 1)) * 255], axis=1).astype(np.uint8)
        mesh.visual.vertex_colors = color.astype(np.uint8)

        if tpose_xform is not None:
            # joint-wise color
            for j in range(njoint):
                color = np.array(cmap(np.array(j)))[:3] * 255
                joint_mesh[j].visual.face_colors = np.concatenate([color, [255]], axis=0)
            
            # result
            joint_mesh = trimesh.util.concatenate(joint_mesh)
            joint_mesh.apply_translation([0, 0, -100])
            if merge_skel_mesh:
                mesh = trimesh.util.concatenate([mesh, joint_mesh])
            else:
                mesh = (mesh, joint_mesh)

    return mesh
    

def _get_axis(start, end, radius=1, color=[255, 0, 0]):
    axis_vec = np.array(end) - np.array(start)
    axis_len = np.linalg.norm(axis_vec)

    # create cylinder
    cylinder = trimesh.creation.cylinder(radius=radius, height=axis_len, sections=12)

    # align the cylinder with the axis vector
    cylinder.apply_translation([0, 0, axis_len / 2])
    cylinder.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], axis_vec))
    cylinder.apply_translation(start)
    
    # set the color of the cylinder
    cylinder.visual.face_colors = color
    
    return cylinder

def _get_joint_box(xform, color=[128, 128, 128]):
    joint_box = trimesh.creation.box(extents=[10, 10, 10])
    joint_box.apply_transform(xform)
    joint_box.visual.face_colors = color

    return joint_box

def _get_plane(radius=100, color=[128, 128, 128]):
    plane = trimesh.creation.box(extents=[radius, 0.1, radius])
    plane.apply_translation([0, -0.05, 0])
    plane.visual.face_colors = color

    return plane


# if __name__ == "__main__":
#     vert = np.load("vert.npy")
#     face = np.load("face.npy")
#     skinning_weights = np.load("skinning_weights.npy")
#     tpose = np.load("tpose.npy")
#     mesh = get_skinning_weights_mesh(vert, face, skinning_weights=skinning_weights, tpose_xform=tpose, perjoint=True)