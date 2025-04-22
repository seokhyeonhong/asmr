import torch
import torch.nn.functional as F
from functools import partial

from utils.tensor_utils import pad_tensor
from utils.mesh_utils import compute_sdf
from asmr.metric import chamfer_distance

def default_mse_loss(loss_key, out, gt):
    y_gt, y_out = gt[loss_key], out[loss_key]

    if ("mask" in gt) and (loss_key != "r") and (loss_key != "ra"):
        mask = gt["mask"].reshape(-1, y_gt.shape[1])[0]
        y_gt, y_out = y_gt[:, ~mask], y_out[:, ~mask]

    if y_gt.dtype == torch.bool:
        y_gt = y_gt.float()
    if y_out.dtype == torch.bool:
        y_out = y_out.float()
    
    return torch.nn.MSELoss()(y_gt, y_out)


def edge_loss(out, gt):
    pred_vtx = out["vtx"] # [sumV, 3]

    gt_mesh = gt["tgt_mesh"]
    gt_vtx = gt_mesh.posed_v # [sumV, 3]
    gt_edge = gt_mesh.edge_index # [2, sumE]

    pred_edge = pred_vtx[gt_edge[0]] - pred_vtx[gt_edge[1]] # [sumE, 3]
    gt_edge = gt_vtx[gt_edge[0]] - gt_vtx[gt_edge[1]] # [sumE, 3]
    
    return torch.nn.MSELoss()(pred_edge, gt_edge)


def joint_chamfer_loss(out, gt):
    pred_skel_pose, gt_skel_pose = out["pred_skel_pose"], gt["tgt_skel_pose"]
    pred_fk, gt_fk = out["fk_tpose"], gt["fk_tgt_tpose"]

    pred_fk = pred_fk[:, :3, 3] # [sumJ1, 3]
    gt_fk = gt_fk[:, :3, 3]     # [sumJ2, 3]

    # chamfer loss
    batch_size = gt_skel_pose.batch.max() + 1
    loss = 0
    for i in range(batch_size):
        pred_idx = (pred_skel_pose.batch == i)
        gt_idx = (gt_skel_pose.batch == i)

        pred = pred_fk[pred_idx]
        gt = gt_fk[gt_idx]

        loss += chamfer_distance(pred, gt)
    
    return loss / batch_size


def joint_mesh_sdf(out, gt):
    pred_skel_pose, gt_mesh = out["pred_skel_pose"], gt["tgt_mesh"]
    pred_fk = out["fk_tpose"][:, :3, 3] # [sumJ, 3]
    # fk_out = out["fk_tpose"][:, :3, 3] # [sumJ, 3]
    # mesh = gt["vtx_tpose"] # [sumV, 3]
    # mesh_faces = gt["f"] # [sumF, 3]

    # loss
    batch_size = pred_skel_pose.batch.max() + 1
    loss = 0
    cumsum_nfaces = torch.cumsum(gt_mesh.nfaces, dim=0) - gt_mesh.nfaces
    for i in range(batch_size):
        pred_skel = pred_fk[pred_skel_pose.batch == i] # [J, 3]
        verts = gt_mesh.posed_v[gt_mesh.batch == i] # [V, 3]
        faces = gt_mesh.f[cumsum_nfaces[i]:cumsum_nfaces[i] + gt_mesh.nfaces[i]] # [F, 3]

        loss += compute_sdf(verts, faces, pred_skel).mean()

    return loss / batch_size


def skinning_loss_v1(out, gt):
    pred_skinning = out["pred_skinning"]
    loss = -pred_skinning * torch.log(pred_skinning + 1e-8)
    return loss.mean()

def skinning_loss_v2(out, gt):
    pred_skinning = out["pred_skinning"] # [B, V, J]
    # small std
    loss = torch.std(pred_skinning, dim=-1)
    return loss.mean()

_loss_matching_ = {
    "vtx": partial(default_mse_loss, "vtx"), # vertex position
    "edge": edge_loss,
    "joint_chamfer": joint_chamfer_loss,
    "joint_mesh_sdf": joint_mesh_sdf,
    # "joint_mesh_sdf_v2": joint_mesh_sdf_v2,
    "skinning_v1": skinning_loss_v1,
    "skinning_v2": skinning_loss_v2,
}

def get_loss_function(ltype):
    return _loss_matching_[ltype]

def compute_loss(loss_cfg, out, gt):
    losses = dict()
    loss_sum = 0.0
    for key, weight in loss_cfg.items():
        if weight > 1e-8:
            ftn = get_loss_function(key)
            loss = ftn(out, gt)
            weighted_loss = loss * weight
            losses[key] = weighted_loss
            loss_sum += weighted_loss
    losses["total"] = loss_sum
    return losses
