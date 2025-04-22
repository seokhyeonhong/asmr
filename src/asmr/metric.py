import torch

def chamfer_distance(x, y):
    """
    Compute Chamfer distance between two point clouds.
    Args:
        x: [V, 3] tensor
        y: [V, 3] tensor
    Returns:
        chamfer_distance: scalar tensor
    """
    dist = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1) # [V, V]
    chamfer_dist = torch.min(dist, dim=1).values.mean() + torch.min(dist, dim=0).values.mean()
    return chamfer_dist


def point2line_distance(x, y0, y1):
    """
    Compute point-to-line distance between a point cloud and a line segment.
    Args:
        x: [X, 3] tensor
        y0: [Y, 3] tensor
        y1: [Y, 3] tensor
    Returns:
        point2line_distance: [X, Y] tensor
    """
    line_vec = y1 - y0 # [Y, 3]
    x2y0_vec = x.unsqueeze(1) - y0.unsqueeze(0) # [X, Y, 3]

    t = torch.sum(x2y0_vec * line_vec.unsqueeze(0), dim=-1) / (torch.sum(line_vec**2, dim=-1) + 1e-8) # [X, Y]
    t = torch.clamp(t, 0, 1)

    closest_pnt = y0.unsqueeze(0) + t.unsqueeze(-1) * line_vec.unsqueeze(0) # [X, Y, 3]
    dist = torch.norm(x.unsqueeze(1) - closest_pnt, dim=-1) # [X, Y]

    return dist


def line2line_distance(x0, x1, y0, y1):
    """
    Compute line-to-line distance between two line segments.
    Args:
        x0: [X, 3] tensor
        x1: [X, 3] tensor
        y0: [Y, 3] tensor
        y1: [Y, 3] tensor
    Returns:
        line2line_distance: [X, Y] tensor
    """
    dx = x1 - x0 # [X, 3]
    dy = y1 - y0 # [Y, 3]

    r = x0.unsqueeze(1) - y0.unsqueeze(0) # [X, Y, 3]

    a = torch.sum(dx**2, dim=-1).unsqueeze(1) # [X, 1]
    b = torch.sum(dx.unsqueeze(1) * dy.unsqueeze(0), dim=-1) # [X, Y]
    c = torch.sum(dy**2, dim=-1).unsqueeze(0) # [1, Y]
    d = torch.sum(dx.unsqueeze(1) * r, dim=-1) # [X, Y]
    e = torch.sum(dy.unsqueeze(0) * r, dim=-1) # [X, Y]

    demon = a * c - b**2

    mask = demon.abs() < 1e-6
    demon[mask] = 1e-6

    t1 = (b * e - c * d) / demon
    t2 = (a * e - b * d) / demon

    t1 = torch.clamp(t1, 0, 1)
    t2 = torch.clamp(t2, 0, 1)

    p1 = x0.unsqueeze(1) + t1.unsqueeze(-1) * dx.unsqueeze(1) # [X, Y, 3]
    p2 = y0.unsqueeze(0) + t2.unsqueeze(-1) * dy.unsqueeze(0) # [X, Y, 3]

    dist = torch.norm(p1 - p2, dim=-1) # [X, Y]

    return dist

def compute_chamfer_distance(out, gt):
    """
    Compute Chamfer distance between two point clouds.
    Returns:
        chamfer_distance: [B] tensor
    """
    pred_v = out["vtx"]
    gt_mesh = gt["tgt_mesh"]

    # metric for each batch
    results = []
    for i in range(gt_mesh.batch.max() + 1):
        idx = (gt_mesh.batch == i)
        x = pred_v[idx]
        y = gt_mesh.posed_v[idx]
        results.append(chamfer_distance(x, y))
        
    return torch.stack(results, dim=0)


def compute_pointwise_mesh_distance(out, gt):
    """
    Compute point-wise mesh euclidean distance (PMD) between two mesh vertices.
    Returns:
        pmd: [B] tensor
    """
    pred_v = out["vtx"]
    gt_mesh = gt["tgt_mesh"]

    results = []
    for i in range(gt_mesh.batch.max() + 1):
        idx = (gt_mesh.batch == i)
        x = pred_v[idx]
        y = gt_mesh.posed_v[idx]
        dist = torch.norm(x - y, dim=-1, p=2) # [V]
        results.append(dist.mean())
    return torch.stack(results, dim=0)


def compute_max_mesh_distance(out, gt):
    """
    Compute point-wise mesh euclidean distance (PMD) between two mesh vertices.
    Returns:
        pmd: [B] tensor
    """
    pred_v = out["vtx"]
    gt_mesh = gt["tgt_mesh"]

    results = []
    for i in range(gt_mesh.batch.max() + 1):
        idx = (gt_mesh.batch == i)
        x = pred_v[idx]
        y = gt_mesh.posed_v[idx]
        dist = torch.norm(x - y, dim=-1, p=2) # [V]
        results.append(dist.max())
    return torch.stack(results, dim=0)



def compute_edge_length_score(out, gt):
    """
    Compute edge length score (ELS) between two meshes.
    Returns:
        els: [B] tensor
    """
    # batch_idx = gt["batch_idx"]
    # v_out, v_gt = out["vtx"], gt["vtx"]
    # edge_idx = gt["edge_index"]
    # num_edges = gt["num_edges"]
    pred_v = out["vtx"]
    gt_mesh = gt["tgt_mesh"]
    
    results = []
    num_edges = torch.tensor(gt_mesh.nedges, device=pred_v.device)
    cumsum_nedges = torch.cumsum(num_edges, dim=0) - num_edges
    for i in range(gt_mesh.batch.max() + 1):
        # idx = edge_idx[:, cum_num_edges[i]:cum_num_edges[i+1]] # [2, E]
        idx = gt_mesh.edge_index[:, cumsum_nedges[i]:cumsum_nedges[i] + gt_mesh.nedges[i]]
        x0, x1 = pred_v[idx[0]], pred_v[idx[1]]
        y0, y1 = gt_mesh.posed_v[idx[0]], gt_mesh.posed_v[idx[1]]
        dist_x = torch.norm(x0 - x1, dim=-1, p=2)
        dist_y = torch.norm(y0 - y1, dim=-1, p=2) + 1e-8
        els = torch.sum(1 - torch.abs(dist_x / dist_y - 1)) / gt_mesh.nedges[i]
        results.append(els)

    return torch.stack(results, dim=0)


def compute_chamfer_joint2joint(out, gt):
    """
    Compute Chamfer distance between two sets of joint positions.
    Returns:
        chamfer_j2j: [B] tensor
    """
    pred_skel_pose = out["pred_skel_pose"]
    gt_skel_pose = gt["tgt_skel_pose"]
    
    pred_go = pred_skel_pose.go # [sumJ1, 3]
    gt_go = gt_skel_pose.go # [sumJ2, 3]

    res = []
    for i in range(pred_skel_pose.batch.max() + 1):
        pred_idx = (pred_skel_pose.batch == i)
        gt_idx = (gt_skel_pose.batch == i)
        pred = pred_go[pred_idx]
        gt = gt_go[gt_idx]
        res.append(chamfer_distance(pred, gt))

    return torch.stack(res, dim=0)


def compute_chamfer_joint2bone(out, gt):
    """
    Compute Chamfer distance between joint positions and bone positions.
    Returns:
        chamfer_j2b: [B] tensor
    """
    pred_skel_pose = out["pred_skel_pose"]
    gt_skel_pose = gt["tgt_skel_pose"]
    
    pred_go = pred_skel_pose.go # [sumJ1, 3]
    gt_go = gt_skel_pose.go # [sumJ2, 3]
    pred_edge_idx = pred_skel_pose.edge_index
    gt_edge_idx = gt_skel_pose.edge_index

    res = []
    for i in range(pred_skel_pose.batch.max() + 1):
        pred_go_i = pred_go[pred_skel_pose.batch == i]
        gt_go_i = gt_go[gt_skel_pose.batch == i]

        pred_bone_i = pred_edge_idx[:, pred_skel_pose.batch[pred_edge_idx[0]] == i] # [2, E_pred]
        gt_bone_i = gt_edge_idx[:, gt_skel_pose.batch[gt_edge_idx[0]] == i] # [2, E_gt]

        dist1 = point2line_distance(pred_go_i, gt_go[gt_bone_i[0]], gt_go[gt_bone_i[1]]) # [J_pred, E_gt]
        dist2 = point2line_distance(gt_go_i, pred_go[pred_bone_i[0]], pred_go[pred_bone_i[1]]) # [J_gt, E_pred]

        res.append(dist1.min(dim=1).values.mean() + dist2.min(dim=1).values.mean())

    return torch.stack(res, dim=0)


def compute_chamfer_bone2bone(out, gt):
    """
    Compute Chamfer distance between bone positions.
    Returns:
        chamfer_b2b: [B] tensor
    """
    pred_skel_pose = out["pred_skel_pose"]
    gt_skel_pose = gt["tgt_skel_pose"]
    
    pred_go = pred_skel_pose.go # [sumJ1, 3]
    gt_go = gt_skel_pose.go # [sumJ2, 3]
    pred_edge_idx = pred_skel_pose.edge_index
    gt_edge_idx = gt_skel_pose.edge_index

    res = []
    for i in range(pred_skel_pose.batch.max() + 1):
        pred_bone_i = pred_edge_idx[:, pred_skel_pose.batch[pred_edge_idx[0]] == i] # [2, E_pred]
        gt_bone_i = gt_edge_idx[:, gt_skel_pose.batch[gt_edge_idx[0]] == i] # [2, E_gt]

        dist1 = line2line_distance(pred_go[pred_bone_i[0]], pred_go[pred_bone_i[1]], gt_go[gt_bone_i[0]], gt_go[gt_bone_i[1]]) # [E_pred, E_gt]
        dist2 = line2line_distance(gt_go[gt_bone_i[0]], gt_go[gt_bone_i[1]], pred_go[pred_bone_i[0]], pred_go[pred_bone_i[1]]) # [E_gt, E_pred]

        res.append(dist1.min(dim=1).values.mean() + dist2.min(dim=1).values.mean())

    return torch.stack(res, dim=0)

    
_metric_matching_ = {
    "chamfer": compute_chamfer_distance,
    "pmd": compute_pointwise_mesh_distance,
    "els": compute_edge_length_score,
    "mde": compute_max_mesh_distance,
    "chamfer_j2j": compute_chamfer_joint2joint,
    "chamfer_j2b": compute_chamfer_joint2bone,
    "chamfer_b2b": compute_chamfer_bone2bone,
}

def get_metric_function(ltype):
    return _metric_matching_[ltype]


def get_metric_names():
    return list(_metric_matching_.keys())


def compute_metric(metric_cfg, out, gt):
    metrics = dict()
    for key in metric_cfg:
        ftn = get_metric_function(key)
        metrics[key] = ftn(out, gt)
    return metrics