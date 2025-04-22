import torch, os
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import scatter

from same.model import FK, parse_hatD

from asmr.skel_module.encoder import SkeletonEncoder
from asmr.skel_module.decoder import SkeletonDecoder
from asmr.mesh_module.encoder import MeshEncoder
from asmr.skinning import SkinningPredictor
from mypath import RESULT_DIR

from utils.file_io import load_model_cfg
from utils.tensor_utils import pad_tensor, unpad_tensor
from utils import tensor_utils

class Model(torch.nn.Module):
    def __init__(self, model_cfg, rep_cfg):
        super(Model, self).__init__()
        self.model_cfg = model_cfg

        z_dim = model_cfg["z_dim"]
        skel_enc_cfg = model_cfg["SkeletonEncoder"]
        skel_dec_cfg = model_cfg["SkeletonDecoder"]
        mesh_enc_cfg = model_cfg["MeshEncoder"]

        self.skel_sym = model_cfg.get("skel_sym", False)
        self.pred_offset_scale = model_cfg.get("pred_offset_scale", True) # if false, predict offsets directly

        # encoders for mesh and skeleton
        self.mesh_enc = MeshEncoder(rep_cfg, model_cfg)
        self.src_skel_enc = SkeletonEncoder(rep_cfg, model_cfg)
        self.tgt_skel_enc = SkeletonEncoder(rep_cfg, model_cfg)

        # skeleton prediction module
        self.tgt_skel_dec = SkeletonDecoder(model_cfg)

        # skinning weights prediction module
        self.skinning_predictor = SkinningPredictor(z_dim)
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def load_params(self, dir, epoch=None, prefix="", freeze=False):
        load_model_name = (
            "last_model.pt" if epoch is None else f"model_{epoch:03d}.pt"
        )
        load_path = os.path.join(RESULT_DIR, dir, load_model_name)

        saved = torch.load(load_path, map_location=self.device)
        model_dict = self.state_dict()

        pretrained_dict = {
            k: v for k, v in saved["model"].items() if k.startswith(prefix)
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        # for key in model_dict.keys():
        #     if key not in dict(model.named_parameters()).keys():
        #         print(key)
        if freeze:
            for name, param in self.named_parameters():
                if name in pretrained_dict:
                    param.requires_grad = False
        print("load model from : ", load_path, " DONE")
        print("prefix : ", prefix, ", freeze:", freeze, "\n")
    
    # def _pre_process_src_skel(self, src_skel_pose, tgt_mesh):
    #     # normalize global offsets and local offsets of hip
    #     v_max = scatter(tgt_mesh.tpose_v, tgt_mesh.batch, dim=0, reduce='max')
    #     v_min = scatter(tgt_mesh.tpose_v, tgt_mesh.batch, dim=0, reduce='min')
    #     j_max = scatter(src_skel_pose.go, src_skel_pose.batch, dim=0, reduce='max')
    #     j_min = scatter(src_skel_pose.go, src_skel_pose.batch, dim=0, reduce='min')
    #     scale = (v_max - v_min) / (j_max - j_min)
    #     src_skel_pose.lo = src_skel_pose.lo * scale[src_skel_pose.batch]
    #     src_skel_pose.go = src_skel_pose.go * scale[src_skel_pose.batch]

    #     return src_skel_pose

    def _post_process_pred_skel(self, batch_size, src_pose, pred_lo):
        # # symmetric
        # x_mirror = torch.tensor([-1, 1, 1], dtype=pred_lo.dtype, device=pred_lo.device)
        # pred_lo = 0.5 * (pred_lo + pred_lo[src_pose.sym_idx] * x_mirror)

        # predicted root height
        pred_r = torch.zeros((batch_size, 4), dtype=src_pose.lo.dtype, device=self.device) # [B, 4]
        root_idx = src_pose.ptr[:-1]
        pred_r[:, -1] = pred_lo[root_idx, 1]

        # forward kinematics for T-pose
        qI = torch.eye(3, dtype=src_pose.lo.dtype, device=self.device)[None].repeat(src_pose.lo.size(0), 1, 1)  # [J, 3, 3]
        FK_Tpose = FK(
            lo = pred_lo, 
            qR = qI,
            r = pred_r,
            root_ids = src_pose.ptr[:-1],
            skel_depth = src_pose.skel_depth,
            skel_edge_index = src_pose.edge_index,
        )
        pred_go = FK_Tpose[:, :3, 3]                        # [sumJ, 3]

        # translate global offsets and local offset of hip to touch the ground
        pred_go_split = torch.split(pred_go, src_pose.njoints.tolist(), dim=0)  # [B] list with [Ji, 3] each
        min_y = [torch.min(go[:, 1]) for go in pred_go_split]                   # [B]

        pred_go = torch.cat([
            go - torch.tensor([0, min_y[i], 0], dtype=go.dtype, device=go.device)
            for i, go in enumerate(pred_go_split)
        ], dim=0)
        pred_lo[root_idx, 1] = pred_lo[root_idx, 1] - torch.stack(min_y, dim=0)

        return pred_lo, pred_go

    def forward(self, tgt_mesh, src_pose, fix_tgt_skel=False):
        """
        Args:
            tgt_mesh (MeshGraph): input character mesh [1, V, D]
                - tpoes_v       [sumV, 3]
                - posed_v       [sumV, 3]
                - diff3f        [sumV, 2048]
                - f, nverts, mass, L, evals, evecs, grad_X, grad_Y
                - batch_size    [B]
                - faces         [F, 3]
            src_pose (SkelPoseGraph): input character pose [sumJ, D=32]
                - lo            [sumJ, 3]
                - go            [sumJ, 3]
                - edge_index    [2, sumJ]
                - edge_feature  [sumJ, 2]
                - qb            [sumJ]
                - njoints       [sumJ]
                - q             [sumJ, 6]
                - p             [sumJ, 3]
                - qv            [sumJ, 6]
                - pv            [sumJ, 3]
                - pprev         [sumJ, 3]
                - c             [sumJ, 1]
                - r_nopad       [sumJ, 4]
                - batch         [sumJ]
                - ptr           [B+1]
        Return:
            pred_skinning (torch.tensor): predicted skinning weights [B, V, J]
            pred_skel     (torch.tensor): predicted local offsets    [sumJ, 6] 
        """
        # 1. encode tgt mesh (T-pose)
        mesh_enc = self.mesh_enc.forward(tgt_mesh)                  # [sumV, D]
        mesh_enc = pad_tensor(mesh_enc, tgt_mesh.nverts)            # [B, V, D]
        B, V, D = mesh_enc.size()
        
        # 2. encode src skeleton (T-pose)
        src_skel_enc = self.src_skel_enc.forward(src_pose)          # [sumJ, D]
        src_skel_enc = pad_tensor(src_skel_enc, src_pose.njoints)   # [B, J, D]
        J = src_skel_enc.size(1)

        # 3. decode target skeleton's local offsets
        H = self.tgt_skel_dec.heads_num
        attn_mask = torch.ones(B * H, J, V, dtype=torch.bool, device=self.device) # True for invalid
        for i in range(src_pose.batch_size):
            attn_mask[i*H:(i+1)*H, :, :tgt_mesh.nverts[i]] = False # not mask out joints due to the possible nan values
        tgt_skel_out = self.tgt_skel_dec.forward(src_skel_enc, mesh_enc, src_pose.njoints, attn_mask=attn_mask)
        # if self.skel_sym: # symmetric scaling with straight-through estimation
        #     lo_scale = (0.5 * (lo_scale[src_pose.sym_idx] - lo_scale)).detach() + lo_scale
        if self.pred_offset_scale:
            tgt_skel_out = F.elu(tgt_skel_out)
            if self.skel_sym:
                tgt_skel_out = 0.5 * (tgt_skel_out[src_pose.sym_idx] + tgt_skel_out)
            pred_lo = tgt_skel_out * src_pose.lo
        else:
            if self.skel_sym:
                x_mirror = torch.tensor([-1, 1, 1], dtype=tgt_skel_out.dtype, device=tgt_skel_out.device)
                tgt_skel_out = 0.5 * (tgt_skel_out + tgt_skel_out[src_pose.sym_idx] * x_mirror)
            pred_lo = tgt_skel_out + src_pose.lo

        # 4. post-processing for ground contact
        pred_lo, pred_go = self._post_process_pred_skel(B, src_pose, pred_lo)
        pred_skel_pose = src_pose.clone().detach()
        pred_skel_pose.lo = pred_lo
        pred_skel_pose.go = pred_go

        if fix_tgt_skel:
            pred_skel_pose = src_pose.clone().detach()

        # 5. encode target skeleton (T-pose)
        tgt_skel_enc = self.tgt_skel_enc.forward(pred_skel_pose)          # [sumJ, D]
        tgt_skel_enc = pad_tensor(tgt_skel_enc, pred_skel_pose.njoints)   # [B, J, D]

        # 6. attention block for skinning weights (True for invalid)
        attn_mask = torch.ones(B, V, J, dtype=torch.bool, device=self.device)
        for i in range(src_pose.batch_size):
            attn_mask[i, :, :src_pose.njoints[i]] = False # not mask out vertices due to the possible nan values
        pred_skinning = self.skinning_predictor.forward(tgt_skel_enc, mesh_enc, attn_mask=attn_mask)

        return pred_skinning, pred_skel_pose
    
def attn(x, mem, additive_mask=None, mask=None, nnz=None):
    """
    Args:
        x (torch.tensor): output feature from mesh encoder [B, T1, D]
        mem (torch.tensor): output feature from skeleton encoder [B, T2, D]
        additive_mask (torch.tensor): additive mask for skeleton encoder output, with float values [T1, T2]
        mask (torch.tensor): mask for skeleton encoder output, True for masking out the value [T1, T2]
        nnz (int): number of non-zero values in the attention matrix
    Return:
        out (torch.tensor): output feature [1, V, J]
    """
    attn_out = torch.matmul(x, mem.transpose(-1, -2))  # [B, T1, T2]
    attn_out = attn_out / (x.size(-1) ** 0.5)

    # mask
    if additive_mask is not None:
        attn_out = attn_out + additive_mask
    if mask is not None:
        attn_out = attn_out.masked_fill(~mask, float('-inf'))

    # (opt.) retain top-k values
    if nnz is not None:
        nnz_idx = torch.topk(attn_out, nnz, dim=-1).indices
        mask = torch.zeros_like(attn_out, dtype=torch.bool).scatter_(-1, nnz_idx, True)
        attn_out = attn_out.masked_fill(~mask, float('-inf'))

    # softmax
    attn_out = torch.softmax(attn_out, dim=-1)

    return attn_out


def make_load_model(model_epoch, device="cuda"):
    if (len(model_epoch.split("/")) == 2) and (model_epoch.split("/")[-1].isdigit()):
        load_epoch = model_epoch.split("/")[-1]
        load_model = model_epoch[: model_epoch.find(load_epoch) - 1]
        load_epoch = int(load_epoch)
    else:
        load_epoch = None
        load_model = model_epoch
        print(model_epoch)
    config = load_model_cfg(load_model)

    model = Model(config["model"], config["representation"]).to(device)
    model.load_params(load_model, load_epoch)

    return model, config

######################## model fwd result post-processing functions ########################

def out_post_fwd(out, src_skel_pose, pred_skel_pose, tgt_mesh, ms_dict, out_rep_cfg, tgt_skel_pose=None):
    """
    Args:
        out = {
            "pred_skinning": pred_skinning,     # [sumV, sumJ]
            "pred_pose": pred_pose              # [sumJ, 11]
        }
        pred_skel_pose (SkelPoseGraphBatch)
            - lo [sumJ, 3]
            - go [sumJ, 3]
        tgt_mesh (MeshGraphBatch)
        
        # ! should match the keys() of out and gt to saus.yml['loss']
    """
    
    # ------- OUT ------- #
    # initialize
    device = tgt_mesh.tpose_v.device
    sumJ = pred_skel_pose.lo.shape[0]
    skin_weights = out["pred_skinning"]
            
    # defompose hatD -> q, r, c
    tgt_root_ids = pred_skel_pose.ptr[:-1]
    out.update(parse_hatD(out["pred_pose"], tgt_root_ids, out_rep_cfg, ms_dict))    # {'r_n', 'r', 'q_n', 'q', 'c'} added

    # q (6d representation) -> qR (rotation matrix)
    out["qR"] = tensor_utils.tensor_q2qR(out["q"])  # [sumJ, 3, 3]

    # qR for identity 
    qR_I = torch.eye(3, dtype=tgt_mesh.tpose_v.dtype, device=device)[None].repeat(sumJ, 1, 1)  # [sumJ, 3, 3] 
    
    # get global joint transformation of identity 
    root_I = torch.zeros((len(pred_skel_pose.ptr) - 1, 4), dtype=pred_skel_pose.lo.dtype, device=device)
    root_I[:, -1] = pred_skel_pose.lo[pred_skel_pose.ptr[:-1], 1]
    FK_I = FK(
        lo = pred_skel_pose.lo,
        qR = qR_I,
        r = root_I,
        root_ids = pred_skel_pose.ptr[:-1],
        skel_depth = pred_skel_pose.skel_depth,
        skel_edge_index = pred_skel_pose.edge_index,
    )
    
    # get globl joint transformation for target rotation 
    FK_T = FK(
        lo = pred_skel_pose.lo,
        qR = out["qR"],
        r = out["r"],
        root_ids = pred_skel_pose.ptr[:-1],
        skel_depth = pred_skel_pose.skel_depth,
        skel_edge_index = pred_skel_pose.edge_index,
    )

    # apply skinning on tgt_mesh     
    pred_v = lbs(
        tpose_v = tgt_mesh.tpose_v,     # [sumV, 3] 
        skin_weights = skin_weights,    # [sumV, sumJ]
        FK_I = FK_I,                    # [sumJ, 4, 4]
        FK_T = FK_T,                    # [sumJ, 4, 4]
        njoints = pred_skel_pose.njoints,
        nverts = tgt_mesh.nverts,
    )
    
    out["vtx"] = pred_v     # [sumV, 3]
    out["fk"] = FK_T        # [sumJ, 4, 4]
    out["fk_tpose"] = FK_I  # [sumJ, 4, 4]
    out["pred_skel_pose"] = pred_skel_pose

    # ------- GT ------- #
    FK_T_gt = FK(
        lo = src_skel_pose.lo,
        qR = tensor_utils.tensor_q2qR(src_skel_pose.q),
        r = src_skel_pose.r_nopad,
        root_ids = src_skel_pose.ptr[:-1],
        skel_depth = src_skel_pose.skel_depth,
        skel_edge_index = src_skel_pose.edge_index,
    )
    gt_root_I = torch.zeros((len(src_skel_pose.ptr) - 1, 4), dtype=src_skel_pose.lo.dtype, device=device)
    gt_root_I[:, -1] = src_skel_pose.lo[src_skel_pose.ptr[:-1], 1]
    gt = {
        "vtx": tgt_mesh.posed_v,        # [sumV, 3]
        # "vtx_tpose": tgt_mesh.tpose_v,  # [sumV, 3]
        # "f": tgt_mesh.f,                # [F, 3]
        "fk": FK_T_gt,                  # [sumJ, 4, 4]
        "fk_tpose": FK(
            lo = src_skel_pose.lo,
            qR = qR_I,
            r = gt_root_I,
            root_ids = src_skel_pose.ptr[:-1],
            skel_depth = src_skel_pose.skel_depth,
            skel_edge_index = src_skel_pose.edge_index,
        ),

        # auxiliary info for loss and metric computation
        "tgt_mesh": tgt_mesh,                # MeshGraphBatch
        "src_skel_pose": src_skel_pose,      # SkelPoseGraphBatch
    }

    if tgt_skel_pose is not None:
        sumJ = tgt_skel_pose.lo.shape[0]
        qR_I = torch.eye(3, dtype=tgt_skel_pose.lo.dtype, device=device)[None].repeat(sumJ, 1, 1)
        gt_root_I = torch.zeros((len(tgt_skel_pose.ptr) - 1, 4), dtype=tgt_skel_pose.lo.dtype, device=device)
        gt_root_I[:, -1] = tgt_skel_pose.lo[tgt_skel_pose.ptr[:-1], 1]

        gt["fk_tgt"] = FK(
            lo = tgt_skel_pose.lo,
            qR = tensor_utils.tensor_q2qR(tgt_skel_pose.q),
            r = tgt_skel_pose.r_nopad,
            root_ids = tgt_skel_pose.ptr[:-1],
            skel_depth = tgt_skel_pose.skel_depth,
            skel_edge_index = tgt_skel_pose.edge_index,
        )
        gt["fk_tgt_tpose"] = FK(
            lo = tgt_skel_pose.lo,
            qR = qR_I,
            r = gt_root_I,
            root_ids = tgt_skel_pose.ptr[:-1],
            skel_depth = tgt_skel_pose.skel_depth,
            skel_edge_index = tgt_skel_pose.edge_index,
        )
        gt["tgt_skel_pose"] = tgt_skel_pose
    
    return out, gt


def lbs(tpose_v, skin_weights, FK_I, FK_T, njoints, nverts):
    '''
    Apply Linear Blending Skinning to tpose vertices
    
    Args:
        tpose_v (torch.tensor): vertex positions in tpose [sumV, 3]
        skin_weights (torch.tensor): skinning weights [B, V, J]
        
        FK_I (torch.tensor): global joint transformation for identity [sumJ, 4, 4]
        FK_T (torch.tensor): global joint transformation for tgt rotation [sumJ, 4, 4]
        
    Returns:
        deformed_v (torch.tensor): deformed vertices [sumV, 3]
        
    # ! pre_xforms not considered
    # ! root recovery not considered
    '''
    
    # initilize
    B, V, J = skin_weights.size()
    device = tpose_v.device
    tpose_v = pad_tensor(tpose_v, nverts)  # [B, V, 3]
    
    # transformation for rest -> tgt  
    rest_to_tgt_trf = torch.matmul(FK_T, FK_I.inverse())                    # [sumJ, 4, 4]
    rest_to_tgt_trf = pad_tensor(rest_to_tgt_trf.reshape(-1, 16), njoints)  # [B, J, 16]
    rest_to_tgt_trf = rest_to_tgt_trf.reshape(B, J, 4, 4)                   # [B, J, 4, 4]

    # convert tpose_v to homogeneous coordinates
    ones = torch.ones((B, V, 1), dtype=tpose_v.dtype, device=device)    # [B, V, 1]
    tpose_v_h = torch.cat([tpose_v, ones], dim=-1)                      # [B, V, 4]

    # apply joint transformation on vertices
    v_trf = skin_weights[..., None, None] * rest_to_tgt_trf[:, None]        # [B, V, J, 4, 4]
    v_trf = torch.sum(v_trf, dim=2)                                         # [B, V, 4, 4]
    deformed_v = torch.matmul(v_trf, tpose_v_h.unsqueeze(-1)).squeeze(-1)   # [B, V, 4]

    # flatten
    deformed_v = unpad_tensor(deformed_v[..., :3], nverts)  # [sumV, 3]

    return deformed_v