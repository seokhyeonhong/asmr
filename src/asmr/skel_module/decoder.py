import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.norm import BatchNorm, LayerNorm
from utils.tensor_utils import unpad_tensor

def _get_normalization(norm, dim):
    if norm == "batch":
        return BatchNorm(dim)
    elif norm == "layer":
        return LayerNorm(dim)
    elif norm in [None, "none"]:
        return nn.Identity()
    else:
        raise NotImplementedError(f"Invalid norm type: {norm}")

class SkeletonDecoder(torch.nn.Module):
    def __init__(self, model_cfg):
        super(SkeletonDecoder, self).__init__()

        dec_cfg = model_cfg["SkeletonDecoder"]
        z_dim = model_cfg["z_dim"]
        heads_num = dec_cfg["heads_num"]
        
        self.heads_num = heads_num

        assert z_dim % heads_num == 0, f"z_dim must be divisible by heads_num, got z_dim={z_dim} and heads_num={heads_num}"

        self.num_layers = dec_cfg["num_lyrs"]
        
        self.attn = nn.MultiheadAttention(embed_dim=z_dim, num_heads=heads_num, batch_first=True)
        self.out_mlp = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, 3)
        )
        self.norm = _get_normalization(model_cfg["norm"], 3)
        self.dropout = nn.Dropout(p=model_cfg["dropout"])

    def forward(self, skel_feats, mesh_feats, njoints, attn_mask=None):
        """
        Args:
            - skel_feats: torch.Tensor, shape=(B, maxJ, D)
            - mesh_feats: torch.Tensor, shape=(B, maxV, D)
            - njoints: torch.Tensor, shape=(B,)
            - skel_edge_index_bidirection: torch.Tensor, shape=(2, sumE)
            - attn_mask: torch.Tensor, shape=(B, J, V), True for valid, False for invalid
        """

        # attention
        attn_out = self.attn(skel_feats, mesh_feats, mesh_feats, attn_mask=attn_mask)[0]
        attn_out = self.dropout(attn_out)
        skel_feats = skel_feats + attn_out
        
        # predicted skeleton scale
        skel_feats = unpad_tensor(skel_feats, njoints)
        skel_feats = self.out_mlp(skel_feats)
        skel_feats = self.norm(skel_feats)
#         skel_scale = F.elu(skel_feats) + 1
        
        return skel_feats