import torch
import torch.nn as nn
import torch.nn.functional as F

class SkinningPredictor(nn.Module):
    def __init__(self, z_dim):
        super(SkinningPredictor, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=z_dim, num_heads=1, batch_first=True)

    def forward(self, skel_feats, mesh_feats, attn_mask=None):
        """
        Args:
            - skel_feats: torch.Tensor, shape=(B, maxJ, D)
            - mesh_feats: torch.Tensor, shape=(B, maxV, D)
            - attn_mask: torch.Tensor, shape=(B, V, J), True for valid, False for invalid
            - dist_mask: torch.Tensor, shape=(B, V, J), float tensor between vertices and joints with clamped values (0, 1)
        """
        _, attn = self.attn(mesh_feats, skel_feats, skel_feats, attn_mask=attn_mask, need_weights=True)
        return attn