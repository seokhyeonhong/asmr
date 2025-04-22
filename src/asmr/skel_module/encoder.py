import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.norm import BatchNorm, LayerNorm

from same.gat_conv import GATConv
from same.skel_pose_graph import rep_dim

def _get_normalization(norm, dim):
    if norm == "batch":
        return BatchNorm(dim)
    elif norm == "layer":
        return LayerNorm(dim)
    elif norm in [None, "none"]:
        return nn.Identity()
    else:
        raise NotImplementedError(f"Invalid norm type: {norm}")

class SkeletonEncoder(torch.nn.Module):
    def __init__(self, rep_cfg, model_cfg):
        super(SkeletonEncoder, self).__init__()

        enc_cfg = model_cfg["SkeletonEncoder"]
        skel_dim = sum([rep_dim[k] for k in rep_cfg["skel"]])

        hid_lyrs = [enc_cfg["hid_dim"]] * enc_cfg["num_lyrs"]
        heads_num = enc_cfg["heads_num"]

        e_Fs = [skel_dim] + hid_lyrs + [model_cfg["z_dim"]]
        self.convs = []
        self.norms = []
        for i, (fi_prev, fi) in enumerate(zip(e_Fs[:-1], e_Fs[1:])):
            if i != 0:
                fi_prev *= heads_num
            if i != len(e_Fs) - 2:
                heads = heads_num
            else:
                heads = 1
            self.convs.append(GATConv(fi_prev, fi, heads=heads, add_self_loops=True, fill_value=0))
            self.norms.append(_get_normalization(model_cfg["norm"], fi * heads))

        self.convs = nn.ModuleList(self.convs)
        self.norms = nn.ModuleList(self.norms)
        self.dropout = nn.Dropout(p=model_cfg["dropout"])

    def forward(self, src_graph):
        # normalize
        lo, go = src_graph.lo, src_graph.go
        skel_x = torch.hstack([lo, go])

        lo_mean, lo_std = src_graph.saus_ms_from_key("lo")
        go_mean, go_std = src_graph.saus_ms_from_key("go")
        skel_mean = torch.hstack((lo_mean, go_mean))
        skel_std = torch.hstack((lo_std, go_std))

        x = (skel_x - skel_mean) / skel_std

        edge_index_bi = src_graph.edge_index_bidirection

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index_bi)
            x = norm(x)
            if (i + 1) != len(self.convs):
                x = F.relu(x)
            x = self.dropout(x)
        
        return x