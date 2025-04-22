import torch
import torch.nn as nn

from torch_geometric.nn import MLP, global_max_pool

from utils.tensor_utils import pad_tensor, unpad_tensor
# from saus.mlp import MLP
from asmr.mesh_module.pointnet import PointNet
from asmr.mesh_module.attention import VertexAttention
from asmr.mesh_module.my_dfn import DiffusionNet

mesh_rep_dim = {
    "v": 3, # vertex position
    "n": 3,
    "diff3f": 2560,
    "diff3f_pca": 32,
}

class MeshEncoder(torch.nn.Module):
    def __init__(self, rep_cfg, model_cfg):
        super(MeshEncoder, self).__init__()
        self.use_diff3f = ("diff3f" in rep_cfg["mesh"]) or ("diff3f_pca" in rep_cfg["mesh"])

        enc_cfg = model_cfg["MeshEncoder"]
        mesh_dim = sum([mesh_rep_dim[k] for k in rep_cfg["mesh"]])

        # layers
        self.enc_type = enc_cfg["type"]
        if self.enc_type == "DiffusionNet":
            self.module = DiffusionNet(
                C_in=mesh_dim,
                C_out=model_cfg["z_dim"],
                C_width=enc_cfg["hid_dim"],
                N_block=enc_cfg["num_lyrs"],
                outputs_at=enc_cfg["outputs_at"],
                dropout=True
            )
        elif self.enc_type in ["MLP", "PointNet"]:
            self.module = MLP(
                in_channels=mesh_dim,
                hidden_channels=enc_cfg["hid_dim"],
                out_channels=model_cfg["z_dim"],
                num_layers=enc_cfg["num_lyrs"],
                act="relu",
                norm=model_cfg["norm"],
                dropout=model_cfg["dropout"]
            )
            if self.enc_type == "PointNet":
                self.out = nn.Linear(in_features=model_cfg["z_dim"] * 2, out_features=model_cfg["z_dim"])
        else:
            raise NotImplementedError(f"Invalid encoder type: {self.enc_type}, type should be in [dfn, mlp]")

    def forward(self, meshgraph):
        """
        Args:
            meshgraph (MeshGrpah)
        """
        # reshape
        vtx = meshgraph.tpose_v.detach()
        vtx_mean, vtx_std = meshgraph.ms_from_key("vtx")
        vtx = (vtx - vtx_mean) / vtx_std

        if self.use_diff3f:
            diff3f = meshgraph.diff3f.detach()
            vtx = torch.cat([vtx, diff3f], dim=-1)

        # forward
        if self.enc_type == 'DiffusionNet':
            mass, L, evals, evecs, grad_X, grad_Y = meshgraph.dfnp
            res = []
            dim_evals = evals.shape[0] // meshgraph.batch_size
            for i in range(meshgraph.batch_size):
                idx = (meshgraph.batch == i)
                res.append(self.module(vtx[idx],
                                       mass[idx],
                                       L.to_dense()[idx].to_sparse(),
                                       evals[dim_evals * i:dim_evals * (i + 1)],
                                       evecs[idx],
                                       grad_X.to_dense()[idx].to_sparse(),
                                       grad_Y.to_dense()[idx].to_sparse()))
            
            out = torch.cat(res, dim=0)

        else: # MLP or PointNet
            out = self.module.forward(vtx)
            if self.enc_type == "PointNet":
                pooled_out = global_max_pool(out, meshgraph.batch)
                out = torch.cat([out, pooled_out[meshgraph.batch]], dim=-1)
                out = self.out(out)
        
        return out
    
    def update_precomputes(self, dfnp):
        self.module.update_precomputes(dfnp)
