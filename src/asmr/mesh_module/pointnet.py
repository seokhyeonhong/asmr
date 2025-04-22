import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool

def _get_activation(act):
    if act == "relu":
        return nn.ReLU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "sigmoid":
        return nn.Sigmoid()
    else:
        return nn.ReLU()

class PointNet(nn.Module):
    def __init__(self, in_feats, num_lyrs, hid_dim, out_dim, act="relu"):
        super(PointNet, self).__init__()
        self.in_feats = in_feats
        self.num_lyrs = num_lyrs
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Linear(in_feats, hid_dim),
            _get_activation(act)
        ))
        for i in range(num_lyrs - 2):
            self.layers.append(nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                _get_activation(act)
            ))
        self.layers.append(nn.Linear(hid_dim, hid_dim))
        self.out = nn.Linear(hid_dim * 2, out_dim)


    def forward(self, x, batch_id):
        x = x.view(-1, self.in_feats)
        for i in range(self.num_lyrs):
            x = self.layers[i](x)

        local_feats = x
        global_feats = global_max_pool(x, batch_id)
        x = torch.cat([local_feats, global_feats[batch_id]], dim=1)
        x = self.out(x)
        return x