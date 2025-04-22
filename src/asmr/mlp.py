import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim=32, 
                 hid_dim=128, 
                 num_gn=32, 
                 num_layer=4,
                 act='relu'):
        super(MLP, self).__init__()
        
        if act == 'none':
            self.act = nn.Identity()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU()
        
        # Linear layers
        self.linears = [nn.Linear(in_dim, hid_dim)]
        for _ in range(num_layer):
            self.linears.append(nn.Linear(hid_dim, hid_dim))
        self.linears = nn.ModuleList(self.linears)
        
        # GROUP NORMs
        self.gns = [nn.GroupNorm(num_gn, hid_dim) for _ in range(num_layer+1)]
        self.gns = nn.ModuleList(self.gns)
        
        self.linear_out = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        """
            x (torch.tensor) [B, N, C]: input
            out (torch.tensor)
        """
        out = x
        for i in range(len(self.linears)):
            out = self.act(self.linears[i](out))
            # out = self.gns[i](out.transpose(-1, -2)).transpose(-1, -2)
        out = self.linear_out(out)
        return out

    def update_precomputes(self, dfnp):
        # dummy function
        return
