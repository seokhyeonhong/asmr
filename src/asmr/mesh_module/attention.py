import torch
import torch.nn as nn

class VertexAttention(nn.Module):
    def __init__(self, in_feats, emb_dim, z_dim, num_lyrs, heads_num, ffn_dim):
        super(VertexAttention, self).__init__()
        assert emb_dim % heads_num == 0, f"emb_dim should be divisible by heads_num, got {emb_dim} and {heads_num}"

        self.in_feats = in_feats
        self.emb_dim = emb_dim
        self.num_lyrs = num_lyrs
        self.heads_num = heads_num

        self.in_mlp = nn.Linear(in_feats, emb_dim)

        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_lyrs):
            self.attn_layers.append(nn.MultiheadAttention(emb_dim, heads_num, batch_first=True))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(emb_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, emb_dim)
            ))
        
        self.out_mlp = nn.Linear(emb_dim, z_dim)

    def forward(self, x, attn_mask=None):
        x = self.in_mlp(x)
        
        for i in range(self.num_lyrs):
            attn, _ = self.attn_layers[i](x, x, x, attn_mask=attn_mask)
            x = x + attn

            ffn = self.ffn_layers[i](x)
            x = x + ffn
        
        x = self.out_mlp(x)

        return x