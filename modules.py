import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
    def forward(self, x):
        x_res = x
        x_norm = self.ln_1(x)
        attn_out, _ = self.attn(x_norm.transpose(0, 1),
                                  x_norm.transpose(0, 1),
                                  x_norm.transpose(0, 1))
        attn_out = attn_out.transpose(0, 1)
        x = x_res + attn_out

        x_res = x
        x_norm = self.ln_2(x)
        x = x_res + self.mlp(x_norm)
        return x

class Transformer(nn.Module):
    def __init__(self, layers, width, heads):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads) for _ in range(layers)
        ])
    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x