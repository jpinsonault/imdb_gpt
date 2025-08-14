import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from ..fields import BaseField

def _out_dim(m: nn.Module) -> int:
    for p in reversed(list(m.parameters())):
        if p.dim() == 2:
            return p.size(0)
    return None

class _FieldEncoders(nn.Module):
    def __init__(self, fields: List[BaseField], latent_dim: int):
        super().__init__()
        self.fields = fields
        self.encs = nn.ModuleList([f.build_encoder(latent_dim) for f in fields])
        self.proj = nn.ModuleList([nn.Identity() if _out_dim(m) == latent_dim else nn.Linear(_out_dim(m), latent_dim) for m in self.encs])
        self.fuse = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        outs = []
        for x, enc, proj in zip(xs, self.encs, self.proj):
            y = enc(x)
            if y.dim() > 2:
                y = y.flatten(1)
            y = proj(y)
            outs.append(y)
        tokens = torch.stack(outs, dim=1)
        attn_out, _ = self.fuse(tokens, tokens, tokens)
        z = self.norm(attn_out.mean(dim=1))
        return z

class _FieldDecoders(nn.Module):
    def __init__(self, fields: List[BaseField], latent_dim: int):
        super().__init__()
        self.fields = fields
        self.decs = nn.ModuleList([f.build_decoder(latent_dim) for f in fields])

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for dec in self.decs:
            y = dec(z)
            outs.append(y)
        return outs
