import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from ..fields import BaseField
import logging

def _out_dim(m: nn.Module) -> int:
    for p in reversed(list(m.parameters())):
        if p.dim() == 2:
            return p.size(0)
    return None

class RowEncoder(nn.Module):
    def __init__(self, encs, field_names=None):
        super().__init__()
        self.encs = nn.ModuleList(encs)
        self.field_names = list(field_names) if field_names is not None else [f"field{i}" for i in range(len(encs))]
        self._logged_once = False

    def forward(self, xs):
        outs = []
        for i, (enc, x) in enumerate(zip(self.encs, xs)):
            if not self._logged_once:
                n = self.field_names[i] if i < len(self.field_names) else f"field{i}"
                logging.info(f"RowEncoder.bind: {n} -> {type(enc).__name__} ; x.shape={tuple(x.shape)} dtype={x.dtype}")
            y = enc(x)
            outs.append(y)
        self._logged_once = True
        return torch.cat(outs, dim=-1)


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
            if x.dim() == 1:
                x = x.unsqueeze(0)
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
