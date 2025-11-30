# scripts/simple_set/model.py

import torch
import torch.nn as nn
from scripts.autoencoder.row_autoencoder import _FieldEncoders, TransformerFieldDecoder

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )
        nn.init.constant_(self.net[-2].weight, 0)
        
    def forward(self, x):
        return x + self.net(x)

class HybridSetModel(nn.Module):
    def __init__(
        self,
        fields: list,
        num_people: int,
        heads_config: dict,    # {"cast": 1.0, "director": 0.5} (name -> rank_multiplier)
        latent_dim: int = 128, 
        hidden_dim: int = 1024,
        base_output_rank: int = 64, 
        depth: int = 12,       
        dropout: float = 0.0   
    ):
        super().__init__()
        self.fields = fields
        self.heads_config = heads_config
        
        # 1. Encoders
        self.field_encoder = _FieldEncoders(fields, latent_dim)
        self.field_decoder = TransformerFieldDecoder(fields, latent_dim, num_layers=2, num_heads=4)
        
        self.trunk_proj = nn.Linear(latent_dim, hidden_dim)
        self.trunk = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(depth)])
        
        # 2. Multi-Heads
        self.people_bottlenecks = nn.ModuleDict()
        self.people_expansions = nn.ModuleDict()
        self.count_heads = nn.ModuleDict()
        
        for name, rank_mult in heads_config.items():
            rank = max(8, int(base_output_rank * rank_mult))
            
            # Factorized classification head
            self.people_bottlenecks[name] = nn.Linear(hidden_dim, rank, bias=False)
            self.people_expansions[name] = nn.Linear(rank, num_people)
            
            # Count head
            self.count_heads[name] = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 1)
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, field_tensors: list):
        """
        Returns:
            logits_dict: Dict[str, Tensor(B, NumPeople)]
            counts_dict: Dict[str, Tensor(B, 1)]
            recon_outputs: List[Tensor]
        """
        z = self.field_encoder(field_tensors)
        recon_outputs = self.field_decoder(z)
        
        feat = self.trunk_proj(z)
        feat = self.trunk(feat)
        
        logits_dict = {}
        counts_dict = {}
        
        for name in self.people_bottlenecks.keys():
            # Classify
            bn = self.people_bottlenecks[name](feat)
            logits = self.people_expansions[name](bn)
            logits_dict[name] = logits
            
            # Count
            cnt = self.count_heads[name](feat)
            counts_dict[name] = cnt
            
        return logits_dict, counts_dict, recon_outputs