# scripts/simple_set/model.py

import torch
import torch.nn as nn
from scripts.autoencoder.row_autoencoder import TransformerFieldDecoder, _FieldEncoders

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
        heads_config: dict,     
        head_vocab_sizes: dict, 
        latent_dim: int = 128, 
        hidden_dim: int = 1024,
        base_output_rank: int = 64, 
        depth: int = 12,        
        dropout: float = 0.0    
    ):
        super().__init__()
        self.fields = fields
        self.heads_config = heads_config
        self.latent_dim = latent_dim
        
        # 1. Learnable Encoder (Trained from Scratch)
        self.field_encoder = _FieldEncoders(fields, latent_dim)
        
        # 2. Decoder (Regularizer)
        self.field_decoder = TransformerFieldDecoder(fields, latent_dim, num_layers=2, num_heads=4)
        
        # 3. Trunk
        self.trunk_proj = nn.Linear(latent_dim, hidden_dim)
        self.trunk = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(depth)])
        
        # 4. Multi-Heads
        self.people_bottlenecks = nn.ModuleDict()
        self.people_expansions = nn.ModuleDict()
        self.count_heads = nn.ModuleDict()
        
        for name, rank_mult in heads_config.items():
            rank = max(8, int(base_output_rank * rank_mult))
            vocab = head_vocab_sizes.get(name, num_people)
            
            self.people_bottlenecks[name] = nn.Linear(hidden_dim, rank, bias=False)
            self.people_expansions[name] = nn.Linear(rank, vocab)
            
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

    def forward(self, field_tensors: list, return_embeddings: bool = False):
        """
        Args:
            field_tensors: List[Tensor] - Raw input fields
        """
        # 1. Encode fields to Latent Z (E2E training)
        z = self.field_encoder(field_tensors)
        
        # 2. Decode Z (Regularization)
        recon_outputs = self.field_decoder(z)
        
        # 3. Trunk
        feat = self.trunk_proj(z)
        feat = self.trunk(feat)
        
        logits_dict = {}
        counts_dict = {}
        
        for name in self.people_bottlenecks.keys():
            bn = self.people_bottlenecks[name](feat)
            
            if return_embeddings:
                logits_dict[name] = bn 
            else:
                logits = self.people_expansions[name](bn)
                logits_dict[name] = logits
            
            cnt = self.count_heads[name](feat)
            counts_dict[name] = cnt
            
        return logits_dict, counts_dict, recon_outputs