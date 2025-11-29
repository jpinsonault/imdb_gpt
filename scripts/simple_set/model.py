# scripts/simple_set/model.py

import torch
import torch.nn as nn
from scripts.autoencoder.row_autoencoder import _FieldEncoders

class ResBlock(nn.Module):
    """
    Standard Residual Block.
    """
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
        fields: list,          # List of BaseField objects
        num_people: int,       # Output vocab size
        latent_dim: int = 128, # Output of the Field Transformer
        hidden_dim: int = 1024,# Width of the ResNet Trunk
        output_rank: int = 64, # Low-Rank bottleneck for people head
        depth: int = 12,       # How many ResBlocks
        dropout: float = 0.0   
    ):
        super().__init__()
        self.fields = fields
        
        # 1. Field Encoders & Aggregator
        # This reuses the robust encoder logic from the Joint AE.
        # It handles: Text(CNN), Categorical(Emb), Scalar(Proj) -> Transformer -> Pooling
        # Output is (B, latent_dim)
        self.field_encoder = _FieldEncoders(fields, latent_dim)
        
        # 2. Projection to Trunk
        self.trunk_proj = nn.Linear(latent_dim, hidden_dim)
        
        # 3. Deep ResNet Trunk (Logic Core)
        self.trunk = nn.Sequential(*[
            ResBlock(hidden_dim, dropout) for _ in range(depth)
        ])
        
        # 4. Low-Rank Output Head
        self.people_bottleneck = nn.Linear(hidden_dim, output_rank, bias=False)
        self.people_expansion = nn.Linear(output_rank, num_people) 
        
        # 5. Count Head
        self.count_head = nn.Sequential(
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
        Args:
            field_tensors: List[Tensor], one tensor per field for the batch.
        Returns:
            people_logits: (B, P)
            count_pred: (B, 1)
        """
        # Encode Fields -> Single Vector
        # z: (B, latent_dim)
        z = self.field_encoder(field_tensors)
        
        # Project to Trunk Width
        feat = self.trunk_proj(z) # (B, hidden)
        
        # Deep Logic
        feat = self.trunk(feat)   # (B, hidden)
        
        # Output Factorization
        bottleneck = self.people_bottleneck(feat)         # (B, output_rank)
        people_logits = self.people_expansion(bottleneck) # (B, num_people)
        
        # Count
        count_pred = self.count_head(feat)                # (B, 1)
        
        return people_logits, count_pred