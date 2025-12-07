# scripts/simple_set/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.autoencoder.row_autoencoder import TransformerFieldDecoder
from scripts.autoencoder.fields import TextField


class SparseProjectedHead(nn.Module):
    """
    Projects input z to a query vector using a Cosine-Similarity based approach.
    
    1. Refines z via a residual bottleneck.
    2. Re-normalizes the result to the hypersphere.
    3. Calculates Cosine Similarity with the output vocabulary weights.
    4. Scales by a learnable temperature.
    """
    def __init__(self, in_dim, vocab_size, hidden_dim, dropout=0.1, init_scale=20.0):
        super().__init__()
        
        # 1. Bottleneck / Projector
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim) 
        )
        
        # 2. The Classifier Weights (No bias in the linear part for Cosine, we add bias separately)
        self.weight = nn.Parameter(torch.empty(vocab_size, in_dim))
        self.bias = nn.Parameter(torch.empty(vocab_size))
        
        # 3. Learnable Scale (Temperature)
        # We store it as a log value for numerical stability
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(init_scale))

        self._init_weights()

    def _init_weights(self):
        # Standard Init for projector
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Classifier Init (Xavier Normal often works well for normalized embeddings)
        nn.init.xavier_normal_(self.weight)
        
        # CRITICAL: Initialize bias to -6.0
        # sigmoid(-6.0) ~= 0.002. 
        # This tells the model: "Start by assuming everyone is NOT in the movie."
        nn.init.constant_(self.bias, -6.0)

    def forward(self, z):
        # z: [B, latent_dim] (Assumed normalized coming in, but we handle it anyway)
        
        # 1. Residual Refinement
        # q becomes the "Search Query" for the specific head (e.g., Cast vs Director)
        residual = self.projector(z)
        q = z + residual
        
        # 2. Re-Normalize
        # We want strict Cosine Similarity, so the query must be unit length.
        q = F.normalize(q, p=2, dim=-1)
        
        # 3. Normalize Weights
        w = F.normalize(self.weight, p=2, dim=-1)
        
        # 4. Scaled Dot Product (Cosine Similarity)
        # logits = scale * (q @ w.T) + bias
        scale = self.logit_scale.exp().clamp(max=100) # Clamp for stability
        
        logits = F.linear(q, w) * scale + self.bias
        
        return logits


class HybridSetModel(nn.Module):
    def __init__(
        self,
        fields,
        num_people,
        heads_config,
        head_vocab_sizes,
        latent_dim=256,
        hidden_dim=1024,
        dropout=0.1,
        num_movies=0,
        **kwargs 
    ):
        super().__init__()

        self.fields = fields
        self.latent_dim = int(latent_dim)
        
        if num_movies <= 0:
            raise ValueError("num_movies must be > 0")

        # Extract config values potentially passed in kwargs
        init_scale = kwargs.get("hybrid_set_logit_scale", 20.0)

        self.movie_embeddings = nn.Embedding(num_movies, self.latent_dim)
        nn.init.normal_(self.movie_embeddings.weight, std=0.02)

        self.field_decoder = TransformerFieldDecoder(
            fields,
            self.latent_dim,
            num_layers=2,
            num_heads=4,
        )

        self.heads = nn.ModuleDict()

        for name, _ in heads_config.items():
            vocab = int(head_vocab_sizes.get(name, num_people))
            if vocab <= 0: continue
            
            self.heads[name] = SparseProjectedHead(
                in_dim=self.latent_dim,
                vocab_size=vocab,
                hidden_dim=hidden_dim,
                dropout=dropout,
                init_scale=init_scale
            )

    def forward(self, field_tensors, batch_indices):
        if batch_indices is None:
            raise ValueError("batch_indices is required.")

        idx = batch_indices.to(self.movie_embeddings.weight.device, non_blocking=True)
        z = self.movie_embeddings(idx) 
        
        # Normalize for stability
        # This keeps the embedding space spherical
        z = F.normalize(z, p=2, dim=-1)

        recon_table = self.field_decoder(z)

        logits_dict = {}
        for name, head_module in self.heads.items():
            # z is normalized here, but head_module will add residual and re-normalize
            logits_dict[name] = head_module(z)

        return logits_dict, recon_table, z