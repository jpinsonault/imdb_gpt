# scripts/simple_set/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.autoencoder.row_autoencoder import TransformerFieldDecoder
from scripts.autoencoder.fields import TextField


class SparseProjectedHead(nn.Module):
    """
    Projects input z to a query vector, then uses a standard Linear layer
    to map to vocabulary size.
    
    Crucial Feature: Bias is initialized to a low value to handle sparsity.
    """
    def __init__(self, in_dim, vocab_size, hidden_dim, dropout=0.1):
        super().__init__()
        
        # 1. Bottleneck / Projector
        # Isolate the "Cast" information from the generic "Movie" embedding
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim) 
        )
        
        # 2. The Classifier
        self.out_layer = nn.Linear(in_dim, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        # Standard Init for projector
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Classifier Init
        nn.init.normal_(self.out_layer.weight, std=0.01)
        
        # CRITICAL: Initialize bias to -6.0
        # sigmoid(-6.0) ~= 0.002. 
        # This tells the model: "Start by assuming everyone is NOT in the movie."
        nn.init.constant_(self.out_layer.bias, -6.0)

    def forward(self, z):
        # z: [B, latent_dim]
        q = self.projector(z)
        logits = self.out_layer(q)
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
        **kwargs # Swallow extra args like init_scale
    ):
        super().__init__()

        self.fields = fields
        self.latent_dim = int(latent_dim)
        
        if num_movies <= 0:
            raise ValueError("num_movies must be > 0")

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
                dropout=dropout
            )

    def forward(self, field_tensors, batch_indices):
        if batch_indices is None:
            raise ValueError("batch_indices is required.")

        idx = batch_indices.to(self.movie_embeddings.weight.device, non_blocking=True)
        z = self.movie_embeddings(idx) 
        
        # Normalize for stability
        z = F.normalize(z, p=2, dim=-1)

        recon_table = self.field_decoder(z)

        logits_dict = {}
        for name, head_module in self.heads.items():
            logits_dict[name] = head_module(z)

        return logits_dict, recon_table, z