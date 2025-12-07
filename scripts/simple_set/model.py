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
    
    1. Projects z via a bottleneck (in_dim -> hidden_dim -> proj_dim).
    2. Re-normalizes the result to the hypersphere.
    3. Calculates Cosine Similarity with the output vocabulary weights.
    4. Scales by a learnable temperature.
    """
    def __init__(self, in_dim, proj_dim, vocab_size, hidden_dim, dropout=0.1, init_scale=20.0):
        super().__init__()
        
        # 1. Bottleneck / Projector
        # Note: Skip connection removed, allowing proj_dim != in_dim
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim) 
        )
        
        # 2. The Classifier Weights (No bias in the linear part for Cosine, we add bias separately)
        # Weights match the projection output dimension
        self.weight = nn.Parameter(torch.empty(vocab_size, proj_dim))
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
        # z: [B, in_dim] (Assumed normalized coming in, but we handle it anyway)
        
        # 1. Projection (No Residual)
        # q becomes the "Search Query" for the specific head
        q = self.projector(z)
        
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


class GroupedProjectedHead(nn.Module):
    """
    Splits a large vocabulary into smaller chunks/groups.
    Each chunk gets its own SparseProjectedHead (and thus its own projector/query vector).
    This allows the model to specialize 'queries' for different subsets of the data
    while keeping individual matrices smaller.
    """
    def __init__(self, in_dim, proj_dim, vocab_size, num_groups, hidden_dim, dropout=0.1, init_scale=20.0):
        super().__init__()
        self.chunks = nn.ModuleList()
        
        # Determine chunk size (ceiling division)
        # We keep order: 0..k, k..2k, etc.
        chunk_size = math.ceil(vocab_size / num_groups)
        
        remaining = vocab_size
        while remaining > 0:
            current_size = min(chunk_size, remaining)
            self.chunks.append(
                SparseProjectedHead(
                    in_dim=in_dim,
                    proj_dim=proj_dim,
                    vocab_size=current_size, 
                    hidden_dim=hidden_dim, 
                    dropout=dropout, 
                    init_scale=init_scale
                )
            )
            remaining -= current_size

    def forward(self, z):
        # z: [B, latent_dim]
        # Calculate sub-logits for each chunk and concatenate.
        # Each chunk computes its own independent query vector q based on z.
        chunk_outputs = [chunk(z) for chunk in self.chunks]
        return torch.cat(chunk_outputs, dim=-1)


class HybridSetModel(nn.Module):
    def __init__(
        self,
        fields,
        num_people,
        heads_config,
        head_vocab_sizes,
        head_groups_config=None,
        latent_dim=256,
        hidden_dim=1024,
        proj_dim=None,
        dropout=0.1,
        num_movies=0,
        **kwargs 
    ):
        super().__init__()

        self.fields = fields
        self.latent_dim = int(latent_dim)
        # If proj_dim is not specified, default to latent_dim. 
        # But now they are allowed to differ because skip connection is removed.
        self.proj_dim = int(proj_dim) if proj_dim is not None else self.latent_dim
        
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

        if head_groups_config is None:
            head_groups_config = {}

        for name, _ in heads_config.items():
            vocab = int(head_vocab_sizes.get(name, num_people))
            if vocab <= 0: continue
            
            num_groups = int(head_groups_config.get(name, 1))
            
            if num_groups > 1:
                self.heads[name] = GroupedProjectedHead(
                    in_dim=self.latent_dim,
                    proj_dim=self.proj_dim,
                    vocab_size=vocab,
                    num_groups=num_groups,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    init_scale=init_scale
                )
            else:
                self.heads[name] = SparseProjectedHead(
                    in_dim=self.latent_dim,
                    proj_dim=self.proj_dim,
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
            # z is normalized here, but head_module (or its chunks) 
            # will project and re-normalize internally.
            logits_dict[name] = head_module(z)

        return logits_dict, recon_table, z