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
    Now supports Residual connections.
    """
    def __init__(self, in_dim, proj_dim, vocab_size, hidden_dim, dropout=0.1, init_scale=20.0):
        super().__init__()
        
        # 1. Bottleneck / Projector
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, proj_dim)

        # Residual handling: if dims match, we learn a scale factor for the residual
        self.use_res = (in_dim == proj_dim)
        if self.use_res:
            self.res_scale = nn.Parameter(torch.tensor(0.1)) # Start with small residual influence
        
        # 2. The Classifier Weights (No bias in linear, we add bias separately)
        self.weight = nn.Parameter(torch.empty(vocab_size, proj_dim))
        self.bias = nn.Parameter(torch.empty(vocab_size))
        
        # 3. Learnable Scale (Temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(init_scale))

        self._init_weights()

    def _init_weights(self):
        # Standard Init
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)
        
        # Classifier Init: Normalized embeddings work best with Normal init
        nn.init.normal_(self.weight, std=0.02)
        
        # Initialize bias to negative to encourage sparsity at init
        nn.init.constant_(self.bias, -6.0)

    def forward(self, z):
        # z: [B, in_dim]
        
        x = self.in_proj(z)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        q = self.out_proj(x)

        # Scaled Residual
        if self.use_res:
            q = q + (z * self.res_scale)
        
        # Normalize Query and Weights for Cosine Similarity
        q = F.normalize(q, p=2, dim=-1)
        w = F.normalize(self.weight, p=2, dim=-1)
        
        scale = self.logit_scale.exp().clamp(max=100)
        
        logits = F.linear(q, w) * scale + self.bias
        return logits


class SharedGroupedProjectedHead(nn.Module):
    """
    Optimized Grouped Head. 
    Instead of N independent projectors, we use ONE projector that outputs
    a larger context, which is then chunked for the specific groups.
    
    This correlates the groups (learning 'Action Movie' features helps all groups)
    and reduces computational overhead.
    """
    def __init__(self, in_dim, proj_dim, vocab_size, num_groups, hidden_dim, dropout=0.1, init_scale=20.0):
        super().__init__()
        
        self.num_groups = num_groups
        self.proj_dim = proj_dim
        
        # 1. Shared Projector
        # We project to (proj_dim * num_groups) so each group gets a unique 'view' of z
        # but they share the non-linear computation.
        total_out_dim = proj_dim * num_groups
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, total_out_dim)
        )

        # 2. Weights & Biases for chunks
        chunk_size = math.ceil(vocab_size / num_groups)
        self.chunk_sizes = []
        remaining = vocab_size
        
        # We store weights in a ModuleList of simple linear layers (used for storage)
        # We manually handle the normalization in forward
        self.group_weights = nn.ParameterList()
        self.group_biases = nn.ParameterList()

        for _ in range(num_groups):
            c_size = min(chunk_size, remaining)
            self.chunk_sizes.append(c_size)
            
            w = nn.Parameter(torch.empty(c_size, proj_dim))
            b = nn.Parameter(torch.empty(c_size))
            
            nn.init.normal_(w, std=0.02)
            nn.init.constant_(b, -6.0)
            
            self.group_weights.append(w)
            self.group_biases.append(b)
            
            remaining -= c_size
            if remaining <= 0: break

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(init_scale))

    def forward(self, z):
        # 1. Shared Projection
        # [B, proj_dim * num_groups]
        q_full = self.projector(z)
        
        # 2. Split into chunks: [B, num_groups, proj_dim]
        # We reshaped to process groups. 
        # Note: If chunk sizes were identical we could use tensor ops, 
        # but vocab might not divide evenly, so we split.
        q_chunks = torch.split(q_full, self.proj_dim, dim=1)
        
        outputs = []
        scale = self.logit_scale.exp().clamp(max=100)

        for i, (q, w, b) in enumerate(zip(q_chunks, self.group_weights, self.group_biases)):
            # Normalize
            q_norm = F.normalize(q, p=2, dim=-1)
            w_norm = F.normalize(w, p=2, dim=-1)
            
            # Cosine Sim
            logits = F.linear(q_norm, w_norm) * scale + b
            outputs.append(logits)

        return torch.cat(outputs, dim=-1)


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
        self.proj_dim = int(proj_dim) if proj_dim is not None else self.latent_dim
        
        if num_movies <= 0:
            raise ValueError("num_movies must be > 0")

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
                self.heads[name] = SharedGroupedProjectedHead(
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
        z = F.normalize(z, p=2, dim=-1)

        recon_table = self.field_decoder(z)

        logits_dict = {}
        for name, head_module in self.heads.items():
            logits_dict[name] = head_module(z)

        return logits_dict, recon_table, z