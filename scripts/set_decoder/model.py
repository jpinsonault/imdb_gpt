# scripts/set_decoder/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (FiLM).
    Scales and shifts the normalized input based on an external condition (z_movie).
    """
    def __init__(self, latent_dim: int, condition_dim: int):
        super().__init__()
        # Standard LayerNorm (elementwise_affine=False because we predict params)
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        
        # Projection from condition (movie) to gamma (scale) and beta (shift)
        self.film_proj = nn.Linear(condition_dim, 2 * latent_dim)
        
        # Initialize to Identity: gamma=0 (out = x * (1+0) + 0)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        # x: (B, Seq, Dim)
        # condition: (B, CondDim)
        
        params = self.film_proj(condition) # (B, 2*Dim)
        gamma, beta = params.chunk(2, dim=-1) # (B, Dim)
        
        # Expand for sequence length broadcasting
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        out = self.norm(x)
        out = out * (1 + gamma) + beta
        return out


class FiLMTransformerBlock(nn.Module):
    def __init__(self, dim: int, condition_dim: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.attn_norm = AdaLN(dim, condition_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.mlp_norm = AdaLN(dim, condition_dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        # 1. Attention Block (Pre-LN with FiLM)
        norm_x = self.attn_norm(x, condition)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # 2. MLP Block (Pre-LN with FiLM)
        norm_x = self.mlp_norm(x, condition)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        
        return x


class SetDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_slots: int,
        hidden_mult: float,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_slots = int(num_slots)
        
        # --- Learnable Slot Queries (DETR Style) ---
        # Instead of projecting from z_movie, we start with fixed learnable parameters.
        # These represent "Slot 1", "Slot 2", etc., which will be modulated by the movie context.
        self.slot_queries = nn.Parameter(torch.randn(1, self.num_slots, self.latent_dim))

        # --- The FiLM Transformer Backbone ---
        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(
                dim=self.latent_dim, 
                condition_dim=self.latent_dim, 
                num_heads=num_heads, 
                mlp_ratio=hidden_mult
            )
            for _ in range(num_layers)
        ])
        
        # Output Norm
        self.final_norm = AdaLN(self.latent_dim, self.latent_dim)

        # --- Shared Output Heads ---
        self.latent_head = nn.Linear(self.latent_dim, self.latent_dim)
        self.presence_head = nn.Linear(self.latent_dim, 1)

        # --- Learned Null Embedding ---
        # A learnable anchor point for "empty" slots. 
        self.null_embedding = nn.Parameter(torch.randn(1, 1, self.latent_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z_movie: torch.Tensor):
        """
        Args:
            z_movie: (B, LatentDim)
        Returns:
            z_slots: (B, NumSlots, LatentDim) - Normalized
            presence_logits: (B, NumSlots)
        """
        batch_size = z_movie.size(0)

        # 1. Expand Learnable Queries to Batch Size
        # (1, N, D) -> (B, N, D)
        x = self.slot_queries.expand(batch_size, -1, -1)
        
        # 2. Run Backbone with FiLM Conditioning
        # The z_movie modulates the normalization of the slots at every layer
        for block in self.blocks:
            x = block(x, z_movie)
            
        # 3. Final Heads
        x_aux = self.final_norm(x, z_movie)
        
        z_raw = self.latent_head(x_aux)
        # Sphere Awareness: Normalize output latents
        z_slots = F.normalize(z_raw, p=2, dim=-1)
        
        pres_logits = self.presence_head(x_aux).squeeze(-1)
            
        return z_slots, pres_logits

    @torch.no_grad()
    def predict(
        self,
        z_movie: torch.Tensor,
        threshold: float = 0.5,
        top_k: int | None = None,
    ):
        z_slots, presence_logits = self.forward(z_movie)
        
        probs = torch.sigmoid(presence_logits)

        if top_k is None:
            mask = probs > threshold
            if mask.sum(dim=-1).max().item() == 0:
                top_k = 1
            else:
                top_k = self.num_slots

        top_k = min(self.num_slots, int(top_k))
        
        probs_sorted, idx_sorted = probs.sort(dim=-1, descending=True)

        idx_top = idx_sorted[:, :top_k]
        probs_top = probs_sorted[:, :top_k]

        b = z_movie.size(0)
        z_out = []
        p_out = []
        for i in range(b):
            sel = probs_top[i] > threshold
            if sel.sum().item() == 0:
                sel = torch.zeros_like(probs_top[i], dtype=torch.bool)
                sel[0] = True
                
            chosen_idx = idx_top[i][sel]
            z_out.append(z_slots[i, chosen_idx])
            p_out.append(probs[i, chosen_idx])

        return z_out, p_out