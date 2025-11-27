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
        hidden_mult: float = 2.0,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_slots = int(num_slots)
        
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
        
        # We use the same final norm logic for intermediate auxiliary outputs
        self.final_norm = AdaLN(self.latent_dim, self.latent_dim)

        # --- Shared Output Heads ---
        self.latent_head = nn.Linear(self.latent_dim, self.latent_dim)
        self.presence_head = nn.Linear(self.latent_dim, 1)

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
            List of tuples [(z_slots, presence_logits), ...] per layer.
        """
        B = z_movie.size(0)
        device = z_movie.device
        
        # 1. Initialize Slots with Pure Gaussian Noise
        # No positional embeddings. The model must rely on the noise values + FiLM context.
        x = torch.randn(B, self.num_slots, self.latent_dim, device=device)
        
        outputs = []
        
        # 2. Iterative Refinement (Deep Supervision)
        for block in self.blocks:
            x = block(x, z_movie)
            
            # Auxiliary Output:
            # Apply norm to the current state to prep for heads
            x_aux = self.final_norm(x, z_movie)
            
            # Apply heads
            z_slots_i = self.latent_head(x_aux)
            pres_logits_i = self.presence_head(x_aux).squeeze(-1)
            
            outputs.append((z_slots_i, pres_logits_i))
            
        return outputs

    @torch.no_grad()
    def predict(
        self,
        z_movie: torch.Tensor,
        threshold: float = 0.5,
        top_k: int | None = None,
    ):
        # Run forward pass, grab the FINAL layer output
        outputs = self.forward(z_movie)
        z_slots, presence_logits = outputs[-1]
        
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