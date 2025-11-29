# scripts/set_decoder/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedAdaLN(nn.Module):
    """
    Gated Adaptive Layer Normalization.
    
    1. Global Projection: Converts z_movie into global shift/scale parameters.
    2. Local Gating: Converts current token x into a 0-1 gate.
    3. Combination: The global params are modulated by the local gate.
    """
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        # Standard LayerNorm (elementwise_affine=False because we apply our own affine)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # 1. Global Projection (Movie -> Style)
        # Input: condition_dim
        # Output: 2 * hidden_dim (Gamma, Beta)
        self.film_proj = nn.Linear(condition_dim, 2 * hidden_dim)
        
        # 2. Local Gating (Token -> Relevance)
        # Input: hidden_dim
        # Output: 2 * hidden_dim (GateGamma, GateBeta)
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Sigmoid() # Force values between 0 and 1
        )
        
        # Init to Identity: gamma=0, beta=0
        # This ensures the model starts training behaving like a standard LayerNorm
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        nn.init.zeros_(self.gate_proj[0].weight)
        nn.init.zeros_(self.gate_proj[0].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        x: (B, SeqLen, Hidden)
        condition: (B, CondDim) or (B, 1, CondDim)
        """
        if condition.dim() == 2:
            condition = condition.unsqueeze(1) # (B, 1, CondDim)
            
        # 1. Calculate Global Parameters (The "Signal")
        # global_params: (B, 1, 2*H)
        global_params = self.film_proj(condition)
        
        # 2. Calculate Local Gates (The "Dimmer Switch")
        # gates: (B, SeqLen, 2*H)
        gates = self.gate_proj(x)
        
        # 3. Gating
        # Broadcast Global (1) against Sequence (T)
        # effective_params: (B, SeqLen, 2*H)
        effective_params = global_params * gates
        
        gamma, beta = effective_params.chunk(2, dim=-1) # Split into Scale and Shift
        
        # 4. Apply Norm
        out = self.norm(x)
        out = out * (1 + gamma) + beta
        return out


class FiLMDecoderLayer(nn.Module):
    """
    A Transformer Decoder Layer that uses GatedAdaLN for normalization.
    """
    def __init__(self, hidden_dim: int, latent_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # 1. Self Attention (Causal)
        self.norm1 = GatedAdaLN(hidden_dim, latent_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 2. Cross Attention (Look at Movie Memory)
        self.norm2 = GatedAdaLN(hidden_dim, latent_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 3. Feed Forward
        self.norm3 = GatedAdaLN(hidden_dim, latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, z_movie=None):
        # x: (B, T, H)
        # memory: (B, 1, H) - projected movie latent for cross attn
        # z_movie: (B, D) - raw movie latent for FiLM
        
        # Block 1: Masked Self-Attention
        res = x
        x = self.norm1(x, z_movie)
        x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)
        x = res + self.dropout(x)
        
        # Block 2: Cross-Attention
        res = x
        x = self.norm2(x, z_movie)
        x, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = res + self.dropout(x)
        
        # Block 3: MLP
        res = x
        x = self.norm3(x, z_movie)
        x = self.mlp(x)
        x = res + x
        
        return x


class SequenceDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        max_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.max_len = int(max_len)
        self.hidden_dim = int(hidden_dim)
        
        # Project latents to hidden dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # We also project z_movie to hidden_dim for Cross-Attention keys/values
        self.movie_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Learnable SOS token (represents "Start of Sequence")
        self.sos_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Positional Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, hidden_dim) * 0.02)
        
        # Custom FiLM Decoder Stack
        self.layers = nn.ModuleList([
            FiLMDecoderLayer(hidden_dim, latent_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final Norm
        self.final_norm = GatedAdaLN(hidden_dim, latent_dim)

        # Output Heads
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.presence_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, z_movie: torch.Tensor, z_sequence: torch.Tensor):
        """
        Args:
            z_movie: (B, LatentDim)
            z_sequence: (B, SeqLen, LatentDim)
        """
        B, T, _ = z_sequence.shape
        
        # 1. Prepare Memory (Movie Latent for Cross Attn)
        memory = self.movie_proj(z_movie).unsqueeze(1)
        
        # 2. Prepare Input (Autoregressive Input)
        seq_emb = self.input_proj(z_sequence)
        
        # Prepend SOS & Shift
        sos = self.sos_token.expand(B, -1, -1)
        shifted_seq = seq_emb[:, :-1, :]
        decoder_input = torch.cat([sos, shifted_seq], dim=1)
        
        # 3. Add Positional Embeddings
        decoder_input = decoder_input + self.pos_embedding[:, :T, :]
        
        # 4. Causal Mask
        tgt_mask = self._generate_square_subsequent_mask(T, z_sequence.device)
        
        # 5. Pass through FiLM Layers
        x = decoder_input
        for layer in self.layers:
            x = layer(x, memory=memory, tgt_mask=tgt_mask, z_movie=z_movie)
        
        # 6. Final Norm (also FiLM'd)
        x = self.final_norm(x, z_movie)
        
        # 7. Output Heads
        z_raw = self.latent_head(x)
        z_pred = F.normalize(z_raw, p=2, dim=-1)
        pres_logits = self.presence_head(x).squeeze(-1)
        
        return z_pred, pres_logits

    @torch.no_grad()
    def generate(
        self,
        z_movie: torch.Tensor,
        max_len: int | None = None,
        threshold: float = 0.5
    ):
        if max_len is None: max_len = self.max_len
        
        B = z_movie.size(0)
        device = z_movie.device
        
        memory = self.movie_proj(z_movie).unsqueeze(1)
        
        curr_input = self.sos_token.expand(B, -1, -1)
        curr_input = curr_input + self.pos_embedding[:, 0:1, :]
        
        generated_latents = []
        generated_probs = []
        
        full_input_seq = curr_input
        
        for t in range(max_len):
            tgt_mask = self._generate_square_subsequent_mask(full_input_seq.size(1), device)
            
            x = full_input_seq
            for layer in self.layers:
                x = layer(x, memory=memory, tgt_mask=tgt_mask, z_movie=z_movie)
            
            # Take last step
            last_step = x[:, -1:, :]
            last_step = self.final_norm(last_step, z_movie)
            
            z_raw = self.latent_head(last_step)
            z_next = F.normalize(z_raw, p=2, dim=-1)
            p_next = self.presence_head(last_step).squeeze(-1)
            
            generated_latents.append(z_next.squeeze(1))
            generated_probs.append(torch.sigmoid(p_next).squeeze(1))
            
            if t < max_len - 1:
                next_emb = self.input_proj(z_next)
                next_emb = next_emb + self.pos_embedding[:, t+1:t+2, :]
                full_input_seq = torch.cat([full_input_seq, next_emb], dim=1)
        
        z_out = torch.stack(generated_latents, dim=1)
        p_out = torch.stack(generated_probs, dim=1)
        
        return z_out, p_out