# scripts/set_decoder/recon_logger.py

from typing import List, Optional
import numpy as np
import torch
from prettytable import PrettyTable

class SetReconstructionLogger:
    def __init__(
        self,
        model,
        movie_ae,
        people_ae,
        num_slots: int,
        interval_steps: int = 200,
        num_samples: int = 2,
        table_width: int = 60,
    ):
        self.model = model
        self.m_ae = movie_ae
        self.p_ae = people_ae
        self.num_slots = int(num_slots)
        self.every = max(1, int(interval_steps))
        self.num_samples = max(1, int(num_samples))
        self.w = int(table_width)

    def _to_str(self, field, arr):
        """Convert tensor/array to human readable string based on field type."""
        try:
            a = np.array(arr)
            # Text fields with tokenizers
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                return field.to_string(a)
            # Categorical/Numeric fields
            return field.to_string(a)
        except Exception:
            return "..."

    @torch.no_grad()
    def _decode_movie_title(self, z_movie: torch.Tensor) -> str:
        """Decodes the movie latent to find the Primary Title and Year."""
        # Ensure input is (1, D)
        if z_movie.dim() == 1:
            z_movie = z_movie.unsqueeze(0)
        
        # Run decoder
        outs = self.m_ae.decoder(z_movie)
        
        title = "???"
        year = ""
        
        for f, out in zip(self.m_ae.fields, outs):
            val = self._to_str(f, out.detach().cpu().numpy()[0])
            if f.name == "primaryTitle":
                title = val
            elif f.name == "startYear":
                year = val
        
        return f"{title} ({year})" if year else title

    @torch.no_grad()
    def _decode_slots(self, z_slots: torch.Tensor):
        """
        Decodes a set of person latents.
        z_slots: (NumSlots, LatentDim)
        Returns: List of Lists [FieldVal, FieldVal...] per slot
        """
        # Run decoder on batch of slots
        outs = self.p_ae.decoder(z_slots) # List of Tensors, one per field
        
        # Rearrange to: List of Slots -> List of Fields
        num_slots = z_slots.size(0)
        decoded_people = []
        
        for i in range(num_slots):
            person_data = []
            for f_idx, f in enumerate(self.p_ae.fields):
                # outs[f_idx] is (NumSlots, ...)
                val_tensor = outs[f_idx][i]
                val_str = self._to_str(f, val_tensor.detach().cpu().numpy())
                person_data.append((f.name, val_str))
            decoded_people.append(person_data)
            
        return decoded_people

    def step(
        self,
        global_step: int,
        z_movies: torch.Tensor,
        mask: torch.Tensor,
        run_logger,
        sample_tconsts: Optional[List[str]] = None
    ):
        if (global_step + 1) % self.every != 0:
            return

        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Pick samples
        B = z_movies.size(0)
        n_samp = min(self.num_samples, B)
        idxs = np.random.choice(B, size=n_samp, replace=False)

        output_log = []

        with torch.no_grad():
            z_movies_dev = z_movies.to(device)
            
            # Run Model
            # FIX: Model now returns (z_slots, logits) directly, not a list of layers
            z_slots_batch, presence_logits_batch = self.model(z_movies_dev)
            
            probs_batch = torch.sigmoid(presence_logits_batch)

            for i in idxs:
                z_mov = z_movies_dev[i]     # (Latent,)
                z_slots = z_slots_batch[i]  # (Slots, Latent)
                probs = probs_batch[i]      # (Slots,)
                k_true = int(mask[i].sum().item())

                # 1. Decode Movie Context
                movie_str = self._decode_movie_title(z_mov)
                tconst = sample_tconsts[i] if sample_tconsts else "?"
                
                header = f"\n=== {movie_str} [{tconst}] ==="
                subhead = f"True Set Cardinality: {k_true}"
                output_log.append(header)
                output_log.append(subhead)

                # 2. Sort slots by Presence Confidence
                sort_idx = torch.argsort(probs, descending=True)
                
                z_slots_sorted = z_slots[sort_idx]
                probs_sorted = probs[sort_idx]

                # 3. Decode People
                decoded_slots = self._decode_slots(z_slots_sorted)

                # 4. Build Table
                # Columns: Rank | Prob | Field1 | Field2 ...
                field_names = [f.name for f in self.p_ae.fields]
                tab = PrettyTable(["Rank", "Prob"] + field_names)
                tab.align = "l"
                
                for rank, (person_fields, p_val) in enumerate(zip(decoded_slots, probs_sorted)):
                    # Visual cutoff for low probability
                    if rank >= 5 and p_val < 0.1: 
                        break # Don't show garbage slots
                    
                    row = [f"{rank+1}", f"{p_val:.4f}"]
                    for fname, fval in person_fields:
                        # Truncate long strings
                        clean_val = fval[:20] + ".." if len(fval) > 20 else fval
                        row.append(clean_val)
                    tab.add_row(row)

                output_log.append(tab.get_string())

        full_text = "\n".join(output_log)
        print(full_text)
        
        if run_logger and hasattr(run_logger, "add_text"):
            run_logger.add_text("set_decoder/reconstructions", full_text, global_step)

        self.model.train()