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
        self.max_len = int(num_slots)
        self.every = max(1, int(interval_steps))
        self.num_samples = max(1, int(num_samples))
        self.w = int(table_width)

    def _to_str(self, field, arr):
        try:
            a = np.array(arr)
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                return field.to_string(a)
            return field.to_string(a)
        except Exception:
            return "..."

    @torch.no_grad()
    def _decode_movie_title(self, z_movie: torch.Tensor) -> str:
        if z_movie.dim() == 1:
            z_movie = z_movie.unsqueeze(0)
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
    def _decode_sequence(self, z_seq: torch.Tensor):
        """
        Decodes a sequence of person latents.
        z_seq: (T, LatentDim)
        """
        if z_seq.size(0) == 0: return []
        
        outs = self.p_ae.decoder(z_seq)
        seq_len = z_seq.size(0)
        decoded_people = []
        
        for i in range(seq_len):
            person_data = []
            for f_idx, f in enumerate(self.p_ae.fields):
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
        
        B = z_movies.size(0)
        n_samp = min(self.num_samples, B)
        idxs = np.random.choice(B, size=n_samp, replace=False)

        output_log = []

        with torch.no_grad():
            z_movies_dev = z_movies.to(device)
            
            # Autoregressive Generation
            z_gen, p_gen = self.model.generate(z_movies_dev, max_len=self.max_len)
            
            for i in idxs:
                z_mov = z_movies_dev[i]
                z_out = z_gen[i] # (T, D)
                p_out = p_gen[i] # (T)
                
                # Ground truth count
                k_true = int(mask[i].sum().item())

                movie_str = self._decode_movie_title(z_mov)
                tconst = sample_tconsts[i] if sample_tconsts else "?"
                
                header = f"\n=== {movie_str} [{tconst}] ==="
                subhead = f"True Set Cardinality: {k_true}"
                output_log.append(header)
                output_log.append(subhead)

                # Decode Sequence
                decoded_seq = self._decode_sequence(z_out)

                field_names = [f.name for f in self.p_ae.fields]
                tab = PrettyTable(["Step", "Conf"] + field_names)
                tab.align = "l"
                
                for t, (person_fields, prob) in enumerate(zip(decoded_seq, p_out)):
                    prob_val = prob.item()
                    # Visual stop if probability drops too low
                    if prob_val < 0.1 and t >= k_true:
                         # Keep showing if t < k_true to see why it failed, 
                         # but stop showing garbage tail
                         if t > k_true + 2: break
                    
                    row = [f"{t+1}", f"{prob_val:.4f}"]
                    for fname, fval in person_fields:
                        clean_val = fval[:20] + ".." if len(fval) > 20 else fval
                        row.append(clean_val)
                    tab.add_row(row)

                output_log.append(tab.get_string())

        full_text = "\n".join(output_log)
        print(full_text)
        
        if run_logger and hasattr(run_logger, "add_text"):
            run_logger.add_text("seq_decoder/reconstructions", full_text, global_step)

        self.model.train()