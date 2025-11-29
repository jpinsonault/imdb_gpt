# scripts/simple_set/recon.py

import torch
import numpy as np
import textwrap
import logging

class HybridSetReconLogger:
    def __init__(
        self,
        dataset, 
        interval_steps: int = 200,
        num_samples: int = 3,
        table_width: int = 80,
        threshold: float = 0.5
    ):
        self.dataset = dataset
        self.every = max(1, int(interval_steps))
        self.num_samples = num_samples
        self.w = table_width
        self.threshold = threshold

    def _get_name(self, idx):
        return self.dataset.idx_to_name.get(idx, f"ID:{idx}")

    def _decode_movie_title(self, field_tensors_sample):
        """
        field_tensors_sample: List[Tensor] corresponding to one movie
        """
        title = "???"
        year = ""
        
        # We iterate through the dataset fields
        for f, tensor_val in zip(self.dataset.fields, field_tensors_sample):
            # to_string usually expects numpy array
            val_str = f.to_string(tensor_val.numpy())
            
            if f.name == "primaryTitle":
                title = val_str
            elif f.name == "startYear":
                year = val_str
                
        return f"{title} ({year})"

    def _format_list(self, items, header_str):
        if not items:
            return f"{header_str}: [None]"
        entries = [f"{name} ({p:.2f})" for name, p in items]
        full_str = ", ".join(entries)
        prefix = f"{header_str} ({len(items)}): "
        wrapper = textwrap.TextWrapper(initial_indent=prefix, subsequent_indent=" " * 4, width=self.w)
        return wrapper.fill(full_str)

    @torch.no_grad()
    def step(self, global_step: int, model, batch_inputs, targets, count_targets, run_logger):
        if (global_step + 1) % self.every != 0:
            return

        model.eval()
        
        # Forward pass
        logits, pred_counts_scalar = model(batch_inputs)
        probs = torch.sigmoid(logits)
        
        B = logits.size(0)
        n_samp = min(self.num_samples, B)
        sample_idxs = np.random.choice(B, size=n_samp, replace=False)
        
        log_output = []
        
        for i in sample_idxs:
            # Decode Title from Input Tensors
            # batch_inputs is a list of tensors [Field1(B,...), Field2(B,...)]
            # We need to extract the i-th slice from each
            sample_tensors = [t[i].cpu() for t in batch_inputs]
            movie_str = self._decode_movie_title(sample_tensors)
            
            # Gather Ground Truth & Predictions
            true_indices_t = torch.nonzero(targets[i]).flatten()
            true_indices = set(true_indices_t.cpu().numpy().tolist())
            
            pred_indices_t = (probs[i] > self.threshold).nonzero().flatten()
            pred_indices = set(pred_indices_t.cpu().numpy().tolist())
            
            # Set Operations
            tp_idxs = true_indices.intersection(pred_indices)
            fp_idxs = pred_indices - true_indices
            fn_idxs = true_indices - pred_indices
            
            # Helper for display
            def get_display_list(idx_set):
                if not idx_set: return []
                lst = list(idx_set)
                p_vals = probs[i, lst].cpu().numpy()
                zipped = sorted(zip(lst, p_vals), key=lambda x: x[1], reverse=True)
                return [(self._get_name(idx), p) for idx, p in zipped]

            matches_list = get_display_list(tp_idxs)
            extras_list = get_display_list(fp_idxs)
            missed_list = get_display_list(fn_idxs)
            
            true_count = int(count_targets[i].item())
            pred_count_val = pred_counts_scalar[i].item()
            
            header = f"\n=== {movie_str} ==="
            stats = f"Count: True={true_count} | Pred={pred_count_val:.1f} | IoU={len(tp_idxs) / max(1, len(true_indices | pred_indices)):.2f}"
            
            log_output.append(header)
            log_output.append(stats)
            log_output.append("-" * 20)
            
            limit = 15
            
            s_matches = self._format_list(matches_list[:limit], "[+] MATCHES")
            if len(matches_list) > limit: s_matches += f" ... (+{len(matches_list)-limit})"
            
            s_extras = self._format_list(extras_list[:limit], "[x] FALSE POS")
            if len(extras_list) > limit: s_extras += f" ... (+{len(extras_list)-limit})"
            
            s_missed = self._format_list(missed_list[:limit], "[-] MISSED")
            if len(missed_list) > limit: s_missed += f" ... (+{len(missed_list)-limit})"
            
            log_output.append(s_matches)
            log_output.append(s_extras)
            log_output.append(s_missed)

        full_text = "\n".join(log_output)
        print(full_text)
        
        if run_logger:
            run_logger.add_text("hybrid_set/recon", f"<pre>{full_text}</pre>", global_step)
            
        model.train()