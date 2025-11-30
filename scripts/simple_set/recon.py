# scripts/simple_set/recon.py

import torch
import numpy as np
import textwrap
import logging
from prettytable import PrettyTable

class HybridSetReconLogger:
    def __init__(self, dataset, interval_steps=200, num_samples=3, table_width=80, threshold=0.5):
        self.dataset = dataset
        self.every = max(1, int(interval_steps))
        self.num_samples = num_samples
        self.w = table_width
        self.threshold = threshold

    def _get_name(self, idx):
        return self.dataset.idx_to_name.get(idx, f"ID:{idx}")

    def _decode_movie_title(self, field_tensors):
        title, year = "???", ""
        for f, val in zip(self.dataset.fields, field_tensors):
            s = f.render_ground_truth(val)
            if f.name == "primaryTitle": title = s
            elif f.name == "startYear": year = s
        return f"{title} ({year})"

    def _format_list(self, items, header):
        if not items: return f"{header}: [None]"
        text = ", ".join([f"{n} ({p:.2f})" for n, p in items])
        wrapper = textwrap.TextWrapper(initial_indent=f"{header}: ", subsequent_indent="    ", width=self.w)
        return wrapper.fill(text)

    @torch.no_grad()
    def step(self, global_step, model, inputs, coords_dict, count_targets, run_logger):
        """
        inputs: List[Tensor(B, ...)]
        coords_dict: {head_name: Tensor(N_sparse, 2)} [row_idx, col_idx]
        """
        if (global_step + 1) % self.every != 0: return
        
        model.eval()
        
        B = inputs[0].size(0)
        # 1. Pick indices FIRST to avoid massive dense compute on the whole batch
        indices = np.random.choice(B, size=min(self.num_samples, B), replace=False)
        indices_t = torch.tensor(indices, device=inputs[0].device)

        # 2. Slice inputs to just the samples we want to print
        sliced_inputs = [t[indices_t] for t in inputs]
        
        # 3. Run model ONLY on the slice (Cheap!)
        # We allow full dense expansion here because batch size is tiny (e.g. 3)
        logits_dict, counts_dict, recon_outputs = model(sliced_inputs)
        
        output_log = []
        
        # Iterate 0..num_samples (since we sliced, these correspond to the chosen indices)
        for local_idx, global_idx in enumerate(indices):
            
            # --- Metadata ---
            # Inputs are already sliced, so use local_idx
            inp_sample = [t[local_idx].cpu() for t in sliced_inputs]
            rec_sample = [t[local_idx].cpu() for t in recon_outputs]
            movie_title = self._decode_movie_title(inp_sample)
            
            t = PrettyTable(["Field", "Orig", "Recon"])
            t.align = "l"
            for f, orig, rec in zip(self.dataset.fields, inp_sample, rec_sample):
                t.add_row([f.name, f.render_ground_truth(orig)[:40], f.render_prediction(rec)[:40]])
            
            output_log.append(f"\n=== {movie_title} ===")
            output_log.append(str(t))
            
            # --- Heads ---
            for head in logits_dict.keys():
                # Reconstruct True Indices for this sample `global_idx`
                # We need to look up the original coords using the global batch index
                true_idxs = set()
                coords = coords_dict.get(head)
                if coords is not None:
                    if coords.device.type != 'cpu': coords = coords.cpu()
                    
                    # Filter coords where batch_index == global_idx
                    mask = (coords[:, 0] == global_idx)
                    if mask.any():
                        true_idxs = set(coords[mask, 1].numpy().tolist())

                # Counts
                tgt_cnt_t = count_targets.get(head)
                true_count = tgt_cnt_t[global_idx].item() if tgt_cnt_t is not None else 0.0
                pred_count = counts_dict[head][local_idx].item()

                # Get Preds (Dense)
                # local_idx corresponds to the sliced logits
                probs = torch.sigmoid(logits_dict[head][local_idx])
                
                # Fast top-k / thresholding on CPU
                # We only transfer one row of probs to CPU
                probs_cpu = probs.float().cpu().numpy()
                pred_idxs = set(np.where(probs_cpu > self.threshold)[0])
                
                tp = true_idxs & pred_idxs
                fp = pred_idxs - true_idxs
                fn = true_idxs - pred_idxs
                
                def fmt(idx_set):
                    l = sorted([(self._get_name(x), probs_cpu[x]) for x in idx_set], key=lambda x:x[1], reverse=True)
                    return l[:10]
                
                output_log.append(f"\n--- Head: {head.upper()} ---")
                output_log.append(f"Count: True={int(true_count)} Pred={pred_count:.1f}")
                output_log.append(self._format_list(fmt(tp), "[+] Match"))
                output_log.append(self._format_list(fmt(fp), "[x] FalsePos"))
                output_log.append(self._format_list(fmt(fn), "[-] Missed"))
                
        full_text = "\n".join(output_log)
        print(full_text)
        if run_logger: run_logger.add_text("recon/text", f"<pre>{full_text}</pre>", global_step)
        
        model.train()