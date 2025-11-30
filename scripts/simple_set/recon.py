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
        coords_dict: {head_name: Tensor(N, 2)} [row_idx, col_idx]
        """
        if (global_step + 1) % self.every != 0: return
        model.eval()
        logits_dict, counts_dict, recon_outputs = model(inputs)
        
        B = inputs[0].size(0)
        # Pick random samples
        indices = np.random.choice(B, size=min(self.num_samples, B), replace=False)
        
        output_log = []
        
        for i in indices:
            # 1. Metadata
            inp_sample = [t[i].cpu() for t in inputs]
            rec_sample = [t[i].cpu() for t in recon_outputs]
            movie_title = self._decode_movie_title(inp_sample)
            
            t = PrettyTable(["Field", "Orig", "Recon"])
            t.align = "l"
            for f, orig, rec in zip(self.dataset.fields, inp_sample, rec_sample):
                t.add_row([f.name, f.render_ground_truth(orig)[:40], f.render_prediction(rec)[:40]])
            
            output_log.append(f"\n=== {movie_title} ===")
            output_log.append(str(t))
            
            # 2. Heads
            for head in logits_dict.keys():
                # Reconstruct True Indices from Sparse Coords for this sample `i`
                true_idxs = set()
                coords = coords_dict.get(head)
                if coords is not None:
                    # Filter coords where row == i
                    # Note: coords are on CPU in collate, but might be on GPU if passed from train loop
                    if coords.device.type != 'cpu':
                        coords = coords.cpu()
                    
                    # Boolean mask for this batch item
                    mask = (coords[:, 0] == i)
                    if mask.any():
                        true_idxs = set(coords[mask, 1].numpy().tolist())

                tgt_cnt_t = count_targets.get(head)
                true_count = tgt_cnt_t[i].item() if tgt_cnt_t is not None else 0.0

                # Get Preds
                probs = torch.sigmoid(logits_dict[head][i])
                pred_idxs = set((probs > self.threshold).nonzero().flatten().cpu().numpy())
                pred_count = counts_dict[head][i].item()
                
                tp = true_idxs & pred_idxs
                fp = pred_idxs - true_idxs
                fn = true_idxs - pred_idxs
                
                def fmt(idx_set):
                    l = sorted([(self._get_name(x), probs[x].item()) for x in idx_set], key=lambda x:x[1], reverse=True)
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