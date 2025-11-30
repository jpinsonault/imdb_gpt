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
        
        # [Fix 1] Build Inverse Mappings (Local -> Global) for correct name lookup
        self.inverse_mappings = self._build_inverse_mappings()

    def _build_inverse_mappings(self):
        """
        Inverts the dataset.head_mappings (Global -> Local) to (Local -> Global)
        so we can look up names from model predictions.
        """
        inv_maps = {}
        if not hasattr(self.dataset, "head_mappings"):
            return inv_maps

        for head, mapping_tensor in self.dataset.head_mappings.items():
            # mapping_tensor: index=Global, value=Local (-1 if not in head)
            # We want: index=Local, value=Global
            
            # 1. Find valid entries
            valid_mask = (mapping_tensor != -1)
            global_indices = torch.nonzero(valid_mask).squeeze(1)
            local_indices = mapping_tensor[global_indices]
            
            # 2. Create inverse array
            max_local = local_indices.max().item() if local_indices.numel() > 0 else 0
            # Initialize with -1
            inverse = torch.full((max_local + 1,), -1, dtype=torch.long)
            
            # 3. Populate
            inverse[local_indices] = global_indices
            inv_maps[head] = inverse
            
        return inv_maps

    def _get_name(self, global_idx):
        return self.dataset.idx_to_name.get(global_idx, f"ID:{global_idx}")

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
    def step(self, global_step, model, inputs, batch_indices, coords_dict, count_targets, run_logger):
        """
        inputs: List[Tensor(B, ...)]
        batch_indices: Tensor(B,) [CPU or GPU]
        coords_dict: {head_name: Tensor(N_sparse, 2)} [row_idx, local_col_idx]
        """
        if (global_step + 1) % self.every != 0: return
        
        model.eval()
        
        B = inputs[0].size(0)
        # 1. Pick indices FIRST to avoid massive dense compute on the whole batch
        indices = np.random.choice(B, size=min(self.num_samples, B), replace=False)
        indices_t = torch.tensor(indices, device=inputs[0].device)

        # 2. Slice inputs to just the samples we want to print
        sliced_inputs = [t[indices_t] for t in inputs]
        
        # Slice batch indices (handle CPU tensor indexed by numpy array)
        sliced_batch_indices = batch_indices[indices]
        
        # 3. Run model ONLY on the slice (Cheap!)
        # We allow full dense expansion here because batch size is tiny (e.g. 3)
        # UPDATED: Unpack 5 return values
        logits_dict, counts_dict, recon_table, recon_enc, _ = model(sliced_inputs, batch_indices=sliced_batch_indices)
        
        # We visualize the Table Reconstruction (Memory) to ensure the table learned the movie stats
        recon_outputs = recon_table 

        output_log = []
        
        # Iterate 0..num_samples (since we sliced, these correspond to the chosen indices)
        for local_slice_idx, global_batch_idx in enumerate(indices):
            
            # --- Metadata ---
            # Inputs are already sliced, so use local_slice_idx
            inp_sample = [t[local_slice_idx].cpu() for t in sliced_inputs]
            rec_sample = [t[local_slice_idx].cpu() for t in recon_outputs]
            movie_title = self._decode_movie_title(inp_sample)
            
            t = PrettyTable(["Field", "Orig", "Recon (Table)"])
            t.align = "l"
            for f, orig, rec in zip(self.dataset.fields, inp_sample, rec_sample):
                t.add_row([f.name, f.render_ground_truth(orig)[:40], f.render_prediction(rec)[:40]])
            
            output_log.append(f"\n=== {movie_title} ===")
            output_log.append(str(t))
            
            # --- Heads ---
            for head in logits_dict.keys():
                inv_map = self.inverse_mappings.get(head)
                
                # Reconstruct True Indices for this sample `global_batch_idx`
                # coords_dict has [batch_idx, local_person_idx]
                true_local_idxs = set()
                coords = coords_dict.get(head)
                if coords is not None:
                    if coords.device.type != 'cpu': coords = coords.cpu()
                    
                    # Filter coords where batch_index == global_batch_idx
                    mask = (coords[:, 0] == global_batch_idx)
                    if mask.any():
                        true_local_idxs = set(coords[mask, 1].numpy().tolist())

                # Counts
                tgt_cnt_t = count_targets.get(head)
                true_count = tgt_cnt_t[global_batch_idx].item() if tgt_cnt_t is not None else 0.0
                pred_count = counts_dict[head][local_slice_idx].item()

                # Get Preds (Dense)
                # local_slice_idx corresponds to the sliced logits
                probs = torch.sigmoid(logits_dict[head][local_slice_idx])
                
                # Fast top-k / thresholding on CPU
                probs_cpu = probs.float().cpu().numpy()
                pred_local_idxs = set(np.where(probs_cpu > self.threshold)[0])
                
                tp_local = true_local_idxs & pred_local_idxs
                fp_local = pred_local_idxs - true_local_idxs
                fn_local = true_local_idxs - pred_local_idxs
                
                def fmt(idx_set):
                    # Convert Local Index -> Global Index -> Name
                    items = []
                    for local_idx in idx_set:
                        p_val = probs_cpu[local_idx]
                        if inv_map is not None and local_idx < len(inv_map):
                            global_idx = inv_map[local_idx].item()
                            if global_idx != -1:
                                name = self._get_name(global_idx)
                            else:
                                name = f"UnkLocal:{local_idx}"
                        else:
                            name = f"UnkLocal:{local_idx}"
                        items.append((name, p_val))
                    
                    # Sort by probability
                    items.sort(key=lambda x:x[1], reverse=True)
                    return items[:10]
                
                output_log.append(f"\n--- Head: {head.upper()} ---")
                output_log.append(f"Count: True={int(true_count)} Pred={pred_count:.1f}")
                output_log.append(self._format_list(fmt(tp_local), "[+] Match"))
                output_log.append(self._format_list(fmt(fp_local), "[x] FalsePos"))
                output_log.append(self._format_list(fmt(fn_local), "[-] Missed"))
                
        full_text = "\n".join(output_log)
        print(full_text)
        if run_logger: run_logger.add_text("recon/text", f"<pre>{full_text}</pre>", global_step)
        
        model.train()