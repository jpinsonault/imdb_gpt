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
        self.inverse_mappings = self._build_inverse_mappings()

    def _build_inverse_mappings(self):
        inv_maps = {}
        if not hasattr(self.dataset, "head_mappings"):
            return inv_maps
        for head, mapping_tensor in self.dataset.head_mappings.items():
            valid_mask = mapping_tensor != -1
            global_indices = torch.nonzero(valid_mask).squeeze(1)
            local_indices = mapping_tensor[global_indices]
            if local_indices.numel() == 0:
                inv_maps[head] = torch.empty(0, dtype=torch.long)
                continue
            max_local = local_indices.max().item()
            inverse = torch.full((max_local + 1,), -1, dtype=torch.long)
            inverse[local_indices] = global_indices
            inv_maps[head] = inverse
        return inv_maps

    def _get_name(self, global_idx):
        return self.dataset.idx_to_name.get(global_idx, f"ID:{global_idx}")

    def _decode_movie_title(self, field_tensors):
        title, year = "???", ""
        for f, val in zip(self.dataset.fields, field_tensors):
            s = f.render_ground_truth(val)
            if f.name == "primaryTitle":
                title = s
            elif f.name == "startYear":
                year = s
        return f"{title} ({year})"

    def _format_list(self, items, header):
        if not items:
            return f"{header}: [None]"
        text = ", ".join([f"{n} ({p:.2f})" for n, p in items])
        wrapper = textwrap.TextWrapper(
            initial_indent=f"{header}: ",
            subsequent_indent="    ",
            width=self.w,
        )
        return wrapper.fill(text)

    @torch.no_grad()
    def step(self, global_step, model, inputs, batch_indices, coords_dict, count_targets, run_logger):
        if (global_step + 1) % self.every != 0:
            return

        model.eval()

        B = inputs[0].size(0)
        if B == 0:
            return

        indices = np.random.choice(B, size=min(self.num_samples, B), replace=False)
        indices_t = torch.tensor(indices, device=inputs[0].device)

        sliced_inputs = [t[indices_t] for t in inputs]
        sliced_batch_indices = batch_indices[indices]

        logits_dict, counts_dict, recon_table, _ = model(
            sliced_inputs,
            batch_indices=sliced_batch_indices,
        )

        output_log = []

        for local_slice_idx, batch_row_idx in enumerate(indices):
            dataset_idx = int(sliced_batch_indices[local_slice_idx])

            orig_inputs = [
                self.dataset.stacked_fields[i][dataset_idx].cpu()
                for i in range(len(self.dataset.fields))
            ]
            rec_table_sample = [t[local_slice_idx].cpu() for t in recon_table]

            movie_title = self._decode_movie_title(orig_inputs)

            t = PrettyTable(["Field", "Orig", "Recon"])
            t.align = "l"
            for f, orig, rec_tab_field in zip(self.dataset.fields, orig_inputs, rec_table_sample):
                orig_str = f.render_ground_truth(orig)[:40]
                t.add_row(
                    [
                        f.name,
                        orig_str,
                        f.render_prediction(rec_tab_field)[:40],
                    ]
                )

            output_log.append(f"\n=== {movie_title} ===")
            output_log.append(str(t))

            for head in logits_dict.keys():
                inv_map = self.inverse_mappings.get(head)

                true_local_idxs = set()
                coords = coords_dict.get(head)
                if coords is not None:
                    if coords.device.type != "cpu":
                        coords = coords.cpu()
                    mask = coords[:, 0] == dataset_idx
                    if mask.any():
                        true_local_idxs = set(coords[mask, 1].numpy().tolist())

                tgt_cnt_t = count_targets.get(head) if count_targets is not None else None
                if tgt_cnt_t is not None:
                    true_count = tgt_cnt_t[dataset_idx].item()
                else:
                    true_count = 0.0
                pred_count = counts_dict[head][local_slice_idx].item()

                probs = torch.sigmoid(logits_dict[head][local_slice_idx])
                probs_cpu = probs.float().cpu().numpy()
                pred_local_idxs = set(np.where(probs_cpu > self.threshold)[0])

                tp_local = true_local_idxs & pred_local_idxs
                fp_local = pred_local_idxs - true_local_idxs
                fn_local = true_local_idxs - pred_local_idxs

                def fmt(idx_set):
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
                    items.sort(key=lambda x: x[1], reverse=True)
                    return items[:10]

                output_log.append(f"\n--- Head: {head.upper()} ---")
                output_log.append(f"Count: True={int(true_count)} Pred={pred_count:.1f}")
                output_log.append(self._format_list(fmt(tp_local), "[+] Match"))
                output_log.append(self._format_list(fmt(fp_local), "[x] FalsePos"))
                output_log.append(self._format_list(fmt(fn_local), "[-] Missed"))

        full_text = "\n".join(output_log)
        print(full_text)
        if run_logger:
            run_logger.add_text("recon/text", f"<pre>{full_text}</pre>", global_step)

        model.train()
