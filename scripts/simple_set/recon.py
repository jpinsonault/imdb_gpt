import torch
import numpy as np
import textwrap
import logging
from prettytable import PrettyTable


class HybridSetReconLogger:
    def __init__(
        self,
        movie_dataset,
        person_dataset,
        interval_steps=200,
        num_samples=3,
        table_width=80,
        threshold=0.5,
    ):
        self.movie_dataset = movie_dataset
        self.person_dataset = person_dataset
        self.every = max(1, int(interval_steps))
        self.num_samples = num_samples
        self.w = table_width
        self.threshold = threshold

        self.movie_inverse_mappings = self._build_inverse_mappings(
            getattr(self.movie_dataset, "head_mappings", {})
        )
        self.person_inverse_mappings = self._build_inverse_mappings(
            getattr(self.person_dataset, "head_mappings", {})
        )

    def _build_inverse_mappings(self, head_mappings):
        inv_maps = {}
        if not head_mappings:
            return inv_maps
        for head, mapping_tensor in head_mappings.items():
            if mapping_tensor is None:
                continue
            valid_mask = mapping_tensor != -1
            if not valid_mask.any():
                inv_maps[head] = torch.empty(0, dtype=torch.long)
                continue
            global_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            local_indices = mapping_tensor[global_indices]
            max_local = local_indices.max().item()
            inverse = torch.full((max_local + 1,), -1, dtype=torch.long)
            inverse[local_indices] = global_indices
            inv_maps[head] = inverse
        return inv_maps

    def _decode_movie_title(self, field_tensors):
        title, year = "???", ""
        for f, val in zip(self.movie_dataset.fields, field_tensors):
            s = f.render_ground_truth(val)
            if f.name == "primaryTitle":
                title = s
            elif f.name == "startYear":
                year = s
        return f"{title} ({year})"

    def _decode_person_label(self, field_tensors):
        name, birth = "???", ""
        for f, val in zip(self.person_dataset.fields, field_tensors):
            s = f.render_ground_truth(val)
            if f.name == "primaryName":
                name = s
            elif f.name == "birthYear":
                birth = s
        if birth:
            return f"{name} (b. {birth})"
        return name

    def _get_person_name(self, global_idx):
        return self.movie_dataset.idx_to_name.get(global_idx, f"ID:{global_idx}")

    def _get_movie_title_by_idx(self, global_idx):
        idx = int(global_idx)
        if idx < 0 or idx >= len(self.movie_dataset):
            return f"MovieID:{idx}"
        field_tensors = [
            t[idx].cpu()
            for t in self.movie_dataset.stacked_fields
        ]
        return self._decode_movie_title(field_tensors)

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

    def _movie_block(self, model):
        if len(self.movie_dataset) == 0:
            return []

        B = min(self.num_samples, len(self.movie_dataset))
        movie_indices_np = np.random.choice(len(self.movie_dataset), size=B, replace=False)
        device = next(model.parameters()).device
        movie_indices_t = torch.tensor(movie_indices_np, device=device, dtype=torch.long)

        outputs = model(movie_indices=movie_indices_t)
        if "movie" not in outputs:
            return []
        logits_dict, recon_table, _, _ = outputs["movie"]

        lines = []

        for local_idx, dataset_idx in enumerate(movie_indices_np):
            dataset_idx = int(dataset_idx)

            orig_inputs = [
                self.movie_dataset.stacked_fields[i][dataset_idx].cpu()
                for i in range(len(self.movie_dataset.fields))
            ]
            rec_table_sample = [t[local_idx].cpu() for t in recon_table]

            movie_title = self._decode_movie_title(orig_inputs)

            t = PrettyTable(["Field", "Orig", "Recon"])
            t.align = "l"
            for f, orig, rec_tab_field in zip(self.movie_dataset.fields, orig_inputs, rec_table_sample):
                orig_str = f.render_ground_truth(orig)[:40]
                t.add_row(
                    [
                        f.name,
                        orig_str,
                        f.render_prediction(rec_tab_field)[:40],
                    ]
                )

            lines.append(f"\n=== MOVIE: {movie_title} ===")
            lines.append(str(t))

            for head in logits_dict.keys():
                inv_map = self.movie_inverse_mappings.get(head)
                if inv_map is None or inv_map.numel() == 0:
                    continue

                padded = self.movie_dataset.heads_padded.get(head)
                mapping_tensor = self.movie_dataset.head_mappings.get(head)
                if padded is None or mapping_tensor is None:
                    continue

                row = padded[dataset_idx]
                true_local_idxs = set()
                for v in row.tolist():
                    if v == -1:
                        continue
                    g = int(v)
                    if g < 0 or g >= mapping_tensor.shape[0]:
                        continue
                    loc = int(mapping_tensor[g].item())
                    if loc != -1:
                        true_local_idxs.add(loc)

                true_count = len(true_local_idxs)

                probs = torch.sigmoid(logits_dict[head][local_idx])
                probs_cpu = probs.float().cpu().numpy()
                pred_local_idxs = set(np.where(probs_cpu > self.threshold)[0])
                pred_count = float(probs_cpu.sum())

                tp_local = true_local_idxs & pred_local_idxs
                fp_local = pred_local_idxs - true_local_idxs
                fn_local = true_local_idxs - pred_local_idxs

                def fmt(idx_set):
                    items = []
                    for local_idx2 in idx_set:
                        if local_idx2 < 0 or local_idx2 >= len(inv_map):
                            name = f"UnkLocal:{local_idx2}"
                            p_val = probs_cpu[local_idx2] if 0 <= local_idx2 < len(probs_cpu) else 0.0
                            items.append((name, p_val))
                            continue
                        global_idx = inv_map[local_idx2].item()
                        if global_idx != -1:
                            name = self._get_person_name(global_idx)
                        else:
                            name = f"UnkLocal:{local_idx2}"
                        p_val = probs_cpu[local_idx2] if 0 <= local_idx2 < len(probs_cpu) else 0.0
                        items.append((name, p_val))
                    items.sort(key=lambda x: x[1], reverse=True)
                    return items[:10]

                lines.append(f"\n--- Head: {head.upper()} (Movie -> People) ---")
                lines.append(f"Count: True={int(true_count)} Pred={pred_count:.1f}")
                lines.append(self._format_list(fmt(tp_local), "[+] Match"))
                lines.append(self._format_list(fmt(fp_local), "[x] FalsePos"))
                lines.append(self._format_list(fmt(fn_local), "[-] Missed"))

        return lines

    def _person_block(self, model):
        if len(self.person_dataset) == 0:
            return []

        B = min(self.num_samples, len(self.person_dataset))
        person_indices_np = np.random.choice(len(self.person_dataset), size=B, replace=False)
        device = next(model.parameters()).device
        person_indices_t = torch.tensor(person_indices_np, device=device, dtype=torch.long)

        outputs = model(person_indices=person_indices_t)
        if "person" not in outputs:
            return []
        logits_dict, recon_table, _, _ = outputs["person"]

        lines = []

        for local_idx, dataset_idx in enumerate(person_indices_np):
            dataset_idx = int(dataset_idx)

            orig_inputs = [
                self.person_dataset.stacked_fields[i][dataset_idx].cpu()
                for i in range(len(self.person_dataset.fields))
            ]
            rec_table_sample = [t[local_idx].cpu() for t in recon_table]

            person_label = self._decode_person_label(orig_inputs)

            t = PrettyTable(["Field", "Orig", "Recon"])
            t.align = "l"
            for f, orig, rec_tab_field in zip(self.person_dataset.fields, orig_inputs, rec_table_sample):
                orig_str = f.render_ground_truth(orig)[:40]
                t.add_row(
                    [
                        f.name,
                        orig_str,
                        f.render_prediction(rec_tab_field)[:40],
                    ]
                )

            lines.append(f"\n=== PERSON: {person_label} ===")
            lines.append(str(t))

            for head in logits_dict.keys():
                inv_map = self.person_inverse_mappings.get(head)
                if inv_map is None or inv_map.numel() == 0:
                    continue

                padded = self.person_dataset.heads_padded.get(head)
                mapping_tensor = self.person_dataset.head_mappings.get(head)
                if padded is None or mapping_tensor is None:
                    continue

                row = padded[dataset_idx]
                true_local_idxs = set()
                for v in row.tolist():
                    if v == -1:
                        continue
                    g = int(v)
                    if g < 0 or g >= mapping_tensor.shape[0]:
                        continue
                    loc = int(mapping_tensor[g].item())
                    if loc != -1:
                        true_local_idxs.add(loc)

                true_count = len(true_local_idxs)

                probs = torch.sigmoid(logits_dict[head][local_idx])
                probs_cpu = probs.float().cpu().numpy()
                pred_local_idxs = set(np.where(probs_cpu > self.threshold)[0])
                pred_count = float(probs_cpu.sum())

                tp_local = true_local_idxs & pred_local_idxs
                fp_local = pred_local_idxs - true_local_idxs
                fn_local = true_local_idxs - pred_local_idxs

                def fmt(idx_set):
                    items = []
                    for local_idx2 in idx_set:
                        if local_idx2 < 0 or local_idx2 >= len(inv_map):
                            name = f"UnkLocal:{local_idx2}"
                            p_val = probs_cpu[local_idx2] if 0 <= local_idx2 < len(probs_cpu) else 0.0
                            items.append((name, p_val))
                            continue
                        global_idx = inv_map[local_idx2].item()
                        if global_idx != -1:
                            name = self._get_movie_title_by_idx(global_idx)
                        else:
                            name = f"UnkLocal:{local_idx2}"
                        p_val = probs_cpu[local_idx2] if 0 <= local_idx2 < len(probs_cpu) else 0.0
                        items.append((name, p_val))
                    items.sort(key=lambda x: x[1], reverse=True)
                    return items[:10]

                lines.append(f"\n--- Head: {head.upper()} (Person -> Movies) ---")
                lines.append(f"Count: True={int(true_count)} Pred={pred_count:.1f}")
                lines.append(self._format_list(fmt(tp_local), "[+] Match"))
                lines.append(self._format_list(fmt(fp_local), "[x] FalsePos"))
                lines.append(self._format_list(fmt(fn_local), "[-] Missed"))

        return lines

    @torch.no_grad()
    def step(self, global_step, model, run_logger):
        if (global_step + 1) % self.every != 0:
            return

        model.eval()

        output_log = []

        try:
            output_log.extend(self._movie_block(model))
        except Exception as e:
            logging.warning(f"Movie recon logging failed: {e}")

        try:
            output_log.extend(self._person_block(model))
        except Exception as e:
            logging.warning(f"Person recon logging failed: {e}")

        if not output_log:
            model.train()
            return

        full_text = "\n".join(output_log)
        print(full_text)
        if run_logger:
            run_logger.add_text("recon/text", f"<pre>{full_text}</pre>", global_step)

        model.train()
