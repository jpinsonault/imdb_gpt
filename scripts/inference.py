# scripts/inference.py

import logging
import time
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from config import project_config, ProjectConfig
from scripts.simple_set.precompute import ensure_hybrid_cache
from scripts.simple_set.data import HybridSetDataset, PersonHybridSetDataset
from scripts.simple_set.model import HybridSetModel
from scripts.autoencoder.fields import TextField
from scripts.sql_filters import (
    movie_select_clause,
    people_select_clause,
    map_movie_row,
    map_person_row,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class HybridSearchEngine:
    def __init__(self, device: Optional[str] = None, cfg: Optional[ProjectConfig] = None):
        self.cfg = cfg or project_config

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        logging.info("Initializing HybridSearchEngine on %s", self.device)

        cache_path = ensure_hybrid_cache(self.cfg)
        cache_path = Path(cache_path)

        self.movie_ds = HybridSetDataset(str(cache_path), self.cfg)
        self.person_ds = PersonHybridSetDataset(str(cache_path), self.cfg)

        self.num_movies = len(self.movie_ds)
        self.num_people = self.movie_ds.num_people

        logging.info("Loaded movie dataset with %d items", self.num_movies)
        logging.info("Loaded person dataset with %d items", self.num_people)

        self.model = HybridSetModel(
            movie_fields=self.movie_ds.fields,
            person_fields=self.person_ds.fields,
            num_movies=self.num_movies,
            num_people=self.num_people,
            heads_config=self.cfg.hybrid_set_heads,
            movie_head_vocab_sizes=self.movie_ds.head_vocab_sizes,
            movie_head_local_to_global=self.movie_ds.head_local_to_global,
            person_head_vocab_sizes=self.person_ds.head_vocab_sizes,
            person_head_local_to_global=self.person_ds.head_local_to_global,
            movie_dim=self.cfg.hybrid_set_movie_dim,
            hidden_dim=self.cfg.hybrid_set_hidden_dim,
            person_dim=self.cfg.hybrid_set_person_dim,
            dropout=self.cfg.hybrid_set_dropout,
            logit_scale=self.cfg.hybrid_set_logit_scale,
            film_bottleneck_dim=self.cfg.hybrid_set_film_bottleneck_dim,
            noise_std=self.cfg.hybrid_set_noise_std,
            decoder_num_layers=self.cfg.hybrid_set_decoder_num_layers,
            decoder_num_heads=self.cfg.hybrid_set_decoder_num_heads,
            decoder_ff_multiplier=self.cfg.hybrid_set_decoder_ff_multiplier,
            decoder_dropout=self.cfg.hybrid_set_decoder_dropout,
            decoder_norm_first=self.cfg.hybrid_set_decoder_norm_first,
        )

        ckpt_path = Path(self.cfg.model_dir) / "hybrid_set_state.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        logging.info("Loading HybridSetModel weights from %s", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logging.info("Missing keys in checkpoint (new params): %s", missing)
        if unexpected:
            logging.warning("Unexpected keys in checkpoint: %s", unexpected)
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            self.movie_embedding_table = F.normalize(
                self.model.movie_embeddings.weight.detach().to(self.device),
                p=2,
                dim=-1,
            )
            self.person_embedding_table = F.normalize(
                self.model.person_embeddings.weight.detach().to(self.device),
                p=2,
                dim=-1,
            )

        self.movie_field_idx = {f.name: i for i, f in enumerate(self.movie_ds.fields)}
        self.person_field_idx = {f.name: i for i, f in enumerate(self.person_ds.fields)}

        self.movie_title_field = self._find_text_field(self.movie_ds.fields, "primaryTitle")
        self.person_name_field = self._find_text_field(self.person_ds.fields, "primaryName")

        if self.movie_title_field is None:
            raise RuntimeError("No TextField named 'primaryTitle' found in movie fields")
        if self.person_name_field is None:
            raise RuntimeError("No TextField named 'primaryName' found in person fields")

        logging.info("Decoding movie titles and IDs for search")
        self.movie_titles: List[str] = []
        self.movie_titles_lower: List[str] = []
        self.movie_tconst: List[str] = []
        self.tconst_to_index: Dict[str, int] = {}

        tconst_field = self.movie_ds.fields[self.movie_field_idx["tconst"]]
        for idx in range(self.num_movies):
            title_tokens = self.movie_ds.stacked_fields[self.movie_field_idx["primaryTitle"]][idx]
            title_str = self.movie_title_field.render_ground_truth(title_tokens.cpu())
            self.movie_titles.append(title_str)
            self.movie_titles_lower.append(title_str.lower())

            tconst_tokens = self.movie_ds.stacked_fields[self.movie_field_idx["tconst"]][idx]
            digits_str = tconst_field.render_ground_truth(tconst_tokens.cpu())
            tconst = self._digits_to_id(digits_str, "tt")
            self.movie_tconst.append(tconst)
            if tconst:
                if tconst not in self.tconst_to_index:
                    self.tconst_to_index[tconst] = idx

        logging.info("Decoding person names and IDs for search")
        self.person_names: List[str] = []
        self.person_names_lower: List[str] = []
        self.person_nconst: List[str] = []
        self.nconst_to_index: Dict[str, int] = {}

        nconst_field = self.person_ds.fields[self.person_field_idx["nconst"]]
        for idx in range(self.num_people):
            name_tokens = self.person_ds.stacked_fields[self.person_field_idx["primaryName"]][idx]
            name_str = self.person_name_field.render_ground_truth(name_tokens.cpu())
            self.person_names.append(name_str)
            self.person_names_lower.append(name_str.lower())

            nconst_tokens = self.person_ds.stacked_fields[self.person_field_idx["nconst"]][idx]
            digits_str = nconst_field.render_ground_truth(nconst_tokens.cpu())
            nconst = self._digits_to_id(digits_str, "nm")
            self.person_nconst.append(nconst)
            if nconst:
                if nconst not in self.nconst_to_index:
                    self.nconst_to_index[nconst] = idx

        self.db_path = self.cfg.db_path

        # Detect whether search encoders were loaded (not just randomly initialized)
        self.has_movie_search_encoder = (
            self.model.movie_title_encoder is not None
            and not missing  # encoder keys were present in checkpoint
            or (self.model.movie_title_encoder is not None
                and not any("movie_title_encoder" in k for k in missing))
        )
        self.has_person_search_encoder = (
            self.model.person_name_encoder is not None
            and not any("person_name_encoder" in k for k in missing)
        )

        if self.has_movie_search_encoder:
            logging.info("Movie title search encoder available — using vector search")
        else:
            logging.info("No trained movie title encoder — falling back to string search")

        if self.has_person_search_encoder:
            logging.info("Person name search encoder available — using vector search")
        else:
            logging.info("No trained person name encoder — falling back to string search")

        logging.info("HybridSearchEngine ready")

    def _find_text_field(self, fields, name: str) -> Optional[TextField]:
        for f in fields:
            if isinstance(f, TextField) and f.name == name:
                return f
        return None

    @staticmethod
    def _digits_to_id(s: str, prefix: str) -> str:
        s = str(s or "").strip()
        if not s:
            return ""
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            return ""
        return prefix + digits.zfill(7)

    def _string_similarity(self, q: str, text: str) -> float:
        q = (q or "").strip().lower()
        t = (text or "").strip().lower()
        if not q or not t:
            return 0.0
        if q == t:
            return 1.0
        if t.startswith(q):
            return 0.95
        if q in t:
            return 0.85

        q_tokens = [tok for tok in q.split() if tok]
        t_tokens = [tok for tok in t.split() if tok]
        if not q_tokens or not t_tokens:
            return 0.0

        qs = set(q_tokens)
        ts = set(t_tokens)
        inter = len(qs & ts)
        if inter == 0:
            return 0.0
        union = len(qs | ts)
        base = inter / float(union)
        return 0.6 + 0.4 * base

    def _tokenize_title(self, text: str) -> torch.Tensor:
        tokens = self.movie_title_field._transform(text)
        return tokens.unsqueeze(0).to(self.device)

    def _tokenize_name(self, text: str) -> torch.Tensor:
        tokens = self.person_name_field._transform(text)
        return tokens.unsqueeze(0).to(self.device)

    def _vector_search_movies(self, title: str, top_k: int = 50) -> List[Dict[str, Any]]:
        tokens = self._tokenize_title(title)
        query_vec = self.model.encode_movie_query(tokens)  # (1, dim)
        sims = torch.mv(self.movie_embedding_table, query_vec.squeeze(0))  # (num_movies,)
        k = min(top_k, sims.shape[0])
        topk_vals, topk_idx = torch.topk(sims, k)

        results: List[Dict[str, Any]] = []
        for rank, (sim_val, idx) in enumerate(zip(topk_vals.cpu().tolist(), topk_idx.cpu().tolist()), start=1):
            row = self._decode_movie_row_from_dataset(idx)
            row["_id"] = int(idx)
            row["_rank"] = int(rank)
            row["_score"] = float(1.0 - sim_val)
            results.append(row)
        return results

    def _vector_search_people(self, name: str, top_k: int = 50) -> List[Dict[str, Any]]:
        tokens = self._tokenize_name(name)
        query_vec = self.model.encode_person_query(tokens)  # (1, dim)
        sims = torch.mv(self.person_embedding_table, query_vec.squeeze(0))  # (num_people,)
        k = min(top_k, sims.shape[0])
        topk_vals, topk_idx = torch.topk(sims, k)

        results: List[Dict[str, Any]] = []
        for rank, (sim_val, idx) in enumerate(zip(topk_vals.cpu().tolist(), topk_idx.cpu().tolist()), start=1):
            row = self._decode_person_row_from_dataset(idx)
            row["_id"] = int(idx)
            row["_rank"] = int(rank)
            row["_score"] = float(1.0 - sim_val)
            results.append(row)
        return results

    def _string_search_movies(self, title: str, top_k: int = 50) -> List[Dict[str, Any]]:
        query_lower = title.lower()
        sims: List[tuple[int, float]] = []
        for idx, candidate in enumerate(self.movie_titles_lower):
            sim = self._string_similarity(query_lower, candidate)
            if sim > 0.0:
                sims.append((idx, sim))

        if not sims:
            return []

        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[: max(1, min(int(top_k), len(sims)))]

        results: List[Dict[str, Any]] = []
        for rank, (idx, sim) in enumerate(sims, start=1):
            row = self._decode_movie_row_from_dataset(idx)
            row["_id"] = int(idx)
            row["_rank"] = int(rank)
            row["_score"] = float(1.0 - sim)
            results.append(row)
        return results

    def _string_search_people(self, name: str, top_k: int = 50) -> List[Dict[str, Any]]:
        query_lower = name.lower()
        sims: List[tuple[int, float]] = []
        for idx, candidate in enumerate(self.person_names_lower):
            sim = self._string_similarity(query_lower, candidate)
            if sim > 0.0:
                sims.append((idx, sim))

        if not sims:
            return []

        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[: max(1, min(int(top_k), len(sims)))]

        results: List[Dict[str, Any]] = []
        for rank, (idx, sim) in enumerate(sims, start=1):
            row = self._decode_person_row_from_dataset(idx)
            row["_id"] = int(idx)
            row["_rank"] = int(rank)
            row["_score"] = float(1.0 - sim)
            results.append(row)
        return results

    def _decode_movie_row_from_dataset(self, idx: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for field_idx, f in enumerate(self.movie_ds.fields):
            tensor = self.movie_ds.stacked_fields[field_idx][idx]
            row[f.name] = f.render_ground_truth(tensor.cpu())
        row["tconst"] = self.movie_tconst[idx]
        row["entityType"] = "movie"
        return row

    def _decode_person_row_from_dataset(self, idx: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for field_idx, f in enumerate(self.person_ds.fields):
            tensor = self.person_ds.stacked_fields[field_idx][idx]
            row[f.name] = f.render_ground_truth(tensor.cpu())
        row["nconst"] = self.person_nconst[idx]
        row["entityType"] = "person"
        return row

    def search_movies(self, title: str, top_k: int = 50) -> List[Dict[str, Any]]:
        title = (title or "").strip()
        if not title:
            return []

        t0 = time.perf_counter()

        if self.has_movie_search_encoder:
            results = self._vector_search_movies(title, top_k=top_k)
        else:
            results = self._string_search_movies(title, top_k=top_k)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        method = "vector" if self.has_movie_search_encoder else "string"
        logging.info("Movie search (%s) for '%s' returned %d hits in %.2fms", method, title, len(results), duration_ms)
        return results

    def search_people(self, name: str, top_k: int = 50) -> List[Dict[str, Any]]:
        name = (name or "").strip()
        if not name:
            return []

        t0 = time.perf_counter()

        if self.has_person_search_encoder:
            results = self._vector_search_people(name, top_k=top_k)
        else:
            results = self._string_search_people(name, top_k=top_k)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        method = "vector" if self.has_person_search_encoder else "string"
        logging.info("Person search (%s) for '%s' returned %d hits in %.2fms", method, name, len(results), duration_ms)
        return results

    def search_by_title(self, title: str, top_k: int = 50) -> List[Dict[str, Any]]:
        return self.search_movies(title, top_k=top_k)

    def search(self, query_dict: Dict[str, Any], top_k: int = 50) -> List[Dict[str, Any]]:
        title = query_dict.get("primaryTitle") or ""
        return self.search_movies(str(title), top_k=top_k)

    def _db_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _db_fetch_movie_row(self, tconst: str) -> Dict[str, Any]:
        if not tconst:
            return {}
        with self._db_connection() as conn:
            c = conn.cursor()
            c.execute(
                f"""
                SELECT
                    {movie_select_clause(alias='t', genre_alias='g')}
                FROM titles t
                LEFT JOIN title_genres g ON g.tconst = t.tconst
                WHERE t.tconst = ?
                GROUP BY t.tconst
                """,
                (tconst,),
            )
            r = c.fetchone()
        if not r:
            return {"tconst": tconst}
        return map_movie_row(r)

    def _db_fetch_person_row(self, nconst: str) -> Dict[str, Any]:
        if not nconst:
            return {}
        with self._db_connection() as conn:
            c = conn.cursor()
            c.execute(
                f"""
                SELECT
                    {people_select_clause(alias='p', prof_alias='pp')}
                FROM people p
                LEFT JOIN people_professions pp ON pp.nconst = p.nconst
                WHERE p.nconst = ?
                GROUP BY p.nconst
                """,
                (nconst,),
            )
            r = c.fetchone()
        if not r:
            return {"nconst": nconst}
        return map_person_row(r)

    def _db_fetch_movie_people(self, tconst: str) -> Dict[str, List[Dict[str, Any]]]:
        by_head: Dict[str, List[Dict[str, Any]]] = {h: [] for h in self.cfg.hybrid_set_heads}
        if not tconst:
            return by_head
        with self._db_connection() as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT pr.nconst, pr.category, pr.ordering, p.primaryName
                FROM principals pr
                JOIN people p ON p.nconst = pr.nconst
                WHERE pr.tconst = ?
                ORDER BY pr.ordering ASC
                """,
                (tconst,),
            )
            rows = c.fetchall()
        for nconst, category, ordering, name in rows:
            head = self.cfg.hybrid_set_category_to_head.get((category or "").lower().strip())
            if head is None:
                continue
            person_idx = self.nconst_to_index.get(nconst)
            if person_idx is None:
                continue
            by_head.setdefault(head, []).append(
                {
                    "nconst": nconst,
                    "primaryName": name,
                    "category": category,
                    "ordering": ordering,
                    "person_index": person_idx,
                }
            )
        return by_head

    def _db_fetch_person_movies(self, nconst: str) -> Dict[str, List[Dict[str, Any]]]:
        by_head: Dict[str, List[Dict[str, Any]]] = {h: [] for h in self.cfg.hybrid_set_heads}
        if not nconst:
            return by_head
        with self._db_connection() as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT pr.tconst, pr.category, pr.ordering, t.primaryTitle, t.startYear
                FROM principals pr
                JOIN titles t ON t.tconst = pr.tconst
                WHERE pr.nconst = ?
                ORDER BY t.startYear ASC, pr.ordering ASC
                """,
                (nconst,),
            )
            rows = c.fetchall()
        for tconst, category, ordering, title, year in rows:
            head = self.cfg.hybrid_set_category_to_head.get((category or "").lower().strip())
            if head is None:
                continue
            movie_idx = self.tconst_to_index.get(tconst)
            if movie_idx is None:
                continue
            by_head.setdefault(head, []).append(
                {
                    "tconst": tconst,
                    "primaryTitle": title,
                    "startYear": year,
                    "category": category,
                    "ordering": ordering,
                    "movie_index": movie_idx,
                }
            )
        return by_head

    @staticmethod
    def _render_recon_fields(fields, orig_inputs, recon_table):
        recon_fields = []
        recon_counts: Dict[str, int] = {}
        for f, orig_t, rec_field in zip(fields, orig_inputs, recon_table):
            rec_t = rec_field[0].cpu()
            db_render = f.render_ground_truth(orig_t)
            pred_render = f.render_prediction(rec_t)
            recon_fields.append({"name": f.name, "db": db_render, "recon": pred_render})
            if f.name.endswith("Count"):
                try:
                    recon_counts[f.name] = int(pred_render)
                except (TypeError, ValueError):
                    pass
        return recon_fields, recon_counts

    @staticmethod
    def _select_top_k(probs, local_to_global, k_hat, num_items, true_ids, item_render_fn, threshold):
        vocab_size = int(local_to_global.shape[0])
        order = probs.argsort()[::-1]

        def _collect(use_threshold):
            items: List[Dict[str, Any]] = []
            for li in order:
                if li < 0 or li >= vocab_size:
                    continue
                p_val = float(probs[li])
                if use_threshold and p_val < threshold:
                    break
                global_idx = int(local_to_global[li].item())
                if global_idx < 0 or global_idx >= num_items:
                    continue
                item = item_render_fn(global_idx, p_val, true_ids)
                items.append(item)
                if len(items) >= k_hat:
                    break
            return items

        head_list = _collect(use_threshold=True)
        if len(head_list) < k_hat and k_hat > 0:
            head_list = _collect(use_threshold=False)
        return head_list

    def _decode_head_predictions(self, logits_dict, head_l2g, db_relations, id_key,
                                 recon_counts, num_items, item_render_fn, threshold, max_items):
        pred_heads: Dict[str, List[Dict[str, Any]]] = {}
        for head, logits in logits_dict.items():
            probs_t = torch.sigmoid(logits[0])
            probs = probs_t.cpu().numpy()
            local_to_global = head_l2g.get(head)
            if local_to_global is None or local_to_global.numel() == 0:
                continue

            vocab_size = int(local_to_global.shape[0])
            true_ids = {
                item[id_key]
                for item in db_relations.get(head, [])
                if item.get(id_key)
            }
            true_count = len(true_ids)

            pred_count = recon_counts.get(f"{head}Count")
            soft_count = float(probs.sum())

            if true_count > 0:
                k_hat = true_count
            elif pred_count is not None:
                k_hat = max(0, int(pred_count))
            else:
                k_hat = int(round(soft_count))

            k_hat = max(0, min(k_hat, vocab_size))
            if max_items is not None:
                k_hat = min(k_hat, int(max_items))

            if k_hat == 0:
                pred_heads[head] = []
                continue

            pred_heads[head] = self._select_top_k(
                probs, local_to_global, k_hat, num_items, true_ids, item_render_fn, threshold,
            )
        return pred_heads

    def _movie_item_render(self, global_idx, p_val, true_ids):
        nconst = self.person_nconst[global_idx]
        return {
            "person_index": global_idx,
            "primaryName": self.person_names[global_idx],
            "nconst": nconst,
            "prob": p_val,
            "is_true": nconst in true_ids,
        }

    def _person_item_render(self, global_idx, p_val, true_ids):
        tconst = self.movie_tconst[global_idx]
        return {
            "movie_index": global_idx,
            "primaryTitle": self.movie_titles[global_idx],
            "tconst": tconst,
            "prob": p_val,
            "is_true": tconst in true_ids,
        }

    def get_movie_detail(self, movie_index: int, threshold: float = 0.5, max_items: int = 20) -> Dict[str, Any]:
        idx = int(movie_index)
        if idx < 0 or idx >= self.num_movies:
            raise KeyError(f"Movie index out of range: {idx}")

        tconst = self.movie_tconst[idx]
        db_row = self._db_fetch_movie_row(tconst)
        db_people = self._db_fetch_movie_people(tconst)
        orig_inputs = [t[idx].cpu() for t in self.movie_ds.stacked_fields]

        with torch.no_grad():
            idx_t = torch.tensor([idx], device=self.device, dtype=torch.long)
            model_out = self.model(movie_indices=idx_t).get("movie")

        if model_out is None:
            recon_fields, pred_heads = [], {}
        else:
            logits_dict, recon_table, _, _ = model_out
            recon_fields, recon_counts = self._render_recon_fields(
                self.movie_ds.fields, orig_inputs, recon_table,
            )
            pred_heads = self._decode_head_predictions(
                logits_dict, self.movie_ds.head_local_to_global, db_people, "nconst",
                recon_counts, self.num_people, self._movie_item_render, threshold, max_items,
            )

        return {
            "movie_index": idx,
            "tconst": tconst,
            "title": self.movie_titles[idx],
            "db": {"row": db_row, "people_by_head": db_people},
            "reconstructed": {"fields": recon_fields, "people_by_head": pred_heads},
        }

    def get_person_detail(self, person_index: int, threshold: float = 0.5, max_items: int = 20) -> Dict[str, Any]:
        idx = int(person_index)
        if idx < 0 or idx >= self.num_people:
            raise KeyError(f"Person index out of range: {idx}")

        nconst = self.person_nconst[idx]
        db_row = self._db_fetch_person_row(nconst)
        db_movies = self._db_fetch_person_movies(nconst)
        orig_inputs = [t[idx].cpu() for t in self.person_ds.stacked_fields]

        with torch.no_grad():
            idx_t = torch.tensor([idx], device=self.device, dtype=torch.long)
            model_out = self.model(person_indices=idx_t).get("person")

        if model_out is None:
            recon_fields, pred_heads = [], {}
        else:
            logits_dict, recon_table, _, _ = model_out
            recon_fields, recon_counts = self._render_recon_fields(
                self.person_ds.fields, orig_inputs, recon_table,
            )
            pred_heads = self._decode_head_predictions(
                logits_dict, self.person_ds.head_local_to_global, db_movies, "tconst",
                recon_counts, self.num_movies, self._person_item_render, threshold, max_items,
            )

        return {
            "person_index": idx,
            "nconst": nconst,
            "name": self.person_names[idx],
            "db": {"row": db_row, "movies_by_head": db_movies},
            "reconstructed": {"fields": recon_fields, "movies_by_head": pred_heads},
        }



MovieSearchEngine = HybridSearchEngine


if __name__ == "__main__":
    engine = HybridSearchEngine()

    q = "Matrix"
    print(f"\nSearching movies for: {q}")
    hits = engine.search_movies(q, top_k=5)
    for h in hits:
        print(f"[{h['_score']:.4f}] {h['primaryTitle']} ({h.get('startYear', '')})")

    q2 = "Keanu Reeves"
    print(f"\nSearching people for: {q2}")
    hits_p = engine.search_people(q2, top_k=5)
    for h in hits_p:
        print(f"[{h['_score']:.4f}] {h['primaryName']} (b. {h.get('birthYear', '')})")
