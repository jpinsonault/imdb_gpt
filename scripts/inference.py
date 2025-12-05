# scripts/inference.py

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

from config import project_config
from scripts.simple_set.model import HybridSetModel
from scripts.simple_set.data import HybridSetDataset
from scripts.autoencoder.fields import TextField


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MovieSearchEngine:
    def __init__(self, device: Optional[str] = None):
        self.cfg = project_config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logging.info(f"Initializing Search Engine on {self.device}...")

        cache_path = Path(self.cfg.data_dir) / "hybrid_set_cache.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found at {cache_path}. Run training/precompute first.")

        self.dataset = HybridSetDataset(str(cache_path), self.cfg)
        self.num_movies = len(self.dataset)
        logging.info(f"Loaded dataset with {self.num_movies} movies.")

        self.model = HybridSetModel(
            fields=self.dataset.fields,
            num_people=self.dataset.num_people,
            heads_config=self.cfg.hybrid_set_heads,
            head_vocab_sizes=self.dataset.head_vocab_sizes,
            latent_dim=int(self.cfg.hybrid_set_latent_dim),
            hidden_dim=int(self.cfg.hybrid_set_hidden_dim),
            base_output_rank=int(self.cfg.hybrid_set_output_rank),
            depth=int(self.cfg.hybrid_set_depth),
            dropout=0.0,
            num_movies=self.num_movies,
            title_noise_prob=float(self.cfg.hybrid_set_w_title > 0.0),
            head_prior_prob=float(self.cfg.hybrid_set_head_prior),
        )

        ckpt_path = Path(self.cfg.model_dir) / "hybrid_set_state.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        logging.info(f"Loading weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            emb = self.model.movie_embeddings.weight.detach().to(self.device)
            self.embedding_table = F.normalize(emb, p=2, dim=-1)

        self.title_field = None
        self.title_field_index = None
        for idx, f in enumerate(self.dataset.fields):
            if isinstance(f, TextField) and f.name == "primaryTitle":
                self.title_field = f
                self.title_field_index = idx
                break

        if self.title_field is None or self.title_field_index is None:
            raise RuntimeError("No TextField named 'primaryTitle' found in dataset fields.")

        logging.info("Search Engine Ready.")

    @torch.no_grad()
    def encode_title(self, title: str) -> torch.Tensor:
        if not title:
            raise ValueError("Title must be a non-empty string.")

        tokens = self.title_field.transform(title)
        tokens = tokens.unsqueeze(0).to(self.device)

        z = self.model.title_encoder(tokens)
        z = F.normalize(z, p=2, dim=-1)
        return z

    @torch.no_grad()
    def _decode_result_row(self, idx: int, dist: float, rank: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "_score": float(dist),
            "_rank": int(rank),
            "_id": int(idx),
            "titleType": "movie",
        }

        for f_idx, field in enumerate(self.dataset.fields):
            val_tensor = self.dataset.stacked_fields[f_idx][idx]
            row[field.name] = field.render_ground_truth(val_tensor)

        return row

    @torch.no_grad()
    def search_by_title(self, title: str, top_k: int = 50) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()

        title = (title or "").strip()
        if not title:
            return []

        z_query = self.encode_title(title)
        dists = torch.cdist(z_query, self.embedding_table, p=2).squeeze(0)

        k = int(top_k)
        k = max(1, min(k, dists.numel()))
        values, indices = torch.topk(dists, k=k, largest=False)

        indices_cpu = indices.cpu().numpy()
        distances_cpu = values.cpu().numpy()

        results: List[Dict[str, Any]] = []
        for rank, (idx, dist) in enumerate(zip(indices_cpu, distances_cpu), start=1):
            results.append(self._decode_result_row(int(idx), float(dist), rank))

        duration_ms = (time.perf_counter() - t0) * 1000.0
        logging.info(f"Search for '{title}' took {duration_ms:.2f}ms")

        return results

    @torch.no_grad()
    def search(self, query_dict: Dict[str, Any], top_k: int = 50) -> List[Dict[str, Any]]:
        title = query_dict.get("primaryTitle") or ""
        return self.search_by_title(str(title), top_k=top_k)


if __name__ == "__main__":
    engine = MovieSearchEngine()

    q = "Matrix"
    print(f"\nSearching for: {q}")
    hits = engine.search_by_title(q, top_k=5)

    for h in hits:
        print(f"[{h['_score']:.4f}] {h['primaryTitle']} ({h['startYear']})")
