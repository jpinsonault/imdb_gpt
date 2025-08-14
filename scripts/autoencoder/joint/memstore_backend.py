# scripts/autoencoder/joint/memstore_backend.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .dataset import SharedSamplerState

def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    if str(arr.dtype).startswith("int"):
        return torch.from_numpy(np.array(arr, copy=False)).long()
    return torch.from_numpy(np.array(arr, copy=False)).float()

class MemStore:
    def __init__(self, root: str, device: torch.device):
        self.root = Path(root)
        self.device = device

        with open(self.root / "manifest.json", "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        counts = self.manifest["counts"]
        self.num_movies = int(counts["movies"])
        self.num_people = int(counts["people"])
        self.num_edges = int(counts["edges"])

        movies_meta = self.manifest["movies"]
        people_meta = self.manifest["people"]
        edges_meta = self.manifest["edges"]

        movies_dir = self.root / movies_meta.get("dir", "movies")
        people_dir = self.root / people_meta.get("dir", "people")
        edges_dir = self.root / edges_meta.get("dir", "edges") if "dir" in edges_meta else self.root

        self.movie_specs = list(movies_meta["fields"])
        self.person_specs = list(people_meta["fields"])
        self.movie_names = [spec["name"] for spec in self.movie_specs]
        self.person_names = [spec["name"] for spec in self.person_specs]

        self._movie_mm: Dict[str, np.memmap] = {}
        self._person_mm: Dict[str, np.memmap] = {}

        for spec in self.movie_specs:
            name = spec["name"]
            dtype = spec.get("dtype", "float32")
            shape = (self.num_movies,) + tuple(spec["shape"])
            path = spec.get("path", f"{name}.mm")
            path = movies_dir / path
            self._movie_mm[name] = np.memmap(path, dtype=dtype, mode="r", shape=shape)

        for spec in self.person_specs:
            name = spec["name"]
            dtype = spec.get("dtype", "float32")
            shape = (self.num_people,) + tuple(spec["shape"])
            path = spec.get("path", f"{name}.mm")
            path = people_dir / path
            self._person_mm[name] = np.memmap(path, dtype=dtype, mode="r", shape=shape)

        m_idx_path = edges_meta.get("movie_idx", "edges/movie_idx.mm")
        p_idx_path = edges_meta.get("person_idx", "edges/person_idx.mm")
        self.edges_movie_idx = np.memmap(
            edges_dir / Path(m_idx_path).name, dtype=np.int32, mode="r", shape=(self.num_edges,)
        )
        self.edges_person_idx = np.memmap(
            edges_dir / Path(p_idx_path).name, dtype=np.int32, mode="r", shape=(self.num_edges,)
        )

    def gather_movies(self, idxs: np.ndarray) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for spec in self.movie_specs:
            name = spec["name"]
            arr = self._movie_mm[name][idxs]
            out.append(_to_tensor(arr))
        return out

    def gather_people(self, idxs: np.ndarray) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for spec in self.person_specs:
            name = spec["name"]
            arr = self._person_mm[name][idxs]
            out.append(_to_tensor(arr))
        return out

    def get_movie_names(self) -> List[str]:
        return list(self.movie_names)

    def get_person_names(self) -> List[str]:
        return list(self.person_names)

class MemmapEdgeSampler:
    def __init__(
        self,
        store: MemStore,
        movie_ae,
        person_ae,
        batch_size: int,
        boost: float = 0.10,
        shared_state: Optional[SharedSamplerState] = None,
    ):
        self.store = store
        self.bs = int(batch_size)
        self.boost = float(boost)
        self.num_edges = int(store.num_edges)
        if shared_state is None:
            init = np.ones((self.num_edges,), dtype=np.float32) / max(1, self.num_edges)
            self.state = SharedSamplerState(self.num_edges, init)
        else:
            self.state = shared_state

        self._movie_order = None
        self._person_order = None

        try:
            store_m = self.store.get_movie_names()
            want_m = [f.name for f in getattr(movie_ae, "fields", [])]
            m_pos = {n: i for i, n in enumerate(store_m)}
            self._movie_order = [m_pos[n] for n in want_m]
        except Exception as e:
            self._movie_order = None

        try:
            store_p = self.store.get_person_names()
            want_p = [f.name for f in getattr(person_ae, "fields", [])]
            p_pos = {n: i for i, n in enumerate(store_p)}
            self._person_order = [p_pos[n] for n in want_p]
        except Exception as e:
            self._person_order = None

    def sample_indices(self, k: int) -> np.ndarray:
        p, a = self.state.arrays()
        n = int(p.size)
        if n == 0 or k <= 0:
            return np.empty((0,), dtype=np.int64)
        i = np.random.randint(0, n, size=k, dtype=np.int64)
        acc = np.random.random(size=k) < p[i]
        out = i.copy()
        out[~acc] = a[i[~acc]]
        return out.astype(np.int64, copy=False)

    def sample_batch(self):
        idxs = self.sample_indices(self.bs)
        if idxs.size == 0:
            return [], [], torch.zeros((0,), dtype=torch.long)
        mi = self.store.edges_movie_idx[idxs].astype(np.int64, copy=False)
        pi = self.store.edges_person_idx[idxs].astype(np.int64, copy=False)
        M = self.store.gather_movies(mi)
        P = self.store.gather_people(pi)

        if self._movie_order is not None:
            M = [M[i] for i in self._movie_order]
        if self._person_order is not None:
            P = [P[i] for i in self._person_order]

        e = torch.from_numpy(idxs.astype(np.int64, copy=False))
        return M, P, e

def make_memmap_edge_sampler(
    memstore_dir: str,
    movie_ae,
    person_ae,
    batch_size: int,
    boost: float = 0.10,
    shared_state: Optional[SharedSamplerState] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = MemStore(memstore_dir, device)
    return MemmapEdgeSampler(
        store=store,
        movie_ae=movie_ae,
        person_ae=person_ae,
        batch_size=batch_size,
        boost=boost,
        shared_state=shared_state,
    )
