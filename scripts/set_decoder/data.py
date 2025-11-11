import sqlite3
from typing import Dict, List, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder


class _LRUCache:
    def __init__(self, capacity: int):
        self.capacity = int(max(0, capacity))
        self.data = OrderedDict()

    def get(self, key):
        if self.capacity <= 0:
            return None
        if key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


class TitlePeopleSetDataset(Dataset):
    def __init__(
        self,
        db_path: str,
        movie_ae: TitlesAutoencoder,
        people_ae: PeopleAutoencoder,
        num_slots: int,
        movie_limit: int | None = None,
    ):
        super().__init__()
        self.db_path = db_path
        self.movie_ae = movie_ae
        self.people_ae = people_ae
        self.num_slots = int(num_slots)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cur = self.conn.cursor()

        self.movie_sql = """
        SELECT primaryTitle,startYear,endYear,runtimeMinutes,
               averageRating,numVotes,
               (SELECT GROUP_CONCAT(genre,',') FROM title_genres WHERE tconst = ?)
        FROM titles WHERE tconst = ? LIMIT 1
        """

        self.person_sql = """
        SELECT primaryName,birthYear,deathYear,
               (SELECT GROUP_CONCAT(profession,',') FROM people_professions WHERE nconst = ?)
        FROM people WHERE nconst = ? LIMIT 1
        """

        self.movies = self._load_movies(movie_limit)

        self.movie_cache: Dict[str, Dict] = {}
        self.person_cache: Dict[str, Dict] = {}

        people_lru_cap = min(len(self.movies), 200000) if self.movies else 0
        movie_latent_cap = min(len(self.movies), 200000) if self.movies else 0
        person_latent_cap = min(len(self.movies) * self.num_slots, 500000) if self.movies else 0

        self.people_for_movie_cache = _LRUCache(people_lru_cap)
        self.movie_latent_cache = _LRUCache(movie_latent_cap)
        self.person_latent_cache = _LRUCache(person_latent_cap)

    def _load_movies(self, movie_limit: int | None) -> List[str]:
        if movie_limit is not None and movie_limit > 0:
            rows = self.cur.execute(
                """
                SELECT tconst
                FROM edges
                GROUP BY tconst
                HAVING COUNT(*) > 0
                ORDER BY tconst
                LIMIT ?
                """,
                (int(movie_limit),),
            ).fetchall()
        else:
            rows = self.cur.execute(
                """
                SELECT tconst
                FROM edges
                GROUP BY tconst
                HAVING COUNT(*) > 0
                ORDER BY tconst
                """,
            ).fetchall()
        return [str(r[0]) for r in rows]

    def __len__(self):
        return len(self.movies)

    def _movie_row(self, tconst: str) -> Dict:
        if tconst in self.movie_cache:
            return self.movie_cache[tconst]
        r = self.cur.execute(self.movie_sql, (tconst, tconst)).fetchone()
        if r is None:
            row = {
                "tconst": tconst,
                "primaryTitle": None,
                "startYear": None,
                "endYear": None,
                "runtimeMinutes": None,
                "averageRating": None,
                "numVotes": None,
                "genres": [],
            }
        else:
            row = {
                "tconst": tconst,
                "primaryTitle": r[0],
                "startYear": r[1],
                "endYear": r[2],
                "runtimeMinutes": r[3],
                "averageRating": r[4],
                "numVotes": r[5],
                "genres": r[6].split(",") if r[6] else [],
            }
        self.movie_cache[tconst] = row
        return row

    def _person_row(self, nconst: str) -> Dict:
        if nconst in self.person_cache:
            return self.person_cache[nconst]
        r = self.cur.execute(self.person_sql, (nconst, nconst)).fetchone()
        if r is None:
            row = {
                "primaryName": None,
                "birthYear": None,
                "deathYear": None,
                "professions": None,
            }
        else:
            row = {
                "primaryName": r[0],
                "birthYear": r[1],
                "deathYear": r[2],
                "professions": r[3].split(",") if r[3] else None,
            }
        self.person_cache[nconst] = row
        return row

    def _people_for_movie(self, tconst: str) -> List[str]:
        cached = self.people_for_movie_cache.get(tconst)
        if cached is not None:
            return cached
        rows = self.cur.execute(
            """
            SELECT DISTINCT nconst
            FROM edges
            WHERE tconst = ?
            """,
            (tconst,),
        ).fetchall()
        people = [str(r[0]) for r in rows]
        self.people_for_movie_cache.put(tconst, people)
        return people

    def _encode_movie_latent(self, tconst: str) -> torch.Tensor:
        cached = self.movie_latent_cache.get(tconst)
        if cached is not None:
            return cached
        row = self._movie_row(tconst)
        xs = []
        for f in self.movie_ae.fields:
            x = f.transform(row.get(f.name))
            xs.append(x.unsqueeze(0))
        xs = [x.cpu() for x in xs]
        with torch.no_grad():
            z = self.movie_ae.encoder(xs)
        z = z.squeeze(0).cpu()
        self.movie_latent_cache.put(tconst, z)
        return z

    def _encode_person_latent_and_targets(
        self,
        nconst: str,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        cached = self.person_latent_cache.get(nconst)
        if cached is not None:
            return cached
        row = self._person_row(nconst)
        xs = []
        ys = []
        for f in self.people_ae.fields:
            x = f.transform(row.get(f.name))
            y = f.transform_target(row.get(f.name))
            xs.append(x.unsqueeze(0))
            ys.append(y)
        xs = [x.cpu() for x in xs]
        with torch.no_grad():
            z = self.people_ae.encoder(xs)
        z = z.squeeze(0).cpu()
        ys = [y.cpu() for y in ys]
        out = (z, ys)
        self.person_latent_cache.put(nconst, out)
        return out

    def __getitem__(self, idx: int):
        tconst = self.movies[int(idx)]

        z_movie = self._encode_movie_latent(tconst)

        nconsts = self._people_for_movie(tconst)
        nconsts = nconsts[: self.num_slots]
        k = len(nconsts)

        latent_dim = z_movie.size(-1)
        Z_gt = torch.zeros(self.num_slots, latent_dim, dtype=torch.float32)
        mask = torch.zeros(self.num_slots, dtype=torch.float32)

        num_fields = len(self.people_ae.fields)
        Y_fields: List[torch.Tensor] = []
        for f in self.people_ae.fields:
            base = f.get_base_padding_value()
            if not isinstance(base, torch.Tensor):
                base = torch.tensor(base)
            if base.dtype == torch.long:
                dtype = torch.long
            else:
                dtype = torch.float32
            base = base.to(dtype=dtype)
            Y_fields.append(torch.stack([base.clone() for _ in range(self.num_slots)], dim=0))

        for j, nconst in enumerate(nconsts):
            z_p, ys = self._encode_person_latent_and_targets(nconst)
            Z_gt[j] = z_p
            mask[j] = 1.0
            for fi in range(num_fields):
                Y_fields[fi][j] = ys[fi]

        return z_movie, Z_gt, mask, Y_fields


def collate_set_decoder(batch):
    z_movies, Z_gts, masks, Y_fields_list = zip(*batch)

    z_movies = torch.stack(z_movies, dim=0)
    Z_gts = torch.stack(Z_gts, dim=0)
    masks = torch.stack(masks, dim=0)

    num_fields = len(Y_fields_list[0])
    Y_batch: List[torch.Tensor] = []
    for fi in range(num_fields):
        ys = [sample[fi] for sample in Y_fields_list]
        Y_batch.append(torch.stack(ys, dim=0))

    return z_movies, Z_gts, masks, Y_batch
