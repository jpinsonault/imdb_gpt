import sqlite3
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder


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

        self.movie_to_people: Dict[str, List[str]] = {}
        self._build_index(movie_limit=movie_limit)

        self.movie_cache: Dict[str, Dict] = {}
        self.person_cache: Dict[str, Dict] = {}

    def _build_index(self, movie_limit: int | None):
        q = "SELECT tconst, nconst FROM edges ORDER BY tconst"
        rows = self.cur.execute(q)

        last_t = None
        people: List[str] = []
        titles: List[str] = []

        for tconst, nconst in rows:
            tconst = str(tconst)
            nconst = str(nconst)
            if tconst != last_t and last_t is not None:
                unique = []
                seen = set()
                for n in people:
                    if n not in seen:
                        seen.add(n)
                        unique.append(n)
                if unique:
                    self.movie_to_people[last_t] = unique
                    titles.append(last_t)
                    if movie_limit is not None and len(titles) >= movie_limit:
                        break
                people = []
            last_t = tconst
            people.append(nconst)

        if last_t is not None and people:
            unique = []
            seen = set()
            for n in people:
                if n not in seen:
                    seen.add(n)
                    unique.append(n)
            if unique:
                self.movie_to_people[last_t] = unique

        self.movies = [t for t in self.movie_to_people.keys() if self.movie_to_people[t]]

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

    def _encode_movie_latent(self, tconst: str) -> torch.Tensor:
        row = self._movie_row(tconst)
        xs = []
        for f in self.movie_ae.fields:
            x = f.transform(row.get(f.name))
            xs.append(x.unsqueeze(0))
        xs = [x.cpu() for x in xs]
        with torch.no_grad():
            z = self.movie_ae.encoder(xs)
        return z.squeeze(0).cpu()

    def _encode_person_latent_and_targets(
        self,
        nconst: str,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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
        return z, ys

    def __getitem__(self, idx: int):
        tconst = self.movies[int(idx)]

        z_movie = self._encode_movie_latent(tconst)

        nconsts = self.movie_to_people[tconst][: self.num_slots]
        k = len(nconsts)

        latent_dim = z_movie.size(-1)
        Z_gt = torch.zeros(self.num_slots, latent_dim, dtype=torch.float32)

        mask = torch.zeros(self.num_slots, dtype=torch.float32)

        num_fields = len(self.people_ae.fields)
        Y_fields: List[torch.Tensor] = []
        for f in self.people_ae.fields:
            base = f.get_base_padding_value()
            base = base if isinstance(base, torch.Tensor) else torch.tensor(base)
            base = base.to(dtype=torch.long if base.dtype == torch.long else torch.float32)
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
