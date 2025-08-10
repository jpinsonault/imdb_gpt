from typing import List, Dict, Iterator

from ..fields import (
    NumericDigitCategoryField,
    TextField,
    BaseField,
)
from .base import RowAutoencoder
from . import db as row_db

class TitlesAutoencoder(RowAutoencoder):
    def build_fields(self) -> List[BaseField]:
        return [
            TextField("primaryTitle"),
            NumericDigitCategoryField("startYear"),
        ]

    def row_generator(self) -> Iterator[Dict]:
        return row_db.iter_titles(self.db_path, int(self.config["movie_limit"]))

    def row_by_tconst(self, tconst: str) -> Dict:
        return row_db.get_title_by_tconst(self.db_path, tconst)

class PeopleAutoencoder(RowAutoencoder):
    def build_fields(self) -> List[BaseField]:
        return [
            TextField("primaryName"),
            NumericDigitCategoryField("birthYear"),
        ]

    def row_generator(self) -> Iterator[Dict]:
        return row_db.iter_people(self.db_path)

    def row_by_nconst(self, nconst: str) -> Dict:
        return row_db.get_person_by_nconst(self.db_path, nconst)
