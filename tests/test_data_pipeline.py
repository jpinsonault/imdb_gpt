import sqlite3

from scripts.create_normalized_tables import create_normalized_schema
from scripts.sql_filters import map_movie_row, map_person_row


EXPECTED_TABLES = [
    "titles",
    "title_genres",
    "episodes",
    "people",
    "people_professions",
    "people_known_for",
    "crew",
    "principals",
    "principal_characters",
]


def test_normalized_schema_tables_exist(tmp_path):
    db_path = tmp_path / "schema.db"
    conn = sqlite3.connect(db_path)
    create_normalized_schema(conn)

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = sorted(row[0] for row in cursor.fetchall())
    conn.close()

    for t in EXPECTED_TABLES:
        assert t in tables, f"Missing table: {t}"
    assert len(tables) == len(EXPECTED_TABLES)


def test_titles_row_count(fixture_db):
    conn = sqlite3.connect(fixture_db)
    count = conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
    conn.close()
    assert count == 3


def test_title_genres_one_to_many(fixture_db):
    conn = sqlite3.connect(fixture_db)
    count = conn.execute("SELECT COUNT(*) FROM title_genres").fetchone()[0]
    conn.close()
    assert count == 5  # 2 + 1 + 2


def test_people_row_count(fixture_db):
    conn = sqlite3.connect(fixture_db)
    count = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    conn.close()
    assert count == 4


def test_principals_row_count(fixture_db):
    conn = sqlite3.connect(fixture_db)
    count = conn.execute("SELECT COUNT(*) FROM principals").fetchone()[0]
    conn.close()
    assert count == 6


def test_map_movie_row_structure():
    row = ("tt0000001", "Jaws", 1975, None, 124, 8.1, 100000, "Adventure,Thriller", 5)
    result = map_movie_row(row)

    assert set(result.keys()) == {
        "tconst", "primaryTitle", "startYear", "endYear",
        "runtimeMinutes", "averageRating", "numVotes", "genres", "peopleCount",
    }
    assert result["tconst"] == "tt0000001"
    assert result["primaryTitle"] == "Jaws"
    assert isinstance(result["genres"], list)
    assert result["genres"] == ["Adventure", "Thriller"]
    assert result["peopleCount"] == 5


def test_map_person_row_structure():
    row = ("Steven Spielberg", 1946, None, "director,producer", 50, "nm0000001")
    result = map_person_row(row)

    assert set(result.keys()) == {
        "primaryName", "birthYear", "deathYear", "professions", "titleCount", "nconst",
    }
    assert result["primaryName"] == "Steven Spielberg"
    assert isinstance(result["professions"], list)
    assert result["professions"] == ["director", "producer"]
    assert result["nconst"] == "nm0000001"
