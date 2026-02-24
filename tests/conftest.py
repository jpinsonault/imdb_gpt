import sqlite3
import pytest

from scripts.create_normalized_tables import create_normalized_schema


@pytest.fixture
def fixture_db(tmp_path):
    """Create a temp SQLite DB with normalized schema and sample data."""
    db_path = tmp_path / "test_imdb.db"
    conn = sqlite3.connect(db_path)
    create_normalized_schema(conn)

    # 3 movies
    conn.executemany(
        "INSERT INTO titles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("tt0000001", "movie", "Jaws", "Jaws", 0, 1975, None, 124, 8.1, 100000),
            ("tt0000002", "movie", "Star Wars", "Star Wars", 0, 1977, None, 121, 8.6, 200000),
            ("tt0000003", "movie", "Alien", "Alien", 0, 1979, None, 117, 8.5, 150000),
        ],
    )

    # 5 genre rows (2 + 1 + 2)
    conn.executemany(
        "INSERT INTO title_genres VALUES (?, ?)",
        [
            ("tt0000001", "Adventure"),
            ("tt0000001", "Thriller"),
            ("tt0000002", "Sci-Fi"),
            ("tt0000003", "Horror"),
            ("tt0000003", "Sci-Fi"),
        ],
    )

    # 4 people
    conn.executemany(
        "INSERT INTO people VALUES (?, ?, ?, ?)",
        [
            ("nm0000001", "Steven Spielberg", 1946, None),
            ("nm0000002", "George Lucas", 1944, None),
            ("nm0000003", "Roy Scheider", 1932, 2008),
            ("nm0000004", "Sigourney Weaver", 1949, None),
        ],
    )

    # 6 profession rows
    conn.executemany(
        "INSERT INTO people_professions VALUES (?, ?)",
        [
            ("nm0000001", "director"),
            ("nm0000001", "producer"),
            ("nm0000002", "director"),
            ("nm0000002", "writer"),
            ("nm0000003", "actor"),
            ("nm0000004", "actress"),
        ],
    )

    # 6 principal rows
    conn.executemany(
        "INSERT INTO principals VALUES (?, ?, ?, ?, ?)",
        [
            ("tt0000001", 1, "nm0000001", "director", None),
            ("tt0000001", 2, "nm0000003", "actor", None),
            ("tt0000002", 1, "nm0000002", "director", None),
            ("tt0000002", 2, "nm0000004", "actress", None),
            ("tt0000003", 1, "nm0000004", "actress", None),
            ("tt0000003", 2, "nm0000001", "director", None),
        ],
    )

    conn.commit()
    conn.close()
    return db_path
