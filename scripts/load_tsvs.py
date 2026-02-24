import csv
import os
import simplejson
from pathlib import Path
import sqlite3
from tqdm import tqdm
from config import project_config

BATCH_SIZE = 50_000


def _estimate_line_count(file_path):
    """Estimate line count from file size (~80 bytes/line avg for IMDB TSVs)."""
    size = os.path.getsize(file_path)
    return size // 80


def _open_conn(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = -64000")
    return conn


def _bulk_load(db_path, file_path, table_name, create_sql, columns, desc, row_transform=None):
    """
    Generic batch loader: reads a TSV, converts \\N→None, optionally transforms
    rows, and bulk-inserts with executemany.
    """
    conn = _open_conn(db_path)
    cursor = conn.cursor()

    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    cursor.execute(create_sql)

    placeholders = ",".join(["?"] * len(columns))
    insert_sql = f"INSERT OR IGNORE INTO {table_name} ({','.join(columns)}) VALUES ({placeholders});"

    batch = []
    est_total = _estimate_line_count(file_path)

    with open(file_path, mode='rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc=desc, total=est_total, unit='rows'):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            if row_transform:
                row_transform(row)

            batch.append(tuple(row[c] for c in columns))

            if len(batch) >= BATCH_SIZE:
                cursor.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()

    if batch:
        cursor.executemany(insert_sql, batch)
        conn.commit()

    conn.close()


def load_raw_title_basics(db_path, file_path):
    _bulk_load(
        db_path, file_path,
        table_name="raw_title_basics",
        create_sql="""
            CREATE TABLE raw_title_basics (
                tconst TEXT PRIMARY KEY,
                titleType TEXT,
                primaryTitle TEXT,
                originalTitle TEXT,
                isAdult INTEGER,
                startYear INTEGER,
                endYear INTEGER,
                runtimeMinutes INTEGER,
                genres TEXT
            );
        """,
        columns=["tconst", "titleType", "primaryTitle", "originalTitle",
                 "isAdult", "startYear", "endYear", "runtimeMinutes", "genres"],
        desc="Loading raw_title_basics",
    )


def load_raw_title_crew(db_path, file_path):
    _bulk_load(
        db_path, file_path,
        table_name="raw_title_crew",
        create_sql="""
            CREATE TABLE raw_title_crew (
                tconst TEXT PRIMARY KEY,
                directors TEXT,
                writers TEXT
            );
        """,
        columns=["tconst", "directors", "writers"],
        desc="Loading raw_title_crew",
    )


def load_raw_title_episode(db_path, file_path):
    _bulk_load(
        db_path, file_path,
        table_name="raw_title_episode",
        create_sql="""
            CREATE TABLE raw_title_episode (
                tconst TEXT PRIMARY KEY,
                parentTconst TEXT,
                seasonNumber INTEGER,
                episodeNumber INTEGER
            );
        """,
        columns=["tconst", "parentTconst", "seasonNumber", "episodeNumber"],
        desc="Loading raw_title_episode",
    )


def _validate_principals_json(row):
    """Null out invalid JSON in the characters column."""
    if row['characters']:
        try:
            simplejson.loads(row['characters'])
        except (ValueError, simplejson.JSONDecodeError):
            row['characters'] = None


def load_raw_title_principals(db_path, file_path):
    _bulk_load(
        db_path, file_path,
        table_name="raw_title_principals",
        create_sql="""
            CREATE TABLE raw_title_principals (
                tconst TEXT,
                ordering INTEGER,
                nconst TEXT,
                category TEXT,
                job TEXT,
                characters TEXT,
                PRIMARY KEY(tconst, ordering)
            );
        """,
        columns=["tconst", "ordering", "nconst", "category", "job", "characters"],
        desc="Loading raw_title_principals",
        row_transform=_validate_principals_json,
    )


def load_raw_title_ratings(db_path, file_path):
    _bulk_load(
        db_path, file_path,
        table_name="raw_title_ratings",
        create_sql="""
            CREATE TABLE raw_title_ratings (
                tconst TEXT PRIMARY KEY,
                averageRating REAL,
                numVotes INTEGER
            );
        """,
        columns=["tconst", "averageRating", "numVotes"],
        desc="Loading raw_title_ratings",
    )


def load_raw_name_basics(db_path, file_path):
    _bulk_load(
        db_path, file_path,
        table_name="raw_name_basics",
        create_sql="""
            CREATE TABLE raw_name_basics (
                nconst TEXT PRIMARY KEY,
                primaryName TEXT,
                birthYear INTEGER,
                deathYear INTEGER,
                primaryProfession TEXT,
                knownForTitles TEXT
            );
        """,
        columns=["nconst", "primaryName", "birthYear", "deathYear",
                 "primaryProfession", "knownForTitles"],
        desc="Loading raw_name_basics",
    )


def load_raw_title_akas(db_path, file_path):
    _bulk_load(
        db_path, file_path,
        table_name="raw_title_akas",
        create_sql="""
            CREATE TABLE raw_title_akas (
                titleId TEXT,
                ordering INTEGER,
                title TEXT,
                region TEXT,
                language TEXT,
                types TEXT,
                attributes TEXT,
                isOriginalTitle INTEGER,
                PRIMARY KEY (titleId, ordering)
            );
        """,
        columns=["titleId", "ordering", "title", "region", "language",
                 "types", "attributes", "isOriginalTitle"],
        desc="Loading raw_title_akas",
    )


def load_db_files(db_path, file_paths):
    """
    Create new raw_* tables and load IMDb data files into them.
    """
    print(f"Loading raw data into: {db_path}")

    if file_paths.get('title_basics'):
        load_raw_title_basics(db_path, file_paths['title_basics'])

    if file_paths.get('title_crew'):
        load_raw_title_crew(db_path, file_paths['title_crew'])

    if file_paths.get('title_episode'):
        load_raw_title_episode(db_path, file_paths['title_episode'])

    if file_paths.get('title_principals'):
        load_raw_title_principals(db_path, file_paths['title_principals'])

    if file_paths.get('title_ratings'):
        load_raw_title_ratings(db_path, file_paths['title_ratings'])

    if file_paths.get('name_basics'):
        load_raw_name_basics(db_path, file_paths['name_basics'])

    # If you still want to load akas, uncomment:
    # if file_paths.get('akas'):
    #     load_raw_title_akas(db_path, file_paths['akas'])

    print("Raw IMDb tables created and data loaded successfully.")


if __name__ == '__main__':
    # Use raw_db_path for loading raw data
    db_path = Path(project_config.raw_db_path)
    data_dir = Path(project_config.data_dir)
    tsv_dir = data_dir / 'imdb_tsvs'

    # Ensure parent dir exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    file_paths = {
        'title_basics'    : tsv_dir / 'title.basics.tsv',
        'title_crew'      : tsv_dir / 'title.crew.tsv',
        'title_episode'   : tsv_dir / 'title.episode.tsv',
        'title_principals': tsv_dir / 'title.principals.tsv',
        'title_ratings'   : tsv_dir / 'title.ratings.tsv',
        'name_basics'     : tsv_dir / 'name.basics.tsv',
    }

    load_db_files(db_path, file_paths)
