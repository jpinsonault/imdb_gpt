import csv
import simplejson
from pathlib import Path
import sqlite3
from tqdm import tqdm
from config import project_config


def get_file_line_count(file_path):
    with open(file_path, mode='rt', encoding='utf-8') as file:
        return sum(1 for line in file)


def load_raw_title_basics(db_path, file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop and recreate as raw
    cursor.execute("DROP TABLE IF EXISTS raw_title_basics;")
    cursor.execute("""
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
    """)

    with open(file_path, mode='rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc='Loading raw_title_basics', total=get_file_line_count(file_path)):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute("""
                INSERT OR IGNORE INTO raw_title_basics (
                    tconst, titleType, primaryTitle, originalTitle,
                    isAdult, startYear, endYear, runtimeMinutes, genres
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                row['tconst'],
                row['titleType'],
                row['primaryTitle'],
                row['originalTitle'],
                row['isAdult'],
                row['startYear'],
                row['endYear'],
                row['runtimeMinutes'],
                row['genres']
            ))

    conn.commit()
    conn.close()


def load_raw_title_crew(db_path, file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS raw_title_crew;")
    cursor.execute("""
        CREATE TABLE raw_title_crew (
            tconst TEXT PRIMARY KEY,
            directors TEXT,
            writers TEXT
        );
    """)

    with open(file_path, mode='rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc='Loading raw_title_crew', total=get_file_line_count(file_path)):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute("""
                INSERT OR IGNORE INTO raw_title_crew (
                    tconst, directors, writers
                ) VALUES (?, ?, ?);
            """, (
                row['tconst'],
                row['directors'],
                row['writers']
            ))

    conn.commit()
    conn.close()


def load_raw_title_episode(db_path, file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS raw_title_episode;")
    cursor.execute("""
        CREATE TABLE raw_title_episode (
            tconst TEXT PRIMARY KEY,
            parentTconst TEXT,
            seasonNumber INTEGER,
            episodeNumber INTEGER
        );
    """)

    with open(file_path, mode='rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc='Loading raw_title_episode', total=get_file_line_count(file_path)):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute("""
                INSERT OR IGNORE INTO raw_title_episode (
                    tconst, parentTconst, seasonNumber, episodeNumber
                ) VALUES (?, ?, ?, ?);
            """, (
                row['tconst'],
                row['parentTconst'],
                row['seasonNumber'],
                row['episodeNumber']
            ))

    conn.commit()
    conn.close()


def load_raw_title_principals(db_path, file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS raw_title_principals;")
    cursor.execute("""
        CREATE TABLE raw_title_principals (
            tconst TEXT,
            ordering INTEGER,
            nconst TEXT,
            category TEXT,
            job TEXT,
            characters TEXT,
            PRIMARY KEY(tconst, ordering)
        );
    """)

    with open(file_path, mode='rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc='Loading raw_title_principals', total=get_file_line_count(file_path)):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            # Some 'characters' can be JSON arrays
            if row['characters']:
                try:
                    _test = simplejson.loads(row['characters'])  # just to ensure it's valid JSON
                except (ValueError, simplejson.JSONDecodeError):
                    row['characters'] = None

            cursor.execute("""
                INSERT OR IGNORE INTO raw_title_principals (
                    tconst, ordering, nconst, category, job, characters
                ) VALUES (?, ?, ?, ?, ?, ?);
            """, (
                row['tconst'],
                row['ordering'],
                row['nconst'],
                row['category'],
                row['job'],
                row['characters']
            ))

    conn.commit()
    conn.close()


def load_raw_title_ratings(db_path, file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS raw_title_ratings;")
    cursor.execute("""
        CREATE TABLE raw_title_ratings (
            tconst TEXT PRIMARY KEY,
            averageRating REAL,
            numVotes INTEGER
        );
    """)

    with open(file_path, mode='rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc='Loading raw_title_ratings', total=get_file_line_count(file_path)):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute("""
                INSERT OR IGNORE INTO raw_title_ratings (
                    tconst, averageRating, numVotes
                ) VALUES (?, ?, ?);
            """, (
                row['tconst'],
                row['averageRating'],
                row['numVotes']
            ))

    conn.commit()
    conn.close()


def load_raw_name_basics(db_path, file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS raw_name_basics;")
    cursor.execute("""
        CREATE TABLE raw_name_basics (
            nconst TEXT PRIMARY KEY,
            primaryName TEXT,
            birthYear INTEGER,
            deathYear INTEGER,
            primaryProfession TEXT,
            knownForTitles TEXT
        );
    """)

    with open(file_path, mode='rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc='Loading raw_name_basics', total=get_file_line_count(file_path)):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute("""
                INSERT OR IGNORE INTO raw_name_basics (
                    nconst, primaryName, birthYear, deathYear,
                    primaryProfession, knownForTitles
                ) VALUES (?, ?, ?, ?, ?, ?);
            """, (
                row['nconst'],
                row['primaryName'],
                row['birthYear'],
                row['deathYear'],
                row['primaryProfession'],
                row['knownForTitles']
            ))

    conn.commit()
    conn.close()


def load_raw_title_akas(db_path, file_path):
    """
    If you decide you really don't care about akas, remove or ignore this function.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS raw_title_akas;")
    cursor.execute("""
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
    """)

    with open(file_path, mode='rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc='Loading raw_title_akas', total=get_file_line_count(file_path)):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute("""
                INSERT OR IGNORE INTO raw_title_akas (
                    titleId, ordering, title, region, language, types, attributes, isOriginalTitle
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                row['titleId'],
                row['ordering'],
                row['title'],
                row['region'],
                row['language'],
                row['types'],
                row['attributes'],
                row['isOriginalTitle']
            ))

    conn.commit()
    conn.close()


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