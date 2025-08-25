import sqlite3
import simplejson
from tqdm import tqdm
from pathlib import Path
from config import project_config


def create_normalized_schema(conn):
    # Drop existing, if any
    conn.execute("DROP TABLE IF EXISTS titles;")
    conn.execute("DROP TABLE IF EXISTS title_genres;")
    conn.execute("DROP TABLE IF EXISTS episodes;")
    conn.execute("DROP TABLE IF EXISTS people;")
    conn.execute("DROP TABLE IF EXISTS people_professions;")
    conn.execute("DROP TABLE IF EXISTS people_known_for;")
    conn.execute("DROP TABLE IF EXISTS crew;")
    conn.execute("DROP TABLE IF EXISTS principals;")
    conn.execute("DROP TABLE IF EXISTS principal_characters;")
    # No 'ratings' table anymore

    # Create new schema with rating columns folded into titles and episodes
    conn.execute("""
    CREATE TABLE titles (
        tconst TEXT PRIMARY KEY,
        titleType TEXT,
        primaryTitle TEXT,
        originalTitle TEXT,
        isAdult INTEGER,
        startYear INTEGER,
        endYear INTEGER,
        runtimeMinutes INTEGER,
        averageRating REAL,
        numVotes INTEGER
    );
    """)

    conn.execute("""
    CREATE TABLE title_genres (
        tconst TEXT,
        genre TEXT,
        PRIMARY KEY (tconst, genre)
    );
    """)

    conn.execute("""
    CREATE TABLE episodes (
        tconst TEXT PRIMARY KEY,
        parentTconst TEXT,
        seasonNumber INTEGER,
        episodeNumber INTEGER,
        averageRating REAL,
        numVotes INTEGER
    );
    """)

    conn.execute("""
    CREATE TABLE people (
        nconst TEXT PRIMARY KEY,
        primaryName TEXT,
        birthYear INTEGER,
        deathYear INTEGER
    );
    """)

    conn.execute("""
    CREATE TABLE people_professions (
        nconst TEXT,
        profession TEXT,
        PRIMARY KEY(nconst, profession)
    );
    """)

    conn.execute("""
    CREATE TABLE people_known_for (
        nconst TEXT,
        tconst TEXT,
        PRIMARY KEY(nconst, tconst)
    );
    """)

    conn.execute("""
    CREATE TABLE crew (
        tconst TEXT,
        nconst TEXT,
        role TEXT CHECK(role IN ('director','writer')),
        PRIMARY KEY (tconst, nconst, role)
    );
    """)

    conn.execute("""
    CREATE TABLE principals (
        tconst TEXT,
        ordering INTEGER,
        nconst TEXT,
        category TEXT,
        job TEXT,
        PRIMARY KEY(tconst, ordering)
    );
    """)

    conn.execute("""
    CREATE TABLE principal_characters (
        tconst TEXT,
        ordering INTEGER,
        nconst TEXT,
        character TEXT,
        PRIMARY KEY(tconst, ordering, character)
    );
    """)

    conn.commit()


def populate_titles_and_genres(conn):
    # Insert into titles with ratings included
    # We do a LEFT JOIN on raw_title_ratings so that titles with no rating remain valid
    insert_titles = """
    INSERT INTO titles (
        tconst,
        titleType,
        primaryTitle,
        originalTitle,
        isAdult,
        startYear,
        endYear,
        runtimeMinutes,
        averageRating,
        numVotes
    )
    SELECT
        b.tconst,
        b.titleType,
        b.primaryTitle,
        b.originalTitle,
        b.isAdult,
        b.startYear,
        b.endYear,
        b.runtimeMinutes,
        r.averageRating,
        r.numVotes
    FROM raw_title_basics b
    LEFT JOIN raw_title_ratings r ON b.tconst = r.tconst;
    """
    conn.execute(insert_titles)

    # Now populate title_genres
    cursor = conn.cursor()
    cursor.execute("SELECT tconst, genres FROM raw_title_basics WHERE genres IS NOT NULL;")
    rows = cursor.fetchall()
    for (tconst, genres_str) in tqdm(rows, desc="Populating title_genres"):
        genres = genres_str.split(',')
        for g in genres:
            g = g.strip()
            if g:
                cursor.execute("""
                    INSERT OR IGNORE INTO title_genres (tconst, genre)
                    VALUES (?, ?);
                """, (tconst, g))

    conn.commit()


def populate_episodes(conn):
    # Insert into episodes, also folding in ratings if they exist
    insert_episodes = """
    INSERT INTO episodes (
        tconst,
        parentTconst,
        seasonNumber,
        episodeNumber,
        averageRating,
        numVotes
    )
    SELECT
        e.tconst,
        e.parentTconst,
        e.seasonNumber,
        e.episodeNumber,
        r.averageRating,
        r.numVotes
    FROM raw_title_episode e
    LEFT JOIN raw_title_ratings r ON e.tconst = r.tconst;
    """
    conn.execute(insert_episodes)
    conn.commit()


def populate_people(conn):
    # Insert into people
    insert_people = """
    INSERT INTO people (nconst, primaryName, birthYear, deathYear)
    SELECT nconst, primaryName, birthYear, deathYear
    FROM raw_name_basics;
    """
    conn.execute(insert_people)

    # Split primaryProfession => people_professions
    cursor = conn.cursor()
    cursor.execute("SELECT nconst, primaryProfession FROM raw_name_basics WHERE primaryProfession IS NOT NULL;")
    rows = cursor.fetchall()
    for (nconst, prof_string) in tqdm(rows, desc="Populating people_professions"):
        profs = prof_string.split(',')
        for p in profs:
            p = p.strip()
            if p:
                cursor.execute("""
                    INSERT OR IGNORE INTO people_professions (nconst, profession)
                    VALUES (?, ?);
                """, (nconst, p))

    # Split knownForTitles => people_known_for
    cursor.execute("SELECT nconst, knownForTitles FROM raw_name_basics WHERE knownForTitles IS NOT NULL;")
    rows = cursor.fetchall()
    for (nconst, kft_string) in tqdm(rows, desc="Populating people_known_for"):
        tconsts = kft_string.split(',')
        for t in tconsts:
            t = t.strip()
            if t:
                cursor.execute("""
                    INSERT OR IGNORE INTO people_known_for (nconst, tconst)
                    VALUES (?, ?);
                """, (nconst, t))

    conn.commit()


def populate_crew(conn):
    # Expand raw_title_crew's comma-separated directors/writers into multiple rows
    cursor = conn.cursor()
    cursor.execute("SELECT tconst, directors, writers FROM raw_title_crew;")
    rows = cursor.fetchall()
    for (tconst, directors_str, writers_str) in tqdm(rows, desc="Populating crew"):
        if directors_str and directors_str != '\\N':
            for d in directors_str.split(','):
                d = d.strip()
                if d:
                    cursor.execute("""
                        INSERT OR IGNORE INTO crew (tconst, nconst, role)
                        VALUES (?, ?, 'director');
                    """, (tconst, d))
        if writers_str and writers_str != '\\N':
            for w in writers_str.split(','):
                w = w.strip()
                if w:
                    cursor.execute("""
                        INSERT OR IGNORE INTO crew (tconst, nconst, role)
                        VALUES (?, ?, 'writer');
                    """, (tconst, w))

    conn.commit()


def populate_principals_and_characters(conn):
    # Insert principal data
    insert_principals = """
    INSERT INTO principals (tconst, ordering, nconst, category, job)
    SELECT tconst, ordering, nconst, category, job
    FROM raw_title_principals;
    """
    conn.execute(insert_principals)

    # Expand the JSON array in 'characters'
    cursor = conn.cursor()
    cursor.execute("""
        SELECT tconst, ordering, nconst, characters
        FROM raw_title_principals
        WHERE characters IS NOT NULL
    """)
    rows = cursor.fetchall()

    for (tconst, ordering, nconst, characters_json) in tqdm(rows, desc="Populating principal_characters"):
        try:
            c_list = simplejson.loads(characters_json)
            if isinstance(c_list, list):
                for c in c_list:
                    if c:
                        cursor.execute("""
                            INSERT OR IGNORE INTO principal_characters (tconst, ordering, nconst, character)
                            VALUES (?, ?, ?, ?);
                        """, (tconst, ordering, nconst, c))
        except:
            # Malformed JSON, skip
            pass

    conn.commit()


if __name__ == '__main__':
    data_dir = Path(project_config.data_dir)
    db_path = data_dir / 'imdb.db'

    conn = sqlite3.connect(db_path)

    # 1) Create the new normalized schema (with averageRating/numVotes in titles + episodes)
    create_normalized_schema(conn)

    # 2) Populate each table
    populate_titles_and_genres(conn)
    populate_episodes(conn)
    populate_people(conn)
    populate_crew(conn)
    populate_principals_and_characters(conn)

    conn.close()
    print("Normalized tables created and populated successfully.")
