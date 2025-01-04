# create_search_tables.py

import json
from pathlib import Path
import sqlite3
import sys
import db.database_schemas as schemas
from config import project_config
from tqdm import tqdm
import simplejson

from scripts import utils


def populate_search_movies(conn):
    query = """
    INSERT INTO search_movies (tconst, title, year, genres, rating, numVotes, isAdult, runtimeMinutes)
    SELECT 
        b.tconst,
        b.primaryTitle,
        b.startYear,
        b.genres,
        r.averageRating,
        r.numVotes,
        b.isAdult,
        b.runtimeMinutes
    FROM title_basics b
    LEFT JOIN title_ratings r ON b.tconst = r.tconst
    WHERE b.titleType = 'movie';
    """
    
    print("Populating search_movies table...")
    conn.execute("""DROP TABLE IF EXISTS search_movies;""")
    conn.execute(schemas.search_movies_table_schema)
    
    conn.execute(query)
    conn.commit()


def populate_search_shows(conn):
    query = """
    INSERT INTO search_shows (tconst, title, startYear, endYear, genres, rating, numVotes, isAdult, runtimeMinutes)
    SELECT 
        b.tconst,
        b.primaryTitle,
        b.startYear,
        b.endYear,
        b.genres,
        r.averageRating,
        r.numVotes,
        b.isAdult,
        b.runtimeMinutes
    FROM title_basics b
    LEFT JOIN title_ratings r ON b.tconst = r.tconst
    WHERE b.titleType IN ('tvSeries', 'tvMiniSeries');
    """
    
    print("Populating search_shows table...")
    conn.execute("""DROP TABLE IF EXISTS search_shows;""")
    conn.execute(schemas.search_shows_table_schema)
    
    conn.execute(query)
    conn.commit()

def populate_search_episodes(conn):
    query = """
    INSERT INTO search_episodes (tconst, parentTconst, title, parentTitle, seasonNumber, episodeNumber, isAdult)
    SELECT 
        e.tconst,
        e.parentTconst,
        b.primaryTitle,
        pb.primaryTitle,
        e.seasonNumber,
        e.episodeNumber,
        pb.isAdult
    FROM title_episode e
    JOIN title_basics b ON e.tconst = b.tconst
    JOIN title_basics pb ON e.parentTconst = pb.tconst;
    """
    
    print("Populating search_episodes table...")
    conn.execute("""DROP TABLE IF EXISTS search_episodes;""")
    conn.execute(schemas.search_episodes_table_schema)
    
    conn.execute(query)
    conn.commit()

import json
import sqlite3

def populate_search_characters(conn):
    # Drop the existing table and recreate it
    print("Populating search_characters table...")
    conn.execute("""DROP TABLE IF EXISTS search_characters;""")
    conn.execute(schemas.search_characters_table_schema)

    # Define the select query
    select_query = """
    SELECT 
        p.tconst,
        p.nconst,
        b.primaryTitle,
        n.primaryName,
        p.characters,
        p.category,
        p.job
    FROM title_principals p
    JOIN title_basics b ON p.tconst = b.tconst
    JOIN name_basics n ON p.nconst = n.nconst
    WHERE p.characters IS NOT NULL;
    """

    # Define the insert query
    insert_query = """
    INSERT INTO search_characters (tconst, nconst, title, name, character, category, job)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """

    print("Querying search_characters table...")
    cursor = conn.cursor()
    cursor.execute(select_query)

    batch_size = 1000000  # Adjust the batch size as needed
    batch = []
    
    
    total_rows = conn.execute("SELECT COUNT(*) FROM title_principals WHERE characters IS NOT NULL;").fetchone()[0]

    for row in tqdm(cursor, total=total_rows, desc="Processing title_principals"):
        tconst, nconst, title, name, characters_json, category, job = row
        try:
            characters = simplejson.loads(characters_json)
            for character in characters:
                batch.append((tconst, nconst, title, name, character, category, job))
                if len(batch) >= batch_size:
                    conn.executemany(insert_query, batch)
                    batch.clear()
        except Exception as e:
            print(f"Malformed JSON or incorrect format in tconst {tconst}: {characters_json}")

    # Insert any remaining items in the batch
    if batch:
        conn.executemany(insert_query, batch)

    conn.commit()
    cursor.close()



def populate_search_people(conn):
    insertQuery = """
    INSERT INTO search_people (nconst, primaryName, birthYear, deathYear, primaryProfession, knownForTitles)
    SELECT 
        nconst,
        primaryName,
        birthYear,
        deathYear,
        primaryProfession,
        knownForTitles
    FROM name_basics;
    """

    create_query = """
    CREATE TABLE search_people (
        nconst TEXT PRIMARY KEY,
        primaryName TEXT,
        birthYear INTEGER,
        deathYear INTEGER,
        primaryProfession TEXT,
        knownForTitles TEXT
    );
    """
    
    print("Populating search_people table...")
    conn.execute("""DROP TABLE IF EXISTS search_people;""")
    conn.execute(create_query)
    conn.execute(insertQuery)
    conn.commit()

def check_principle_character_json(conn):
    # load all characters
    query = """
    SELECT tconst, characters
    FROM title_principals
    WHERE characters <> '[]';
    """
    
    characters = conn.execute(query).fetchall()
    
    # check if all characters are valid JSON arrays of strings or empty arrays
    for tconst, character_json in tqdm(characters):
        try:
            character_list = simplejson.loads(character_json)

            if not isinstance(character_list, list):
                raise ValueError("Not a list")

            if not all(isinstance(item, str) for item in character_list):
                raise ValueError("List elements are not strings")

        except Exception as e:
            print(f"Malformed JSON or incorrect format in tconst {tconst}: {character_json}")
            print(f"Error: {e}")
            break
    else:
        print("All character entries are valid JSON arrays of strings or empty arrays")



def create_alphabet(db_path, max_entity_length):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    unique_chars = set()

    # drop the existing alphabet table
    conn.execute("DROP TABLE IF EXISTS alphabet;")
    conn.execute("CREATE TABLE alphabet (characters TEXT);")

    print("Querying titles and names...")

    num_movies = conn.execute("SELECT COUNT(*) FROM search_movies;").fetchone()[0]
    cursor.execute("""SELECT title FROM search_movies""")
    
    for row in tqdm(cursor, desc="Processing search_movies", total=num_movies):
        text = row[0]
        if len(text) <= max_entity_length:
            unique_chars.update(text)
            
    num_shows = conn.execute("SELECT COUNT(*) FROM search_shows;").fetchone()[0]
    cursor.execute("""SELECT title FROM search_shows""")
    
    for row in tqdm(cursor, desc="Processing search_shows", total=num_shows):
        text = row[0]
        if len(text) <= max_entity_length:
            unique_chars.update(text)
            
    num_episodes = conn.execute("SELECT COUNT(*) FROM search_episodes;").fetchone()[0]
    cursor.execute("""SELECT title FROM search_episodes""")

    for row in tqdm(cursor, desc="Processing search_episodes", total=num_episodes):
        text = row[0]
        if len(text) <= max_entity_length:
            unique_chars.update(text)

    num_characters = conn.execute("SELECT COUNT(*) FROM search_characters;").fetchone()[0]
    cursor.execute("""SELECT name FROM search_characters""")

    for row in tqdm(cursor, desc="Processing search_characters", total=num_characters):
        text = row[0]
        if len(text) <= max_entity_length:
            unique_chars.update(text)
            
    # Convert the set of characters to a sorted list
    alphabet = sorted(unique_chars)
    
    print(f"Alphabet: {alphabet}")
    print(f"{len(alphabet)} characters")

    # Insert the alphabet into the alphabet table
    alphabet_json = json.dumps(alphabet)
    conn.execute("DELETE FROM alphabet")  # Clear existing data
    conn.execute("INSERT INTO alphabet (characters) VALUES (?)", (alphabet_json,))

    conn.commit()
    conn.close()


def create_entity_vectors_table(conn):
    print("Creating entity_vectors table. This takes a while...")
    progress = tqdm(total=5, desc="Creating entity_vectors table")
    conn.execute("""DROP TABLE IF EXISTS entity_vectors;""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS entity_vectors (
    vectorId INTEGER PRIMARY KEY AUTOINCREMENT,
    entityName TEXT UNIQUE NOT NULL
    );""")
    conn.commit()
    
    progress.desc = "Inserting entities from search_movies"
    progress.update(0)
    query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(title)
    FROM search_movies
    WHERE title IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    
    conn.execute(query)
    conn.commit()
    
    progress.desc = "Inserting entities from search_shows"
    progress.update(1)
    query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(title)
    FROM search_shows
    WHERE title IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """

    conn.execute(query)
    conn.commit()

    progress.desc = "Inserting entities from search_episodes"
    progress.update(1)
    query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(title)
    FROM search_episodes
    WHERE title IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """

    conn.execute(query)
    conn.commit()
    
    progress.desc = "Inserting entities from search_characters"
    progress.update(1)
    query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(name)
    FROM search_characters
    WHERE name IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """

    conn.execute(query)
    conn.commit()
    
    progress.desc = "Inserting entities from search_people"
    progress.update(1)
    query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(primaryName)
    FROM search_people
    WHERE primaryName IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """

    conn.execute(query)
    conn.commit()

    progress.close()

        
if __name__ == '__main__':
    data_dir = Path(project_config['data_dir'])
    max_entity_length = project_config['entities']['max_entity_length']
    db_path = data_dir / 'imdb.db'
    
    # utils.verify_destructive(message="This will delete the existing database. Are you sure you want to continue?")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    tables = {
        'search_movies': populate_search_movies,
        'search_shows': populate_search_shows,
        'search_episodes': populate_search_episodes,
        'search_characters': populate_search_characters,
        'search_people': populate_search_people,
    }
    
    for table, populate_table in tables.items():
        populate_table(conn)
        
    create_entity_vectors_table(conn)
        
    create_alphabet(db_path, max_entity_length)

    # Close the database connection
    conn.close()