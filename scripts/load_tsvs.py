import csv
import json
from pathlib import Path
from pprint import pprint
import random
import sqlite3
import sys
from tqdm import tqdm
import re
from collections import defaultdict, namedtuple
from db.database_schema import db_schema
from config import project_config

def create_db(db_path):
    """Create a new database at db_path"""
    db = sqlite3.connect(db_path)
    db.executescript(db_schema)
    db.close()

def describe_tables(db_path):
    """Print the schema of the database at db_path, and the number of rows in each table"""
    db = sqlite3.connect(db_path)
    cursor = db.cursor()

    # Get the table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [name[0] for name in cursor.fetchall()]

    # Print the schema of each table
    for table_name in table_names:
        print(f'Schema for {table_name}:')
        cursor.execute(f'PRAGMA table_info({table_name});')
        for row in cursor.fetchall():
            print(row)
        print()

    # Print the number of rows in each table
    for table_name in table_names:
        cursor.execute(f'SELECT COUNT(*) FROM {table_name};')
        print(f'{table_name} has {cursor.fetchone()[0]} rows')

    db.close()


def get_file_line_count(file_path):
    """Return the number of lines in the file at file_path"""
    with open(file_path, mode='rt', encoding='utf-8') as file:
        return sum(1 for line in file)

def load_akas_file_into_sqlite(db_path, akas_file_path: Path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the title_akas table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS title_akas (
            titleId TEXT NOT NULL,
            ordering INTEGER NOT NULL,
            title TEXT,
            region TEXT,
            language TEXT,
            types TEXT,
            attributes TEXT,
            isOriginalTitle INTEGER,
            PRIMARY KEY (titleId, ordering),
            FOREIGN KEY (titleId) REFERENCES title_basics (tconst)
        );
    ''')

    # Open the akas file and load the data
    with open(akas_file_path, mode='rt', encoding='utf-8') as akas_file:
        akas_reader = csv.DictReader(akas_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in tqdm(akas_reader, total=get_file_line_count(akas_file_path), desc='Loading title_akas'):
            # Replace '\\N' with None
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            # Insert the row into the title_akas table
            cursor.execute('''
                INSERT OR IGNORE INTO title_akas (
                    titleId, ordering, title, region, language, types, attributes, isOriginalTitle
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['titleId'], 
                row['ordering'], 
                row['title'], 
                row['region'], 
                row['language'], 
                row['types'], 
                row['attributes'], 
                row['isOriginalTitle']
            ))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def load_title_basics_into_sqlite(db_path, title_basics_file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(title_basics_file_path, mode='rt', encoding='utf-8') as title_basics_file:
        title_basics_reader = csv.DictReader(title_basics_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in tqdm(title_basics_reader, total=get_file_line_count(title_basics_file_path), desc='Loading title_basics'):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute('''
                INSERT OR IGNORE INTO title_basics (
                    tconst, titleType, primaryTitle, originalTitle, isAdult, startYear, endYear, runtimeMinutes, genres
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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


def load_title_crew_into_sqlite(db_path, title_crew_file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(title_crew_file_path, mode='rt', encoding='utf-8') as title_crew_file:
        title_crew_reader = csv.DictReader(title_crew_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in tqdm(title_crew_reader, total=get_file_line_count(title_crew_file_path), desc='Loading title_crew'):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute('''
                INSERT OR IGNORE INTO title_crew (
                    tconst, directors, writers
                ) VALUES (?, ?, ?)
            ''', (
                row['tconst'],
                row['directors'],
                row['writers']
            ))

    conn.commit()
    conn.close()


def load_title_episode_into_sqlite(db_path, title_episode_file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(title_episode_file_path, mode='rt', encoding='utf-8') as title_episode_file:
        title_episode_reader = csv.DictReader(title_episode_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in tqdm(title_episode_reader, total=get_file_line_count(title_episode_file_path), desc='Loading title_episode'):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute('''
                INSERT OR IGNORE INTO title_episode (
                    tconst, parentTconst, seasonNumber, episodeNumber
                ) VALUES (?, ?, ?, ?)
            ''', (
                row['tconst'],
                row['parentTconst'],
                row['seasonNumber'],
                row['episodeNumber']
            ))

    conn.commit()
    conn.close()

def load_title_principals_into_sqlite(db_path, title_principals_file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(title_principals_file_path, mode='rt', encoding='utf-8') as title_principals_file:
        title_principals_reader = csv.DictReader(title_principals_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in tqdm(title_principals_reader, total=get_file_line_count(title_principals_file_path), desc='Loading title_principals'):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute('''
                INSERT OR IGNORE INTO title_principals (
                    tconst, ordering, nconst, category, job, characters
                ) VALUES (?,                ?, ?, ?, ?, ?)
            ''', (
                row['tconst'],
                row['ordering'],
                row['nconst'],
                row['category'],
                row['job'],
                row['characters']
            ))

    conn.commit()
    conn.close()

def load_title_ratings_into_sqlite(db_path, title_ratings_file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(title_ratings_file_path, mode='rt', encoding='utf-8') as title_ratings_file:
        title_ratings_reader = csv.DictReader(title_ratings_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in tqdm(title_ratings_reader, total=get_file_line_count(title_ratings_file_path), desc='Loading title_ratings'):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute('''
                INSERT OR IGNORE INTO title_ratings (
                    tconst, averageRating, numVotes
                ) VALUES (?, ?, ?)
            ''', (
                row['tconst'],
                row['averageRating'],
                row['numVotes']
            ))

    conn.commit()
    conn.close()

def load_name_basics_into_sqlite(db_path, name_basics_file_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(name_basics_file_path, mode='rt', encoding='utf-8') as name_basics_file:
        name_basics_reader = csv.DictReader(name_basics_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in tqdm(name_basics_reader, total=get_file_line_count(name_basics_file_path), desc='Loading name_basics'):
            for key, value in row.items():
                if value == '\\N':
                    row[key] = None

            cursor.execute('''
                INSERT OR IGNORE INTO name_basics (
                    nconst, primaryName, birthYear, deathYear, primaryProfession, knownForTitles
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                row['nconst'],
                row['primaryName'],
                row['birthYear'],
                row['deathYear'],
                row['primaryProfession'],
                row['knownForTitles']
            ))

    conn.commit()
    conn.close()


def create_and_load_db(db_path, file_paths):
    """Create a new database and load IMDb data files into it."""
    # Create the database
    create_db(db_path)

    # Load each IMDb file into the database
    load_akas_file_into_sqlite(db_path, file_paths['akas'])

    load_title_basics_into_sqlite(db_path, file_paths['title_basics'])

    load_title_crew_into_sqlite(db_path, file_paths['title_crew'])

    load_title_episode_into_sqlite(db_path, file_paths['title_episode'])

    load_title_principals_into_sqlite(db_path, file_paths['title_principals'])

    load_title_ratings_into_sqlite(db_path, file_paths['title_ratings'])

    load_name_basics_into_sqlite(db_path, file_paths['name_basics'])

    print("Database created and data loaded successfully.")


def load_db_files(data_dir: Path):
    db_path = data_dir / 'imdb.db'

    file_paths = {
        'akas'            : data_dir / 'title.akas.tsv',
        'title_basics'    : data_dir / 'title.basics.tsv',
        'title_crew'      : data_dir / 'title.crew.tsv',
        'title_episode'   : data_dir / 'title.episode.tsv',
        'title_principals': data_dir / 'title.principals.tsv',
        'title_ratings'   : data_dir / 'title.ratings.tsv',
        'name_basics'     : data_dir / 'name.basics.tsv'
    }

    create_and_load_db(db_path, file_paths)


if __name__ == '__main__':
    data_dir = Path(project_config["data_dir"]) / 'imdb_tsvs'
    
    load_db_files(data_dir)