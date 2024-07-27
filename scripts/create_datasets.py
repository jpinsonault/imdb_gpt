from collections import defaultdict, namedtuple
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import time

from tqdm import tqdm
import pickle
from config import project_config

MovieDescription = namedtuple('MovieDescription', ['title', 'tconst', 'description'])



db_schema = """
CREATE TABLE title_akas (
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

CREATE TABLE title_basics (
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

CREATE TABLE title_crew (
    tconst TEXT PRIMARY KEY,
    directors TEXT,
    writers TEXT,
    FOREIGN KEY (tconst) REFERENCES title_basics (tconst)
);

CREATE TABLE title_episode (
    tconst TEXT PRIMARY KEY,
    parentTconst TEXT,
    seasonNumber INTEGER,
    episodeNumber INTEGER,
    FOREIGN KEY (tconst) REFERENCES title_basics (tconst),
    FOREIGN KEY (parentTconst) REFERENCES title_basics (tconst)
);

CREATE TABLE title_principals (
    tconst TEXT NOT NULL,
    ordering INTEGER NOT NULL,
    nconst TEXT,
    category TEXT,
    job TEXT,
    characters TEXT,
    PRIMARY KEY (tconst, ordering),
    FOREIGN KEY (tconst) REFERENCES title_basics (tconst),
    FOREIGN KEY (nconst) REFERENCES name_basics (nconst)
);

CREATE TABLE title_ratings (
    tconst TEXT PRIMARY KEY,
    averageRating REAL,
    numVotes INTEGER,
    FOREIGN KEY (tconst) REFERENCES title_basics (tconst)
);

CREATE TABLE name_basics (
    nconst TEXT PRIMARY KEY,
    primaryName TEXT,
    birthYear INTEGER,
    deathYear INTEGER,
    primaryProfession TEXT,
    knownForTitles TEXT
);"""

def format_movie_descriptions(db_path):
    query = f"""
    SELECT
        tb.tconst,
        tb.primaryTitle,
        tb.startYear,
        tb.genres,
        tr.averageRating,
        tr.numVotes,
        group_concat(CASE WHEN tp.category IN ('actor', 'actress') THEN nb.primaryName || ':@' || tp.characters ELSE NULL END, '|') AS actors,
        group_concat(CASE WHEN tp.category = 'director' THEN nb.primaryName ELSE NULL END, ',') AS directors,
        group_concat(CASE WHEN tp.category = 'writer' THEN nb.primaryName ELSE NULL END, ',') AS writers
    FROM
        title_basics AS tb
    LEFT JOIN
        title_ratings AS tr ON tb.tconst = tr.tconst
    LEFT JOIN
        title_principals AS tp ON tb.tconst = tp.tconst
    LEFT JOIN
        name_basics AS nb ON tp.nconst = nb.nconst
    GROUP BY
        tb.tconst    
    """

    print("Connecting to database...")
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the query
    print("Executing query...")
    cursor.execute(query)
    movies = cursor.fetchall()

    # Formatting the results
    formatted_movies = []
    for movie in tqdm(movies, desc="Formatting movies"):
        # Adjusted to match the number of selected columns
        (tconst, title, start_year, genres, rating, num_votes, actors, directors, writers) = movie
        # The rest of your code remains the same...
        genres = genres.replace(',', ', ') if genres else genres

        movie_desc = f"Title: {title}\nYear: {start_year}\nGenres: {genres}\n"

        if directors:
            directors = ', '.join(directors.split(',')) if directors else directors
            movie_desc += f"Directors: {directors}\n"
        if writers:
            writers = ', '.join(writers.split(',')) if writers else writers
            movie_desc += f"Writers: {writers}\n"
        if actors:
            actors = reformat_characters(actors)
            movie_desc += f"Cast:\n{actors}\n"
        if rating:
            movie_desc += f"Rating: {rating} ({num_votes} votes)\n"
        movie_desc += '-'*40 + '\n'
        formatted_movies.append(MovieDescription(title, tconst, movie_desc))

    conn.close()
    return formatted_movies


def reformat_characters(actor_string):
    """
    Reformat the actor string to replace the list of roles with a more natural representation.
    Example: 'Owen: ["Role1", "Role2"]' to 'Owen as Role1 and Role2'
    """
    if not actor_string:
        return ""
    
    actors = actor_string.split('|')
    formatted_actors = []
    for actor in actors:
        parts = actor.split(':@')
        if len(parts) >= 1:
            try:
                name, roles = parts
            except ValueError:
                print(parts, actor)
                raise
            roles = roles.strip("[]").replace('"', '').replace(", ", " and ")

            role_string = f" - {name} as {roles}\n"
            formatted_actors.append(role_string)
        else:
            formatted_actors.append(actor)

    return "".join(formatted_actors)


def create_movie_dataset(db_path):
    formatted_movies = format_movie_descriptions(db_path)

    for movie in random.sample(formatted_movies, 50):
        print(movie.description)


def list_movie_titles(db_path, limit=None):
    query = f"""
    SELECT primaryTitle, tconst FROM title_basics    
    """
    
    if limit:
        query += f" LIMIT {limit}"

    print("Fetching movie titles...")

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)
    titles = cursor.fetchall()

    conn.close()
    return titles

def list_names(db_path, limit=None):
    query = f"""
    SELECT primaryName, nconst FROM name_basics
    """
    
    if limit:
        query += f" LIMIT {limit}"

    print("Fetching names...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(query)
    names = cursor.fetchall()

    conn.close()
    return names

def list_episode_titles(db_path, limit=None):
    query = f"""
    SELECT primaryTitle, tconst
    FROM title_episode 
    JOIN title_basics
    ON title_episode.tconst = title_basics.tconst
    """
    
    if limit:
        query += f" LIMIT {limit}"
    

    print("Fetching episode titles...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(query)
    episode_titles = cursor.fetchall()

    conn.close()
    return episode_titles


def list_characters(db_path, limit=None):
    query = """
    SELECT
        tp.characters, tp.tconst, tp.nconst
    FROM
        title_principals AS tp
    WHERE
        tp.category IN ('actor', 'actress')
    AND
        tp.characters IS NOT NULL    
    """
    
    if limit:
        query += f" LIMIT {limit}"

    print("Fetching character names...")

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)
    character_entries = cursor.fetchall()

    conn.close()

    character_tuples = []
    for characters_str, tconst, nconst in character_entries:
        # Using regex to find all occurrences of characters between quotes
        characters = re.findall(r'\"(.*?)\"', characters_str)
        for character in characters:
            character_tuples.append((character, tconst, nconst))

    return character_tuples


def save_all_entities_to_file(data_dir, db_filename):
    characters = set(list_characters(data_dir / db_filename))
    titles = set(list_movie_titles(data_dir / db_filename))
    names = set(list_names(data_dir / db_filename))
    episode_titles = set(list_episode_titles(data_dir / db_filename))

    all_entities = defaultdict(list)
    
    print(f"found titles: {len(titles)}, first two: {list(titles)[0:2]}")
    for title, tconst in titles:
        all_entities[title.lower()].append({'tconst': tconst, 'type': 'title'})
    
    print(f"found names: {len(names)}, first two: {list(names)[0:2]}")
    for name, nconst in names:
        all_entities[name.lower()].append({'nconst': nconst, 'type': 'name'})
    
    print(f"found episode titles: {len(episode_titles)}, first two: {list(episode_titles)[0:2]}")
    all_entities.extend(episode_titles)
    
    print(f"found characters: {len(characters)}, first two: {list(characters)[0:2]}")
    all_entities.extend(characters)

    all_entities = [entity for entity in all_entities if len(entity) < 94 and len(entity) > 2]

    all_entities = list(set(all_entities))
    
    print(f"first 10 entities: {all_entities[0:10]}")

    print(f"Total entities: {len(all_entities)}")
    print(f"Min length: {min(len(entity) for entity in all_entities)}")
    print(f"Max length: {max(len(entity) for entity in all_entities)}")
    print(f"Average length: {sum(len(entity) for entity in all_entities) / len(all_entities)}")
    percentile_95 = sorted(len(entity) for entity in all_entities)[int(len(all_entities) * 0.95)]
    percentile_99 = sorted(len(entity) for entity in all_entities)[int(len(all_entities) * 0.99)]
    print(f"99th percentile: {percentile_99}")
    print(f"95th percentile: {percentile_95}")

    # write each entity to a file, one per line
    entity_file = './entities.txt'
    with open(entity_file, 'w', encoding='utf-8') as f:
        for entity in tqdm(all_entities, desc="Writing entities to file"):
            try:
                f.write(entity + '\n')
            except UnicodeEncodeError:
                print(f"UnicodeEncodeError: {entity}")
                continue
            
            
def find_long_title_high_rating_movies(db_path):
    query = """
    SELECT
        tb.tconst,
        tb.primaryTitle,
        tr.averageRating,
        tr.numVotes
    FROM
        title_basics AS tb
    JOIN
        title_ratings AS tr ON tb.tconst = tr.tconst
    WHERE
        LENGTH(tb.primaryTitle) > 64
    AND
        tr.averageRating > 5
    AND
        tr.numVotes > 100
    """

    print("Fetching movies with long titles and high ratings...")

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)
    movies = cursor.fetchall()

    conn.close()

    for movie in movies:
        print(f"Title: {movie[1]}, Rating: {movie[2]}, Number of Votes: {movie[3]}")
        
    print(f"Found {len(movies)} movies with long titles and high ratings")
    
    # print out all the length stats
    print(f"Min length: {min(len(movie[1]) for movie in movies)}")
    print(f"Max length: {max(len(movie[1]) for movie in movies)}")
    print(f"Average length: {sum(len(movie[1]) for movie in movies) / len(movies)}")
    percentile_95 = sorted(len(movie[1]) for movie in movies)[int(len(movies) * 0.95)]
    percentile_99 = sorted(len(movie[1]) for movie in movies)[int(len(movies) * 0.99)]
    print(f"99th percentile: {percentile_99}")
    print(f"95th percentile: {percentile_95}")


if __name__ == '__main__':
    data_dir = project_config['data_dir']
    
    save_all_entities_to_file(data_dir, 'imdb.db')