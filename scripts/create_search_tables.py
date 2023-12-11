from pathlib import Path
import sqlite3
import sys
from db.database_schema import search_db_schema

def create_db(db_path):
    """Create a new database at db_path"""
    db = sqlite3.connect(db_path)
    db.executescript(search_db_schema)
    return db


def create_search_tables(conn):
    print("Dropping search tables...")
    conn.executescript("""
        DROP TABLE IF EXISTS search_movies;
        DROP TABLE IF EXISTS search_shows;
        DROP TABLE IF EXISTS search_episodes;
        DROP TABLE IF EXISTS search_characters;
        DROP TABLE IF EXISTS search_people;
        DROP TABLE IF EXISTS search_;
    """)
    conn.commit()


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
    conn.execute(query)
    conn.commit()

def populate_search_characters(conn):
    query = """
    INSERT INTO search_characters (tconst, nconst, title, name, character, category, job)
    SELECT 
        p.tconst,
        p.ordering,
        p.nconst,
        b.primaryTitle,
        n.primaryName,
        p.characters,
        p.category,
        p.job
    FROM title_principals p
    JOIN title_basics b ON p.tconst = b.tconst
    JOIN name_basics n ON p.nconst = n.nconst;
    """
    
    print("Populating search_characters table...")
    conn.execute(query)
    conn.commit()

def populate_search_people(conn):
    query = """
    INSERT INTO search_people (nconst, name, birthYear, deathYear, primaryProfession, knownForTitles)
    SELECT 
        nconst,
        primaryName,
        birthYear,
        deathYear,
        primaryProfession,
        knownForTitles
    FROM name_basics;
    """
    
    print("Populating search_people table...")
    conn.execute(query)
    conn.commit()


if __name__ == '__main__':
    data_dir = Path(sys.argv[1])
    db_path = data_dir / 'imdb.db'
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    create_search_tables(conn)
    populate_search_movies(conn)
    populate_search_shows(conn)
    populate_search_episodes(conn)
    populate_search_characters(conn)
    populate_search_people(conn)

    # Close the database connection
    conn.close()