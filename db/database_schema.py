
from pathlib import Path
import sqlite3


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
);
 
"""

search_db_schema = """
CREATE TABLE search_movies (
    embeddingId INTEGER,
    tconst TEXT PRIMARY KEY,
    title TEXT,
    year INTEGER,
    genres TEXT,
    rating REAL,
    numVotes INTEGER,
    isAdult INTEGER,
    runtimeMinutes INTEGER
);

CREATE TABLE search_shows (
    embeddingId INTEGER,
    tconst TEXT PRIMARY KEY,
    title TEXT,
    startYear INTEGER,
    endYear INTEGER,
    genres TEXT,
    rating REAL,
    numVotes INTEGER,
    isAdult INTEGER,
    runtimeMinutes INTEGER
);

CREATE TABLE search_episodes (
    embeddingId INTEGER,
    tconst TEXT PRIMARY KEY,
    parentTconst TEXT,
    title TEXT,
    parentTitle TEXT,
    seasonNumber INTEGER,
    episodeNumber INTEGER,
    isAdult INTEGER -- from parent
);

CREATE TABLE search_characters (
    embeddingId INTEGER,
    tconst TEXT,
    ordering INTEGER,
    PRIMARY KEY (tconst, ordering),
    nconst TEXT,
    title TEXT,
    name TEXT,
    character TEXT,
    category TEXT,
    job TEXT
    
);

CREATE TABLE search_people (
    embeddingId INTEGER,
    nconst TEXT PRIMARY KEY,
    name TEXT,
    birthYear INTEGER,
    deathYear INTEGER,
    primaryProfession TEXT,
    knownForTitles TEXT
);

CREATE TABLE entities (
    embeddingId INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    tconst TEXT,
    ncounst TEXT
    
);"""