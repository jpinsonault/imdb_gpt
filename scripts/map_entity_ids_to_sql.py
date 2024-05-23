
import re
import sqlite3
from config import project_config
from pathlib import Path
from tqdm import tqdm

find_characters_regex = re.compile(r'\"(.*?)\"')
def populate_entity_vectors_movies(conn):
    print("Populating entity_vectors and entities tables with movies...")
    insert_vector_query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(title)
    FROM search_movies
    WHERE title IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    insert_entity_query = """
    INSERT INTO entities (entityName)
    SELECT DISTINCT title
    FROM search_movies
    WHERE title IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    with conn:
        conn.execute(insert_vector_query)
        conn.execute(insert_entity_query)

def populate_entity_vectors_shows(conn):
    print("Populating entity_vectors and entities tables with shows...")
    insert_vector_query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(title)
    FROM search_shows
    WHERE title IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    insert_entity_query = """
    INSERT INTO entities (entityName)
    SELECT DISTINCT title
    FROM search_shows
    WHERE title IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    with conn:
        conn.execute(insert_vector_query)
        conn.execute(insert_entity_query)

def populate_entity_vectors_episodes(conn):
    print("Populating entity_vectors and entities tables with episodes...")
    insert_vector_query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(title)
    FROM search_episodes
    WHERE title IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    insert_entity_query = """
    INSERT INTO entities (entityName)
    SELECT DISTINCT title
    FROM search_episodes
    WHERE title IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    with conn:
        conn.execute(insert_vector_query)
        conn.execute(insert_entity_query)

def populate_entity_vectors_characters(conn):
    print("Populating entity_vectors and entities tables with characters...")
    insert_vector_query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(character)
    FROM search_characters
    WHERE character IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    insert_entity_query = """
    INSERT INTO entities (entityName)
    SELECT DISTINCT character
    FROM search_characters
    WHERE character IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    with conn:
        conn.execute(insert_vector_query)
        conn.execute(insert_entity_query)

def populate_entity_vectors_people(conn):
    print("Populating entity_vectors and entities tables with people...")
    insert_vector_query = """
    INSERT INTO entity_vectors (entityName)
    SELECT DISTINCT LOWER(name)
    FROM search_people
    WHERE name IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    insert_entity_query = """
    INSERT INTO entities (entityName)
    SELECT DISTINCT name
    FROM search_people
    WHERE name IS NOT NULL
    ON CONFLICT(entityName) DO NOTHING;
    """
    with conn:
        conn.execute(insert_vector_query)
        conn.execute(insert_entity_query)

def create_vector_id_table(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""DROP TABLE IF EXISTS entity_vectors;""")
        conn.execute(create_entity_vector_table_query)

    
        
if __name__ == '__main__':
    data_dir = Path(project_config["data_dir"]) / "imdb.db"
    create_vector_id_table(data_dir)
    
    with sqlite3.connect(data_dir) as conn:
        # populate_entity_vectors_movies(conn)
        populate_entity_vectors_shows(conn)
        populate_entity_vectors_episodes(conn)
        populate_entity_vectors_characters(conn)
        populate_entity_vectors_people(conn)
    