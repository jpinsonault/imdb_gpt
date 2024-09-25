import sqlite3
import json
from pathlib import Path
from tqdm import tqdm
from config import project_config

def load_dictionaries(cursor):
    """
    Load all titles (tconst) and names (nconst) into dictionaries for lookup.
    """
    # Load titles from movies and shows into a single dictionary
    cursor.execute("SELECT tconst, title FROM search_movies")
    tconst_to_title = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("SELECT tconst, title FROM search_shows")
    tconst_to_title.update({row[0]: row[1] for row in cursor.fetchall()})

    # Load names from people into a dictionary
    cursor.execute("SELECT nconst, primaryName FROM search_people")
    nconst_to_name = {row[0]: row[1] for row in cursor.fetchall()}

    return tconst_to_title, nconst_to_name

def take_upto(iterable, n):
    """
    Take up to `n` items from the iterable.
    """
    for i, item in enumerate(iterable):
        if i == n:
            break
        yield item

# Function to save dataset as JSONL files with dynamic templating
def save_datasets(db_path, output_dir, max_input_length):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Load dictionaries for tconst and nconst lookups
    tconst_to_title, nconst_to_name = load_dictionaries(cursor)

    # Corrected queries to extract entities and their types
    queries = {
        'movie': """
            SELECT m.tconst, m.title, m.year, m.genres, m.rating, m.numVotes, m.isAdult, m.runtimeMinutes
            FROM search_movies m
        """,
        'tvSeries': """
            SELECT s.tconst, s.title, s.startYear, s.endYear, s.genres, s.rating, s.numVotes, s.isAdult, s.runtimeMinutes
            FROM search_shows s
        """,
        'person': """
            SELECT p.nconst, p.primaryName, p.birthYear, p.deathYear, p.primaryProfession, p.knownForTitles
            FROM search_people p
        """,
        'character': """
            SELECT c.tconst, c.nconst, c.name, c.character, c.category
            FROM search_characters c
        """
    }

    # Create output directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sets to track valid shows, movies, and people
    included_shows = set()
    included_movies = set()
    included_people = set()
    filtered_counts = {entity_type: 0 for entity_type in queries}

    # Process each entity type
    for entity_type, query in queries.items():
        cursor.execute(query)
        rows = cursor.fetchall()

        output_file = Path(output_dir) / f"{entity_type}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for row in tqdm(rows, desc=f"Processing {entity_type}"):
                # Filter out titles without years or with fewer than 10 votes
                if entity_type in ['movie', 'tvSeries']:
                    year = row[2]
                    votes_row = 5 if entity_type == 'movie' else 6
                    votes = row[votes_row]
                    # Check for missing year and votes
                    if year is None or votes is None or votes < 10:
                        filtered_counts[entity_type] += 1
                        continue

                    # Add valid movies and shows to the corresponding sets
                    if entity_type == 'movie':
                        included_movies.add(row[0])  # Add tconst of the movie to the set
                    elif entity_type == 'tvSeries':
                        included_shows.add(row[0])  # Add tconst of the show to the set

                # Check if person's known titles are in included movies or shows
                if entity_type == 'person':
                    known_tconsts = row[5]
                    if known_tconsts is None:
                        filtered_counts[entity_type] += 1
                        continue

                    known_tconsts = known_tconsts.split(',')
                    # Filter based on tconsts
                    if not any(tconst in included_movies or tconst in included_shows for tconst in known_tconsts):
                        filtered_counts[entity_type] += 1
                        continue
                    included_people.add(row[0])  # Add nconst of the person to the set
                    # Resolve the titles using the lookup dictionary
                    resolved_titles = [tconst_to_title.get(tconst, 'Unknown') for tconst in known_tconsts]
                    row = list(row)  # Convert to a mutable list
                    row[5] = ', '.join(take_upto(resolved_titles, 2))  # Update knownForTitles with resolved titles

                # Check if character's tconst or nconst are in included movies or people
                if entity_type == 'character' and (row[0] not in included_movies or row[1] not in included_people):
                    filtered_counts[entity_type] += 1
                    continue

                # Construct the data dictionary
                if entity_type == 'movie':
                    genres = row[3].replace(',', ', ') if row[3] is not None else 'Unknown'
                    data = {
                        'type': 'movie', 'tconst': row[0], 'title': row[1], 'year': row[2], 'genres': genres,
                        'rating': row[4], 'numVotes': row[5], 'isAdult': row[6], 'runtimeMinutes': row[7]
                    }
                elif entity_type == 'tvSeries':
                    genres = row[4].replace(',', ', ') if row[4] is not None else 'Unknown'
                    data = {
                        'type': 'tvSeries', 'tconst': row[0], 'title': row[1], 'startYear': row[2],
                        'endYear': row[3], 'genres': genres, 'rating': row[5], 'numVotes': row[6],
                        'isAdult': row[7], 'runtimeMinutes': row[8]
                    }
                elif entity_type == 'person':
                    knownForTitles = row[5].split(', ')
                    primaryProfession = row[4].replace(',', ', ') if row[4] is not None else 'Unknown'
                    data = {
                        'type': 'person', 'nconst': row[0], 'primaryName': row[1], 'birthYear': row[2], 
                        'deathYear': row[3], 'primaryProfession': primaryProfession, 'knownForTitles': knownForTitles
                    }
                elif entity_type == 'character':
                    title = tconst_to_title.get(row[0], 'Unknown')
                    data = {
                        'type': 'character', 'tconst': row[0], 'nconst': row[1],
                        'character': row[2], 'title': title, 'category': row[3]
                    }

                # Remove empty or None values for a cleaner output
                data = {k: v for k, v in data.items() if v not in (None, '', '\\\\N')}

                data['type'] = entity_type
                json_record = json.dumps(data)

                if len(json_record) > max_input_length:
                    filtered_counts[entity_type] += 1
                    continue

                # Write the JSON object to file
                f.write(json_record + '\n')

    # Log filtered counts
    print("Filtering Summary:")
    for entity_type, count in filtered_counts.items():
        print(f"Filtered {count} items from {entity_type}")

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    data_dir = Path(project_config['data_dir'])
    max_input_length = int(project_config['llm']['input_length'])
    
    save_datasets(data_dir / 'imdb.db', data_dir, max_input_length)