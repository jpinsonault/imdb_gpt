import sqlite3
import json
from jinja2 import Template, Environment
from pathlib import Path
from tqdm import tqdm
from config import project_config

# Function to create a Jinja2 environment with Python's built-in functions
def create_jinja_environment():
    env = Environment()
    env.globals.update(str=str)  # Provide the 'str' function to the environment
    return env

# Create the Jinja2 environment
jinja_env = create_jinja_environment()

# Dynamic Jinja2 templates with conditional logic
templates = {
'movie': jinja_env.from_string(
"""
Movie
=====
{{ title }} {% if year %}({{ year }}){% endif %})
{% if genres %}{{ genres | replace(',', ', ') }}{% endif %}{% if genres and rating %} | {% endif %}{% if rating %}{{ rating }}/10{% endif %}
{% if isAdult and isAdult == 1 %}Note: Adult Content{% endif %}
"""
),
'tvSeries': jinja_env.from_string(
"""
TV Show
=======
{ title }} {% if startYear or endYear %}({{ startYear }}{% if endYear and endYear != '\\\\N' %} - {{ endYear }}{% endif %}){% endif %}
{% if genres %}{{ genres | replace(',', ', ') }}{% endif %}{% if genres and rating %} | {% endif %}{% if rating %}{{ rating }}/10{% endif %}
"""
),
'person': jinja_env.from_string(
"""
Person
======
{{ primaryName }}
{% if primaryProfession %}{{ primaryProfession | replace(',', ', ') }}{% endif %}
{% if knownForTitles %}Known for
=========
{{ knownForTitles | replace(',', '\n') }}{% endif %}
"""
),
'character': jinja_env.from_string(
"""
Character
=========
{{ character }} {% if title %}in {{ title }}{% endif %}
{% if name %}Played by {{ name }}{% endif %}
"""
)
}

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

# Function to save dataset as JSONL files with dynamic templating
def save_datasets(db_path, output_dir):
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
                    row[5] = ', '.join(resolved_titles)  # Update knownForTitles with resolved titles

                # Check if character's tconst or nconst are in included movies or people
                if entity_type == 'character' and (row[0] not in included_movies or row[1] not in included_people):
                    filtered_counts[entity_type] += 1
                    continue

                # Construct the data dictionary
                if entity_type == 'movie':
                    data = {
                        'tconst': row[0], 'title': row[1], 'year': row[2], 'genres': row[3],
                        'rating': row[4], 'numVotes': row[5], 'isAdult': row[6], 'runtimeMinutes': row[7]
                    }
                    query_value = row[1]
                elif entity_type == 'tvSeries':
                    data = {
                        'tconst': row[0], 'title': row[1], 'startYear': row[2], 'endYear': row[3],
                        'genres': row[4], 'rating': row[5], 'numVotes': row[6],
                        'isAdult': row[7], 'runtimeMinutes': row[8]
                    }
                    query_value = row[1]
                elif entity_type == 'episode':
                    parent_title = tconst_to_title.get(row[1], 'Unknown')
                    data = {
                        'tconst': row[0], 'parentTconst': row[1], 'title': row[2],
                        'parentTitle': parent_title, 'seasonNumber': row[3], 'episodeNumber': row[4], 'isAdult': row[5]
                    }
                    query_value = row[2]
                elif entity_type == 'person':
                    data = {
                        'nconst': row[0], 'primaryName': row[1], 'birthYear': row[2], 'deathYear': row[3],
                        'primaryProfession': row[4], 'knownForTitles': row[5]
                    }
                    query_value = row[1]
                elif entity_type == 'character':
                    title = tconst_to_title.get(row[0], 'Unknown')
                    name = nconst_to_name.get(row[1], 'Unknown')
                    data = {
                        'tconst': row[0], 'nconst': row[1], 'name': name, 'character': row[2],
                        'title': title, 'category': row[3]
                    }
                    query_value = row[2]

                # Remove empty or None values for a cleaner output
                data = {k: v for k, v in data.items() if v not in (None, '', '\\\\N')}

                # Format the template
                formatted_text = templates[entity_type].render(data).strip()

                # Create the JSON object for output
                json_record = json.dumps({
                    "type": entity_type,
                    "query": query_value,
                    "result": formatted_text,
                    **{k: v for k, v in data.items() if k in ['tconst', 'nconst', 'parentTconst']}
                })

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
    
    save_datasets(data_dir / 'imdb.db', data_dir)
