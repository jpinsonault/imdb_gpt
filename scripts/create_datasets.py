from collections import defaultdict
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

ALLOWED_GENRES = {
    'Drama', 'Comedy', 'Documentary', 'Romance', 'Action', 'Crime', 'Thriller', 'Horror', 
    'Adventure', 'Mystery', 'Family', 'Biography', 'Fantasy', 'History', 'Music', 'Unknown', 'Sci-Fi', 'Musical',
    'War', 'Western', 'Animation', 'Adult', 'Sport', 'Film-Noir', 'News',
}

ALLOWED_JOBS = {
    'actor', 'actress', 'producer', 'writer', 'director', 'cinematographer', 'animation_department', 'composer',
}

def save_datasets(db_path, output_dir, max_input_length):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tconst_to_title, nconst_to_name = load_dictionaries(cursor)

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

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    included_shows = set()
    included_movies = set()
    included_people = set()

    filtered_counts = {
        'not_enough_votes': defaultdict(int),
        'not_known_for_anything': defaultdict(int),
        'person_not_in_included_title': defaultdict(int),
        'character_not_in_included_title_or_person': defaultdict(int),
        'missing_year_or_votes': defaultdict(int),
        'genre_whitelist_excluded': defaultdict(int),
        'job_whitelist_excluded': defaultdict(int),
    }

    # Track how many rows we actually keep
    saved_counts = defaultdict(int)

    for entity_type, query in queries.items():
        cursor.execute(query)
        rows = cursor.fetchall()

        output_file = Path(output_dir) / f"{entity_type}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for row in tqdm(rows, desc=f"Processing {entity_type}"):
                if entity_type in ['movie', 'tvSeries']:
                    if entity_type == 'movie':
                        year = row[2]  # m.year
                        votes = row[5]  # m.numVotes
                    else:
                        year = row[2]  # s.startYear
                        votes = row[6]  # s.numVotes

                    if year is None or votes is None:
                        filtered_counts['missing_year_or_votes'][entity_type] += 1
                        continue
                    if votes < 10:
                        filtered_counts['not_enough_votes'][entity_type] += 1
                        continue

                    if entity_type == 'movie':
                        included_movies.add(row[0])
                    else:
                        included_shows.add(row[0])

                if entity_type == 'person':
                    known_tconsts = row[5]  # p.knownForTitles
                    if known_tconsts is None:
                        filtered_counts['not_known_for_anything'][entity_type] += 1
                        continue
                    known_tconsts = known_tconsts.split(',')
                    if not any(tconst in included_movies or tconst in included_shows for tconst in known_tconsts):
                        filtered_counts['person_not_in_included_title'][entity_type] += 1
                        continue
                    included_people.add(row[0])
                    resolved_titles = [tconst_to_title.get(tconst, 'Unknown') for tconst in known_tconsts]
                    row = list(row)
                    row[5] = ', '.join(take_upto(resolved_titles, 2))

                if entity_type == 'character':
                    if row[0] not in included_movies and row[0] not in included_shows:
                        filtered_counts['character_not_in_included_title_or_person'][entity_type] += 1
                        continue
                    if row[1] not in included_people:
                        filtered_counts['character_not_in_included_title_or_person'][entity_type] += 1
                        continue

                if entity_type == 'movie':
                    genres = row[3].split(',') if row[3] is not None else []
                    filtered_genres = [g for g in genres if g in ALLOWED_GENRES]
                    if not filtered_genres:
                        filtered_counts['genre_whitelist_excluded'][entity_type] += 1
                        continue
                    data = {
                        'type': 'movie',
                        'tconst': row[0],
                        'title': row[1],
                        'year': row[2],
                        'genres': filtered_genres,
                        'rating': row[4],
                        'votes': row[5],
                        'adult': row[6],
                        'runtime': row[7]
                    }

                elif entity_type == 'tvSeries':
                    genres = row[4].split(',') if row[4] is not None else []
                    filtered_genres = [g for g in genres if g in ALLOWED_GENRES]
                    if not filtered_genres:
                        filtered_counts['genre_whitelist_excluded'][entity_type] += 1
                        continue
                    data = {
                        'type': 'tvSeries',
                        'tconst': row[0],
                        'title': row[1],
                        'start': row[2],
                        'end': row[3],
                        'genres': filtered_genres,
                        'rating': row[5],
                        'votes': row[6],
                        'adult': row[7],
                        'runtime': row[8]
                    }

                elif entity_type == 'person':
                    knownForTitles = row[5].split(', ')
                    jobs = row[4].split(',') if row[4] is not None else []
                    filtered_jobs = [j for j in jobs if j in ALLOWED_JOBS]
                    if not filtered_jobs:
                        filtered_counts['job_whitelist_excluded'][entity_type] += 1
                        continue
                    data = {
                        'type': 'person',
                        'nconst': row[0],
                        'name': row[1],
                        'birth': row[2],
                        'death': row[3],
                        'jobs': filtered_jobs,
                        'knownFor': knownForTitles
                    }

                else:  # entity_type == 'character'
                    title = tconst_to_title.get(row[0], 'Unknown')
                    data = {
                        'type': 'character',
                        'tconst': row[0],
                        'nconst': row[1],
                        'name': row[2],
                        'inTitle': title,
                        'category': row[4]
                    }

                data = {k: v for k, v in data.items() if v not in (None, '', '\\\\N', [], ['Unknown'])}

                json_record = json.dumps(data, ensure_ascii=False)
                f.write(json_record + '\n')
                saved_counts[entity_type] += 1

    print("Filtered counts per exclusion reason and entity type:")
    for reason, counts in filtered_counts.items():
        print(f"\nReason: {reason}")
        for etype, cnum in counts.items():
            print(f"  {etype}: {cnum}")

    print("\nRows saved for each entity type:")
    for etype, cnum in saved_counts.items():
        print(f"  {etype}: {cnum}")

    conn.close()

if __name__ == '__main__':
    data_dir = Path(project_config['data_dir'])
    max_input_length = int(project_config['llm']['input_length'])
    save_datasets(data_dir / 'imdb.db', data_dir, max_input_length)
