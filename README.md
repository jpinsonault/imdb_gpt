# imdb_gpt
`imdb_gpt` is an IMDB search engine trained on the public IMDB dataset

## Steps to Train the Model

### Install requirements
`pip install -r requirements.txt`

### Optional: Configure build settings
Edit `config.py` to suit your needs

### Download IMDB dataset
Download the .tsv files from https://datasets.imdbws.com/

`python ./scripts/download_imdb_dataset.py`

- `name.basics.tsv` - All the names of cast and crew
- `title.akas.tsv` - Alternative names and foreign versions of titles
- `title.basics.tsv` - Basic info for all movies, shows, episodes
- `title.crew.tsv` - Writers and Directors by title
- `title.episode.tsv` - Maps episodes to shows, includes season/episode number
- `title.principals.tsv` - All cast and crew by title, includes character names
- `title.ratings.tsv` - Ratings and number of votes by title



### Load TSVs into sqlite
Sanitize and process the TSVs into corresponding sql tables

`python ./scripts/load_tsvs.py`

### Create Search Tables
To facilitate the generation and caching of search results, we create a new set of 'search' tables which join relevant information from the IMDB tables for quick access.

`python ./scripts/create_search_tables.py`

### Train Autoencoder
We train an autoencoder on all the unique entity names (people, titles, characters) in the dataset, so that we can embed them into a high dimensional space for search.

`python ./scripts/train_imdb_autoencoder.py`

### Create Vector Database for Search
Now that we have the encoder, we embed every entity name into a high dimensional vector space and insert them into an `annoy` vector database for quick nearest neighbor search.

The entity names are sorted and assigned unique integer IDs. 

`python ./scripts/create_vector_db.py`

### Map Vector IDs to sqlite rows
We update the sqlite search tables with the unique entity IDs.

This completes the creation of the search database. We can now be given a string, embed it, find it's nearest neighbors, and look them up in the sqlite database.

`python ./scripts/map_vector_ids_to_sqlite.py`