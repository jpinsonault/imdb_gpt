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


### Philosophy/Purpose

This project is about memorizing the imdb dataset in a way that is useful for downstream tasks like, knowing who was in a movie, or knowing what movies a person was in and what they did in those movies.
