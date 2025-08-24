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


### TODO Finish 

### Philosophy/Purpose

This project is about memorizing the imdb dataset in a way that is useful for downstream tasks like, knowing who was in a movie, or knowing what movies a person was in and what they did in those movies.

This is more or less a database query, and it's useful to think of the objects in this project as database rows. Every person and movie is just a tuple of values (the matrix, 1999, action, 9.2/10)

In the code, the columns in the imdb data are called `Field`s. You can define any kind of field you want, the project includes `TextField`, `CategoryField`, `NumericDigitCategoryField`, `ScalarField`. Fields have a standard interface that allows them to define a mapping between the raw inputs from the dataset to tensor encodings, as well as build a small custom sub-model that's dedicated to that particlar type. For example the text encoder uses a convolutional network, the category field uses dense layers.

To perform useful downstream tasks, the data in these imdb tables ought to be projected into an compressed latent embedding. The tables can be auto-encoded, each row shrunk down and then reconstructed. This is neat, but a movie isn't very interesting on its own.

To do more interesting stuff, it would be useful to embed movies and the people in them near each other. `train_joint_autoencoder.py` attempts to do this, by using NCE loss. Essentially movies and the people in them are gently nudged toward each other, and every other entity is pushed away.

This should give a boost when trying to distinguish between similarly named movies, as more effort will be put into teasing out info from the other fields like genre and release year, in order to say "this is the Matrix movie with the original Oracle actress and this is the sequel with the replacement actress"