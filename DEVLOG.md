# Jan 2

I've gotten the textfield decoder working well, and the next step is to get the DB into the shape I want going forward - normalized and organized. Then I can work on doing joint embeddings between titles/actors or I can work on defining a project schema - a higher level construction with an entry for each table, defining its encoding fields as well as its relationships. I might do self-supervised learning on the individual rows, and then do sequence modeling of each one-to-many relationship I want to support. e.g. for each movie, produce its credits, for each actor/director produce their filmography in order

It occurs to me that if i want to list credits, there will be movies where the same actors plays 5 characters.

we can probably ignore crew because that data is also in the principal table


jan 3rd

this idea that we only use sql to store data and only use sql to query data has paid off handsomely

here's a concept

let titles_table = TableSchema(table="titles", fields=[TextField("primaryTitle"),
                                                       ScalarField("startYear", scaling=Scaling.STANDARDIZE)])

let people_table = TableSchema(table="people", fields=[TextField("primaryName"),
                                                                 ScalarField("birthYear", scaling=Scaling.STANDARDIZE)])

let people_in_movies = Relationship(for_each="tconst", in="titles)
let people_in_movies = Relationship(for_each=")

let schema = DatasetSchema(
    tables=[titles_table, principals_table], # will be autoencoded

)