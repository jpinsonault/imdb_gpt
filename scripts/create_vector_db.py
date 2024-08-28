from pathlib import Path
from annoy import AnnoyIndex
import numpy as np
from scripts import utils
from train_imdb_autoencoder import SPECIAL_PAD, save_corpus_stats, try_load_model, TokenProcessor, get_corpus_stats
from tqdm import tqdm
from tensorflow.keras.models import Model




def save_embeddings(model, token_processor: TokenProcessor, corpus_file_path, embeddings_path):
    embeddings = []
    entity_names = []
    
    batch_size = 8000

    batch_tokenized = []
    with open(corpus_file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines(), desc="Processing corpus"):
            line = line.strip()
            if not line:
                continue

            tokenized = token_processor.tokenize(line)
            batch_tokenized.append(tokenized)

            if len(batch_tokenized) == batch_size:
                # Predict in batches
                batch_embeddings = model.predict(np.array(batch_tokenized), verbose=0)
                embeddings.extend(batch_embeddings)
                batch_tokenized = []  # Reset the batch

    # Process the remaining batch
    if batch_tokenized:
        batch_embeddings = model.predict(np.array(batch_tokenized))
        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)

    # Build Annoy index
    annoy_index = AnnoyIndex(embeddings.shape[1], 'angular')
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    annoy_index.save(embeddings_path + '.ann')


def search_similar_entities(query, token_processor, model, annoy_index_path, entity_names_path, num_results=10):
    tokenized_query = token_processor.tokenize(query)
    query_embedding = model.predict(tokenized_query)

    annoy_index = AnnoyIndex(len(query_embedding), 'angular')
    annoy_index.load(annoy_index_path)

    nearest_ids = annoy_index.get_nns_by_vector(query_embedding, num_results, include_distances=True)

    with open(entity_names_path, 'r') as file:
        entity_names = file.readlines()

    results = [(entity_names[i].strip(), distance) for i, distance in zip(*nearest_ids)]
    return results


if __name__ == '__main__':
    corpus_file_path = Path("./entities.txt")
    stats_file_path = Path("./corpus_stats.json")
    
    utils.verify_destructive("This will destroy the vector database. Are you sure?")
    
    max_input_length = 96
    
    alphabet, counts = get_corpus_stats(stats_file_path)
    alphabet = alphabet + [SPECIAL_PAD] 

    char_to_index = {char: index for index, char in enumerate(alphabet)}
    index_to_char = {index: char for index, char in enumerate(alphabet)}

    model_path = "models/imdb_autoencoder.h5"
    full_model = try_load_model(model_path)
    
    encoder_layer = full_model.layers[26]
    
    encoder_model = Model(inputs=full_model.input, outputs=encoder_layer.output)
    
    token_processor = TokenProcessor(char_to_index=char_to_index, max_input_length=max_input_length)
    
    embeddings_path = "models/imdb_autoencoder_embeddings.npy"
    save_embeddings(encoder_model, token_processor, corpus_file_path, embeddings_path)
    
    annoy_index_path = embeddings_path + '.ann'
    
    entity_names_path = "models/imdb_autoencoder_embeddings.npy_names.txt"
    query = "The Godfather"
    results = search_similar_entities(query, token_processor, encoder_model, annoy_index_path, entity_names_path)
    print(f"Results for query '{query}':")
    for result in results:
        print(result)
        
    query = "The Godfather: Part II"
    results = search_similar_entities(query, token_processor, encoder_model, annoy_index_path, entity_names_path)
    print(f"Results for query '{query}':")
    for result in results:
        print(result)
        
    