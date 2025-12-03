import torch
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import project_config
from scripts.simple_set.model import HybridSetModel
from scripts.simple_set.data import HybridSetDataset
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder

# Ensure logging is configured
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MovieSearchEngine:
    # Heuristic averages for fields that were mandatory during training.
    # Providing 'None' (padding) for these throws the model Out-Of-Distribution
    # because it never saw missing values for them.
    DEFAULTS = {
        "startYear": 2000,
        "runtimeMinutes": 100,
        "averageRating": 6.5,
        "numVotes": 10000,
        "peopleCount": 15,
        "isAdult": 0,
        "tconst": 0, # Provide a dummy ID so it looks like a valid input
    }

    def __init__(self, device: Optional[str] = None):
        self.cfg = project_config
        
        # Detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logging.info(f"Initializing Search Engine on {self.device}...")

        # 1. Load Dataset Cache
        # We need this for field definitions and to decode the search results back to text
        cache_path = Path(self.cfg.data_dir) / "hybrid_set_cache.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found at {cache_path}. Run training/precompute first.")
        
        self.dataset = HybridSetDataset(str(cache_path), self.cfg)
        self.num_movies = len(self.dataset)
        logging.info(f"Loaded dataset with {self.num_movies} movies.")

        # 2. Load Model
        # We construct the model structure and load weights
        self.model = HybridSetModel(
            fields=self.dataset.fields,
            num_people=self.dataset.num_people,
            heads_config=self.cfg.hybrid_set_heads,
            head_vocab_sizes=self.dataset.head_vocab_sizes,
            latent_dim=int(self.cfg.hybrid_set_latent_dim),
            hidden_dim=int(self.cfg.hybrid_set_hidden_dim),
            base_output_rank=int(self.cfg.hybrid_set_output_rank),
            depth=int(self.cfg.hybrid_set_depth),
            dropout=0.0, # No dropout for inference
            num_movies=self.num_movies
        )

        ckpt_path = Path(self.cfg.model_dir) / "hybrid_set_state.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            
        logging.info(f"Loading weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu") # Load to CPU first
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # 3. Optimize for Inference
        self.model.eval()
        self.model.to(self.device)
        
        # Pre-fetch the embedding table to the compute device for fast search
        # If GPU VRAM is tight, you can keep this on CPU
        self.embedding_table = self.model.movie_embeddings.weight.detach().to(self.device)
        
        logging.info("Search Engine Ready.")

    @torch.no_grad()
    def encode_query(self, user_input: Dict[str, Any]) -> torch.Tensor:
        """
        Transforms user input dictionary into a latent vector.
        Missing fields are filled with DEFAULTS (average values) if available,
        otherwise specific padding values.
        """
        field_tensors = []
        
        for f in self.dataset.fields:
            raw_val = user_input.get(f.name)
            
            # IMPUTATION STRATEGY:
            # If the user didn't provide a value, and we have a reasonable default
            # (central tendency) for a field that was mandatory in training, use it.
            # Otherwise, the model sees a "padding/mask" token it never learned, resulting in garbage latents.
            if raw_val is None or raw_val == "":
                if f.name in self.DEFAULTS:
                    raw_val = self.DEFAULTS[f.name]
            
            if raw_val is not None:
                # Transform provided/imputed value
                try:
                    # transform returns a tensor, usually on CPU. Move to device.
                    t = f.transform(raw_val).to(self.device)
                except Exception as e:
                    logging.warning(f"Error transforming field {f.name} with value {raw_val}: {e}. Falling back to padding.")
                    t = f.get_base_padding_value().to(self.device)
            else:
                # Use specific padding for truly optional/missing fields (like EndYear)
                t = f.get_base_padding_value().to(self.device)
            
            # Add batch dimension (B=1)
            field_tensors.append(t.unsqueeze(0))
            
        # Run the encoder
        # The model expects a list of tensors [Field1(B, ...), Field2(B, ...)]
        z_enc = self.model.field_encoder(field_tensors)
        return z_enc

    @torch.no_grad()
    def search(self, query_dict: Dict[str, Any], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        1. Encodes the query_dict.
        2. Finds nearest neighbors in the embedding table (L2 distance).
        3. Decodes the results from the dataset.
        """
        t0 = time.perf_counter()
        
        # A. Encode
        z_query = self.encode_query(query_dict) # (1, Latent)
        
        # B. Brute Force Search (L2 Distance)
        # Dist = (x - y)^2 = x^2 + y^2 - 2xy
        # z_query: (1, D), embedding_table: (N, D)
        dists = torch.cdist(z_query, self.embedding_table, p=2).squeeze(0) # Result: (N,)
        
        # C. Top K
        # largest=False because smaller distance is better
        values, indices = torch.topk(dists, k=top_k, largest=False)
        
        indices_cpu = indices.cpu().numpy()
        distances_cpu = values.cpu().numpy()
        
        results = []
        
        # D. Decode Metadata
        # We look up the ground truth fields from the dataset using the index
        for rank, (idx, dist) in enumerate(zip(indices_cpu, distances_cpu)):
            decoded_row = {}
            decoded_row["_score"] = float(dist)
            decoded_row["_rank"] = rank + 1
            decoded_row["_id"] = int(idx)
            
            # Retrieve raw tensors from dataset cache
            # dataset.stacked_fields is a List[Tensor(N, ...)]
            for f_idx, field in enumerate(self.dataset.fields):
                # Grab the specific row for this field
                val_tensor = self.dataset.stacked_fields[f_idx][idx]
                
                # Render back to string
                # We use render_ground_truth because these are stored target indices/values
                human_readable = field.render_ground_truth(val_tensor)
                decoded_row[field.name] = human_readable
            
            results.append(decoded_row)
            
        duration = (time.perf_counter() - t0) * 1000.0
        logging.info(f"Search for '{query_dict.get('primaryTitle', '???')}' took {duration:.2f}ms")
        
        return results

if __name__ == "__main__":
    # Simple CLI test
    engine = MovieSearchEngine()
    
    q = "Matrix"
    print(f"\nSearching for: {q}")
    hits = engine.search({"primaryTitle": q}, top_k=5)
    
    for h in hits:
        print(f"[{h['_score']:.4f}] {h['primaryTitle']} ({h['startYear']})")