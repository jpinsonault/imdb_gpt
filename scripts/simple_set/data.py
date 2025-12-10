import logging

import torch
from torch.utils.data import Dataset

from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.row_autoencoder import _apply_field_state


class HybridSetDataset(Dataset):
    def __init__(self, cache_path, cfg):
        super().__init__()
        logging.info(f"Loading movie hybrid dataset from {cache_path}...")
        data = torch.load(cache_path, map_location="cpu")

        self.stacked_fields = data["stacked_fields"]
        self.heads_padded = data["heads_padded"]
        self.head_mappings = data.get("head_mappings", {})
        self.head_vocab_sizes = data.get("head_vocab_sizes", {})
        self.head_local_to_global = data.get("head_local_to_global", {})

        if torch.cuda.is_available():
            logging.info("Pinning movie dataset memory...")
            self.stacked_fields = [t.pin_memory() for t in self.stacked_fields]
            self.heads_padded = {k: v.pin_memory() for k, v in self.heads_padded.items()}
            self.head_mappings = {k: v.pin_memory() for k, v in self.head_mappings.items()}
            self.head_local_to_global = {k: v.pin_memory() for k, v in self.head_local_to_global.items()}

        self.num_people = data["num_people"]
        self.num_movies = data.get("num_movies", self.stacked_fields[0].shape[0])

        self.idx_to_name = data["idx_to_person_name"]

        temp_ae = TitlesAutoencoder(cfg)
        self.fields = temp_ae.fields

        field_configs = data["field_configs"]
        for f in self.fields:
            if f.name in field_configs:
                _apply_field_state(f, field_configs[f.name])

        self.num_items = self.stacked_fields[0].shape[0]

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return idx


class PersonHybridSetDataset(Dataset):
    def __init__(self, cache_path, cfg):
        super().__init__()
        logging.info(f"Loading person hybrid dataset from {cache_path}...")
        data = torch.load(cache_path, map_location="cpu")

        self.stacked_fields = data["person_stacked_fields"]
        self.heads_padded = data["person_heads_padded"]
        self.head_mappings = data.get("person_head_mappings", {})
        self.head_vocab_sizes = data.get("person_head_vocab_sizes", {})
        self.head_local_to_global = data.get("person_head_local_to_global", {})

        if torch.cuda.is_available():
            logging.info("Pinning person dataset memory...")
            self.stacked_fields = [t.pin_memory() for t in self.stacked_fields]
            self.heads_padded = {k: v.pin_memory() for k, v in self.heads_padded.items()}
            self.head_mappings = {k: v.pin_memory() for k, v in self.head_mappings.items()}
            self.head_local_to_global = {k: v.pin_memory() for k, v in self.head_local_to_global.items()}

        self.num_movies = data.get("num_movies")
        self.num_people = data["num_people"]
        self.idx_to_name = data["idx_to_person_name"]

        temp_ae = PeopleAutoencoder(cfg)
        self.fields = temp_ae.fields

        field_configs = data["person_field_configs"]
        for f in self.fields:
            if f.name in field_configs:
                _apply_field_state(f, field_configs[f.name])

        self.num_items = self.stacked_fields[0].shape[0]

    @property
    def person_head_local_to_global(self):
        return self.head_local_to_global

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return idx


class FastInfiniteLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = torch.arange(len(dataset))
        if self.shuffle:
            self.indices = self.indices[torch.randperm(len(dataset))]

        self.ptr = 0
        self.n = len(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > self.n:
            if self.shuffle:
                self.indices = self.indices[torch.randperm(self.n)]
            self.ptr = 0

        batch_idx = self.indices[self.ptr : self.ptr + self.batch_size]
        self.ptr += self.batch_size

        inputs = [t[batch_idx] for t in self.dataset.stacked_fields]
        heads_padded_batch = {k: v[batch_idx] for k, v in self.dataset.heads_padded.items()}

        return inputs, heads_padded_batch, batch_idx

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def collate_hybrid_set(batch_indices, dataset):
    pass
