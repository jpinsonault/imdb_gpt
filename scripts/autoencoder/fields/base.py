from typing import Optional
import numpy as np
import torch
import torch.nn as nn

class BaseField:
    def __init__(self, name: str, optional: bool = False):
        self.name = name
        self.optional = optional
        self._stats_finalized = False

    def _get_input_shape(self):
        raise NotImplementedError

    def _get_output_shape(self):
        raise NotImplementedError

    def _accumulate_stats(self, raw_value):
        raise NotImplementedError

    def _finalize_stats(self):
        raise NotImplementedError

    def _transform(self, raw_value):
        raise NotImplementedError

    def build_encoder(self, latent_dim: int) -> nn.Module:
        raise NotImplementedError

    def build_decoder(self, latent_dim: int) -> nn.Module:
        raise NotImplementedError

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        raise NotImplementedError

    def print_stats(self):
        raise NotImplementedError

    def get_base_padding_value(self):
        raise NotImplementedError

    def get_flag_padding_value(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        return self._get_input_shape()

    @property
    def output_shape(self):
        return self._get_output_shape()

    @property
    def weight(self):
        return self._get_weight()

    def _get_weight(self):
        return 1.0

    def accumulate_stats(self, raw_value):
        self._accumulate_stats(raw_value)

    def finalize_stats(self):
        if self._stats_finalized:
            return
        self._finalize_stats()
        self._stats_finalized = True

    def stats_finalized(self) -> bool:
        return self._stats_finalized

    def transform(self, raw_value):
        if raw_value is None:
            if not self.optional:
                raise ValueError(f"Field '{self.name}' is not optional, but received None.")
            return self.get_base_padding_value()
        return self._transform(raw_value)

    def transform_target(self, raw_value):
        return self.transform(raw_value)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
