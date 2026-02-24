from typing import Optional, Dict, Any
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

    def to_string(self, value: np.ndarray) -> str:
        """
        Low-level formatting. Converts a processed numpy array (probs, indices, or values) to string.
        """
        raise NotImplementedError
    
    def render_prediction(self, prediction_tensor: torch.Tensor) -> str:
        """
        High-level wrapper. Takes raw model output (logits/normalized), processes it (sigmoid/argmax/unscale),
        and returns a string.
        """
        # Default behavior: detach, cpu, numpy, stringify
        return self.to_string(prediction_tensor.detach().cpu().numpy())

    def render_ground_truth(self, target_tensor: torch.Tensor) -> str:
        """
        High-level wrapper. Takes dataset target (indices/normalized), processes it,
        and returns a string.
        """
        # Default behavior: detach, cpu, numpy, stringify
        return self.to_string(target_tensor.detach().cpu().numpy())

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

    @property
    def is_stats_finalized(self) -> bool:
        return self._stats_finalized

    def transform(self, raw_value):
        if raw_value is None:
            if self.optional:
                return self.get_base_padding_value()
            try:
                return self._transform(raw_value)
            except Exception as exc:
                raise ValueError(f"Field '{self.name}' is not optional, but received None.") from exc
        return self._transform(raw_value)

    def transform_target(self, raw_value):
        return self.transform(raw_value)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "optional": self.optional,
            "type": self.__class__.__name__,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("name") and self.name != state["name"]:
            return
        self.optional = bool(state.get("optional", self.optional))