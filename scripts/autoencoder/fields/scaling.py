import numpy as np
from enum import Enum

class Scaling(Enum):
    NONE = 1
    NORMALIZE = 2
    STANDARDIZE = 3
    LOG = 4

    def none_transform(self, x, **kwargs):
        return x

    def none_untransform(self, x, **kwargs):
        return x

    def normalize_transform(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val) if max_val > min_val else 0.0

    def normalize_untransform(self, x, min_val, max_val):
        return x * (max_val - min_val) + min_val

    def standardize_transform(self, x, mean_val, std_val):
        return (x - mean_val) / std_val if std_val != 0 else 0.0

    def standardize_untransform(self, x, mean_val, std_val):
        return x * std_val + mean_val

    def log_transform(self, x):
        return np.log1p(x)

    def log_untransform(self, x):
        return np.expm1(x)
