# scripts/numeric_rep_bench/utils.py

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int):
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
