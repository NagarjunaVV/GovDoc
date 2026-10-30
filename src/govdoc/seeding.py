"""Deterministic seeding utilities across python random, numpy, and torch."""

import os
import random
import numpy as np
import torch
from govdoc.config import load_config


def set_seed(seed: int | None = None) -> int:
    """Set seeds for python random, numpy, and torch (CPU & CUDA) deterministically."""
    if seed is None:
        cfg = load_config()
        seed = int(cfg.get("seeds", {}).get("default", 42))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


if __name__ == "__main__":
    s = set_seed()
    print(f"Random seed successfully set to: {s}")
