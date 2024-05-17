import random
import torch
import numpy as np
from json import load

def read_json(f : str) -> dict:

    config = None

    with open(f, "r") as f:
        config = load(f)

    return config


def seed_everything(seed : int = 6):

    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False