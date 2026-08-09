import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import SEED, NEURAL_NETWORK_BATCH_SIZE


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def create_seeded_data_loader(
    dataset,
    batch_size=NEURAL_NETWORK_BATCH_SIZE,
    shuffle=False,
    seed=SEED,
):
    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )