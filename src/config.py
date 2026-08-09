# src/config.py

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# *** GLOBAL CONSTANTS ***

# `SEED`: A fixed random seed used for all random operations to make experiments reproducible.
# `REAL_LABEL`: The numeric label assigned to real images.
# `FAKE_LABEL`: The numeric label assigned to fake images.
# `CLASS_NAMES`: A list of string names corresponding to the numeric labels, useful for display purposes.
# `TRAIN_SAMPLES_PER_CLASS`: The number of samples (images) to select for each class (real and fake) in the training dataset.
# `VALID_SAMPLES_PER_CLASS`: The number of samples to select for each class in the validation dataset.
# `NEURAL_NETWORK_BATCH_SIZE`: The batch size to be used when training neural networks.

SEED = 42

REAL_LABEL = 0
FAKE_LABEL = 1
CLASS_NAMES = ["Real", "Fake"]

TRAIN_SAMPLES_PER_CLASS = 5000
VALID_SAMPLES_PER_CLASS = 1000
NEURAL_NETWORK_BATCH_SIZE = 32