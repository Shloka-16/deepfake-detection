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

REAL_LABEL = 0
FAKE_LABEL = 1
CLASS_NAMES = ["Real", "Fake"]

NEURAL_NETWORK_BATCH_SIZE = 32

from pathlib import Path
import numpy as np

# -----------------------------
# Dataset and output locations
# -----------------------------

KAGGLE_DATASET_NAME = "xhlulu/140k-real-and-fake-faces"
DATASET_DOWNLOAD_DIRECTORY = Path("/content")
EXPECTED_DATASET_ROOT = DATASET_DOWNLOAD_DIRECTORY / "real_vs_fake" / "real-vs-fake"

PROJECT_OUTPUT_DIRECTORY = Path(
    "/content/drive/MyDrive/deepfake_detection_checkpoints"
)

CHECKPOINT_PATH = PROJECT_OUTPUT_DIRECTORY / "best_resnet50.pth"
HISTORY_PATH = PROJECT_OUTPUT_DIRECTORY / "resnet50_training_history.csv"
SAMPLE_MANIFEST_PATH = PROJECT_OUTPUT_DIRECTORY / "resnet50_sample_manifest.csv"

VALIDATION_METRICS_PATH = PROJECT_OUTPUT_DIRECTORY / "resnet50_validation_metrics.csv"
VALIDATION_PREDICTIONS_PATH = PROJECT_OUTPUT_DIRECTORY / "resnet50_validation_predictions.npz"

TEST_METRICS_PATH = PROJECT_OUTPUT_DIRECTORY / "resnet50_final_test_metrics.csv"
TEST_PREDICTIONS_PATH = PROJECT_OUTPUT_DIRECTORY / "resnet50_final_test_predictions.npz"

# -----------------------------
# Reproducibility and labels
# -----------------------------

SEED = 13
REAL_LABEL = 0
FAKE_LABEL = 1
CLASS_NAMES = ["Real", "Fake"]

# -----------------------------
# Data sampling and loading
# -----------------------------

TRAIN_SAMPLES_PER_CLASS = 50_000
VALID_SAMPLES_PER_CLASS = 10_000

BATCH_SIZE = 32
NUM_WORKERS = 2
RESNET_IMAGE_SIZE = (224, 224)

RESNET_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
RESNET_NORMALIZE_STD = [0.229, 0.224, 0.225]

# -----------------------------
# Model and optimization
# -----------------------------

MAX_EPOCHS = 15
HEAD_ONLY_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 3

HEAD_HIDDEN_UNITS = 256
HEAD_DROPOUT = 0.30

HEAD_LEARNING_RATE = 5e-4
LAYER4_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

MONITOR_METRIC = "Fake F1"
SUPPORTED_MONITOR_METRICS = {
    "Accuracy", "Fake precision", "Fake recall", "Fake F1",
    "Real F1", "Macro F1", "Weighted F1", "ROC-AUC", "PR-AUC",
}

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)

# -----------------------------
# Run controls
# -----------------------------

RUN_TRAINING = True
RUN_TEST_EVALUATION = True
FORCE_RERUN_TEST = True