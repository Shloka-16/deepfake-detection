import random

import numpy as np

from src.config import REAL_LABEL, FAKE_LABEL, SEED


def make_balanced_sample(
    real_paths,
    fake_paths,
    samples_per_class,
    seed=SEED,
):
    """
    Select equal numbers of real and fake paths and shuffle them together.
    """

    # Make sure both classes contain enough images.
    if len(real_paths) < samples_per_class or len(fake_paths) < samples_per_class:
        raise ValueError("The requested sample is larger than one of the classes.")

    # Create an independent seeded generator for path selection.
    python_rng = random.Random(seed)

    # Select unique real and fake paths.
    selected_real_paths = python_rng.sample(
        list(real_paths),
        samples_per_class,
    )
    selected_fake_paths = python_rng.sample(
        list(fake_paths),
        samples_per_class,
    )

    # Put the real paths first and fake paths second.
    selected_paths = selected_real_paths + selected_fake_paths

    # Create matching labels: Real = 0 and Fake = 1.
    selected_targets = np.array(
        [REAL_LABEL] * samples_per_class + [FAKE_LABEL] * samples_per_class,
        dtype=np.int64,
    )

    # Create one reproducible shuffled list of positions.
    shuffle_order = np.random.default_rng(seed).permutation(len(selected_paths))

    # Apply the same order to both the paths and labels.
    shuffled_paths = [selected_paths[index] for index in shuffle_order]
    shuffled_targets = selected_targets[shuffle_order]

    return shuffled_paths, shuffled_targets

