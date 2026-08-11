# src/data_paths.py
from pathlib import Path


def get_dataset_directories(dataset_root):
    return {
        "train_real": dataset_root / "train" / "real",
        "train_fake": dataset_root / "train" / "fake",
        "valid_real": dataset_root / "valid" / "real",
        "valid_fake": dataset_root / "valid" / "fake",
        "test_real": dataset_root / "test" / "real",
        "test_fake": dataset_root / "test" / "fake",
    }


def list_image_paths(directory):
    """Return sorted image paths from one class directory."""
    return sorted(
        path
        for path in directory.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )