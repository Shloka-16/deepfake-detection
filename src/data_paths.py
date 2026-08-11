# src/data_paths.py

import subprocess
from pathlib import Path

from google.colab import userdata


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

def download_kaggle_dataset(
    dataset_root,
    dataset_name="xhlulu/140k-real-and-fake-faces",
    download_directory=Path("/content"),
):
    """Download and unzip the Kaggle dataset into the Colab runtime."""
    if dataset_root.exists():
        print(f"Dataset already found at: {dataset_root}")
        return

    kaggle_token = userdata.get("KAGGLE_API_TOKEN")
    if not kaggle_token:
        raise ValueError("KAGGLE_API_TOKEN was not found in Colab Secrets.")

    kaggle_directory = Path.home() / ".kaggle"
    kaggle_directory.mkdir(parents=True, exist_ok=True)

    token_path = kaggle_directory / "access_token"
    token_path.write_text(kaggle_token)
    token_path.chmod(0o600)

    print(f"Downloading Kaggle dataset: {dataset_name}")

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_name,
            "--unzip",
            "-p",
            str(download_directory),
        ],
        check=True,
    )
    print("Dataset download and extraction complete.")