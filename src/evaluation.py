from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import FAKE_LABEL

def calculate_classification_metrics(
    experiment_name,
    true_targets,
    predicted_targets,
):
    """
    Calculates and returns the main classification metrics for one experiment.

        Real = 0
        Fake = 1

    Note: Fake is treated as the positive class when calculating precision, recall, and F1 score.

    Parameters
    ----------
    experiment_name : str
        A descriptive name for the model or experiment.

    true_targets : array-like
        The correct labels from the validation dataset.

    predicted_targets : array-like
        The labels predicted by the model.

    Returns
    -------
    dict
        A dictionary containing the experiment name, validation-set size,
        accuracy, fake precision, fake recall, and fake F1 score.
    """
    return {
        "Experiment": experiment_name,
        "Evaluation split": "Official validation sample",
        "Validation images": int(len(true_targets)),
        "Accuracy": accuracy_score(true_targets, predicted_targets),
        "Fake precision": precision_score(
            true_targets,
            predicted_targets,
            pos_label=FAKE_LABEL,
            zero_division=0,
        ),
        "Fake recall": recall_score(
            true_targets,
            predicted_targets,
            pos_label=FAKE_LABEL,
            zero_division=0,
        ),
        "Fake F1": f1_score(
            true_targets,
            predicted_targets,
            pos_label=FAKE_LABEL,
            zero_division=0,
        ),
    }
