import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def calculate_detection_metrics(
    targets,
    probabilities,
    threshold=0.5,
):
    """
    Calculate image-level tampering detection metrics.

    Args:
        targets:
            Ground-truth labels.
            0 = Authentic
            1 = Tampered

        probabilities:
            Predicted tampering probabilities.

        threshold:
            Probability threshold for binary prediction.

    Returns:
        Dictionary containing detection metrics.
    """

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if len(targets) != len(probabilities):
        raise ValueError(
            "targets and probabilities must have "
            "the same length."
        )

    if len(targets) == 0:
        raise ValueError(
            "Cannot calculate metrics on empty data."
        )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    # --------------------------------------------------
    # Basic metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------

    if len(np.unique(targets)) >= 2:

        roc_auc = roc_auc_score(
            targets,
            probabilities,
        )

    else:

        # ROC-AUC is undefined when only one class exists.
        roc_auc = None

    # --------------------------------------------------
    # PR-AUC / Average Precision
    # --------------------------------------------------

    if len(np.unique(targets)) >= 2:

        pr_auc = average_precision_score(
            targets,
            probabilities,
        )

    else:

        pr_auc = None

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": (
            None
            if roc_auc is None
            else float(roc_auc)
        ),
        "pr_auc": (
            None
            if pr_auc is None
            else float(pr_auc)
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "threshold": float(threshold),
        "num_samples": int(len(targets)),
    }


def print_detection_metrics(metrics):
    """
    Print detection metrics in a readable format.
    """

    print()
    print("=" * 60)
    print("IMAGE-LEVEL DETECTION METRICS")
    print("=" * 60)

    print(
        f"Accuracy      : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision     : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall        : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score      : "
        f"{metrics['f1_score']:.4f}"
    )

    if metrics["roc_auc"] is not None:

        print(
            f"ROC-AUC       : "
            f"{metrics['roc_auc']:.4f}"
        )

    else:

        print(
            "ROC-AUC       : Undefined "
            "(only one class present)"
        )

    if metrics["pr_auc"] is not None:

        print(
            f"PR-AUC        : "
            f"{metrics['pr_auc']:.4f}"
        )

    else:

        print(
            "PR-AUC        : Undefined "
            "(only one class present)"
        )

    cm = metrics[
        "confusion_matrix"
    ]

    print()
    print("Confusion Matrix")
    print("----------------")

    print(
        f"True Negative  : "
        f"{cm['true_negative']}"
    )

    print(
        f"False Positive : "
        f"{cm['false_positive']}"
    )

    print(
        f"False Negative : "
        f"{cm['false_negative']}"
    )

    print(
        f"True Positive  : "
        f"{cm['true_positive']}"
    )

    print("=" * 60)