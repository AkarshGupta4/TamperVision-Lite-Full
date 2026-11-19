import numpy as np


def _validate_masks(
    targets,
    predictions,
):
    """
    Validate target and predicted masks.

    Expected shape:
        N x H x W

    or:
        N x 1 x H x W
    """

    targets = np.asarray(
        targets
    )

    predictions = np.asarray(
        predictions
    )

    if targets.shape != predictions.shape:
        raise ValueError(
            "Ground-truth masks and predicted "
            "masks must have the same shape. "
            f"Got {targets.shape} and "
            f"{predictions.shape}."
        )

    if targets.ndim == 4:

        if targets.shape[1] != 1:

            raise ValueError(
                "4D masks must have one channel."
            )

        targets = targets[:, 0]
        predictions = predictions[:, 0]

    if targets.ndim != 3:

        raise ValueError(
            "Masks must have shape "
            "N x H x W or N x 1 x H x W."
        )

    return (
        targets.astype(bool),
        predictions.astype(bool),
    )


def calculate_iou(
    target,
    prediction,
):
    """
    Calculate IoU for a single mask.

    Handles empty authentic masks.

    Cases:

        target empty + prediction empty
            -> 1.0

        target empty + prediction non-empty
            -> 0.0

        target non-empty
            -> normal IoU
    """

    target = np.asarray(
        target,
        dtype=bool,
    )

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    intersection = np.logical_and(
        target,
        prediction,
    ).sum()

    union = np.logical_or(
        target,
        prediction,
    ).sum()

    if union == 0:

        return 1.0

    return float(
        intersection / union
    )


def calculate_dice(
    target,
    prediction,
):
    """
    Calculate Dice score for a single mask.

    Handles empty masks safely.
    """

    target = np.asarray(
        target,
        dtype=bool,
    )

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    target_sum = target.sum()
    prediction_sum = prediction.sum()

    intersection = np.logical_and(
        target,
        prediction,
    ).sum()

    denominator = (
        target_sum + prediction_sum
    )

    if denominator == 0:

        return 1.0

    return float(
        2.0 * intersection / denominator
    )


def calculate_pixel_precision(
    target,
    prediction,
):
    """
    Pixel-level precision.
    """

    target = np.asarray(
        target,
        dtype=bool,
    )

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    true_positive = np.logical_and(
        target,
        prediction,
    ).sum()

    false_positive = np.logical_and(
        ~target,
        prediction,
    ).sum()

    denominator = (
        true_positive + false_positive
    )

    if denominator == 0:

        # No predicted positive pixels.
        # If ground truth also has no positives,
        # this is a correct empty prediction.
        if target.sum() == 0:
            return 1.0

        return 0.0

    return float(
        true_positive / denominator
    )


def calculate_pixel_recall(
    target,
    prediction,
):
    """
    Pixel-level recall.
    """

    target = np.asarray(
        target,
        dtype=bool,
    )

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    true_positive = np.logical_and(
        target,
        prediction,
    ).sum()

    false_negative = np.logical_and(
        target,
        ~prediction,
    ).sum()

    denominator = (
        true_positive + false_negative
    )

    if denominator == 0:

        # No positive ground-truth pixels.
        # Recall is treated as 1 when prediction
        # is also empty, otherwise 0.
        if prediction.sum() == 0:
            return 1.0

        return 0.0

    return float(
        true_positive / denominator
    )


def calculate_localization_metrics(
    targets,
    predictions,
):
    """
    Calculate localization metrics over a dataset.

    Args:
        targets:
            N x H x W

        predictions:
            N x H x W

    Returns:
        Dictionary containing:

            mean_iou
            mean_dice
            pixel_precision
            pixel_recall
    """

    targets, predictions = _validate_masks(
        targets,
        predictions,
    )

    num_samples = targets.shape[0]

    if num_samples == 0:
        raise ValueError(
            "Cannot calculate localization "
            "metrics on empty data."
        )

    ious = []
    dices = []
    precisions = []
    recalls = []

    # --------------------------------------------------
    # Per-image metrics
    # --------------------------------------------------

    for index in range(num_samples):

        target = targets[index]
        prediction = predictions[index]

        ious.append(
            calculate_iou(
                target,
                prediction,
            )
        )

        dices.append(
            calculate_dice(
                target,
                prediction,
            )
        )

        precisions.append(
            calculate_pixel_precision(
                target,
                prediction,
            )
        )

        recalls.append(
            calculate_pixel_recall(
                target,
                prediction,
            )
        )

    # --------------------------------------------------
    # Dataset-level micro metrics
    # --------------------------------------------------

    target_flat = targets.reshape(-1)
    prediction_flat = predictions.reshape(-1)

    true_positive = np.logical_and(
        target_flat,
        prediction_flat,
    ).sum()

    false_positive = np.logical_and(
        ~target_flat,
        prediction_flat,
    ).sum()

    false_negative = np.logical_and(
        target_flat,
        ~prediction_flat,
    ).sum()

    intersection = true_positive

    union = np.logical_or(
        target_flat,
        prediction_flat,
    ).sum()

    total_target = target_flat.sum()
    total_prediction = prediction_flat.sum()

    # Micro IoU.
    if union == 0:

        micro_iou = 1.0

    else:

        micro_iou = (
            intersection / union
        )

    # Micro Dice.
    dice_denominator = (
        total_target
        + total_prediction
    )

    if dice_denominator == 0:

        micro_dice = 1.0

    else:

        micro_dice = (
            2.0 * intersection
            / dice_denominator
        )

    # Micro precision.
    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:

        micro_precision = (
            1.0
            if total_target == 0
            else 0.0
        )

    else:

        micro_precision = (
            true_positive
            / precision_denominator
        )

    # Micro recall.
    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:

        micro_recall = (
            1.0
            if total_prediction == 0
            else 0.0
        )

    else:

        micro_recall = (
            true_positive
            / recall_denominator
        )

    return {
        # Mean per-image metrics.
        "mean_iou": float(
            np.mean(ious)
        ),

        "mean_dice": float(
            np.mean(dices)
        ),

        "mean_pixel_precision": float(
            np.mean(precisions)
        ),

        "mean_pixel_recall": float(
            np.mean(recalls)
        ),

        # Dataset-level metrics.
        "micro_iou": float(
            micro_iou
        ),

        "micro_dice": float(
            micro_dice
        ),

        "micro_pixel_precision": float(
            micro_precision
        ),

        "micro_pixel_recall": float(
            micro_recall
        ),

        "num_images": int(
            num_samples
        ),
    }


def print_localization_metrics(
    metrics,
):
    """
    Print localization metrics.
    """

    print()
    print("=" * 60)
    print("PIXEL-LEVEL LOCALIZATION METRICS")
    print("=" * 60)

    print(
        f"Mean IoU              : "
        f"{metrics['mean_iou']:.4f}"
    )

    print(
        f"Mean Dice             : "
        f"{metrics['mean_dice']:.4f}"
    )

    print(
        f"Mean Pixel Precision  : "
        f"{metrics['mean_pixel_precision']:.4f}"
    )

    print(
        f"Mean Pixel Recall     : "
        f"{metrics['mean_pixel_recall']:.4f}"
    )

    print()
    print("Micro Metrics")
    print("-------------")

    print(
        f"Micro IoU             : "
        f"{metrics['micro_iou']:.4f}"
    )

    print(
        f"Micro Dice            : "
        f"{metrics['micro_dice']:.4f}"
    )

    print(
        f"Micro Pixel Precision : "
        f"{metrics['micro_pixel_precision']:.4f}"
    )

    print(
        f"Micro Pixel Recall    : "
        f"{metrics['micro_pixel_recall']:.4f}"
    )

    print("=" * 60)