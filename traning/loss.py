import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.contiguous().view(
            probabilities.size(0), -1
        )

        targets = targets.contiguous().view(
            targets.size(0), -1
        )

        intersection = (probabilities * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class LocalizationLoss(nn.Module):
    """
    BCEWithLogits + Dice loss.
    """

    def __init__(self, dice_weight: float = 1.0):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = bce_loss + self.dice_weight * dice_loss

        return total_loss


class MultiTaskLoss(nn.Module):
    """
    Combined detection + localization loss.

    Total Loss =
        detection_weight * detection_loss
        +
        localization_weight * localization_loss
    """

    def __init__(
        self,
        detection_weight: float = 1.0,
        localization_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()

        self.detection_weight = detection_weight
        self.localization_weight = localization_weight

        self.detection_loss = nn.BCEWithLogitsLoss()

        self.localization_loss = LocalizationLoss(
            dice_weight=dice_weight
        )

    def forward(
        self,
        detection_logits,
        detection_targets,
        localization_logits,
        localization_targets,
    ):
        # Detection
        detection_loss = self.detection_loss(
            detection_logits,
            detection_targets,
        )

        # Localization
        localization_loss = self.localization_loss(
            localization_logits,
            localization_targets,
        )

        # Total
        total_loss = (
            self.detection_weight * detection_loss
            +
            self.localization_weight * localization_loss
        )

        return {
            "total_loss": total_loss,
            "detection_loss": detection_loss,
            "localization_loss": localization_loss,
        }