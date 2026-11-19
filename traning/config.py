from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    # -------------------------
    # Dataset
    # -------------------------
    train_images: str = "dataset/train/images"
    train_masks: str = "dataset/train/masks"

    validation_images: str = "dataset/validation/images"
    validation_masks: str = "dataset/validation/masks"

    test_images: str = "dataset/test/images"
    test_masks: str = "dataset/test/masks"

    image_size: int = 224

    # -------------------------
    # Training
    # -------------------------
    batch_size: int = 16
    num_workers: int = 2

    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # -------------------------
    # Loss weights
    # -------------------------
    detection_loss_weight: float = 1.0
    localization_loss_weight: float = 1.0
    dice_loss_weight: float = 1.0

    # -------------------------
    # Model
    # -------------------------
    use_srm: bool = True
    use_cross_attention: bool = True
    use_localization: bool = True

    fusion_channels: int = 128

    # -------------------------
    # Checkpoint
    # -------------------------
    checkpoint_dir: str = "checkpoints"
    best_checkpoint: str = "checkpoints/best_model.pth"

    # -------------------------
    # Training options
    # -------------------------
    pretrained: bool = True
    freeze_backbone: bool = False

    # -------------------------
    # Reproducibility
    # -------------------------
    seed: int = 42

    # -------------------------
    # Device
    # -------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # AMP
    # -------------------------
    use_amp: bool = torch.cuda.is_available()