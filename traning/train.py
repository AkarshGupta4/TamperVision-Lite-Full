import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.tampering_model import TamperingModel

from training.config import TrainingConfig
from training.loss import MultiTaskLoss

# Dataset loader will be created in the next step
from dataset_loader import CASIA2Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(
    images_dir,
    masks_dir,
    image_size,
    batch_size,
    num_workers,
    train=False,
):
    dataset = CASIA2Dataset(
        image_dir=images_dir,
        mask_dir=masks_dir,
        image_size=image_size,
        train=train,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    use_amp=False,
):
    model.train()

    total_loss = 0.0
    total_detection_loss = 0.0
    total_localization_loss = 0.0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        detection_targets = (
            masks.flatten(start_dim=1).max(dim=1).values
        )

        detection_targets = detection_targets.float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            outputs = model(images)

            losses = criterion(
                detection_logits=outputs["detection_logits"],
                detection_targets=detection_targets,
                localization_logits=outputs["localization_logits"],
                localization_targets=masks,
            )

            loss = losses["total_loss"]

        if scaler is not None and use_amp:

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        total_loss += loss.item()

        total_detection_loss += (
            losses["detection_loss"].item()
        )

        total_localization_loss += (
            losses["localization_loss"].item()
        )

    num_batches = max(len(loader), 1)

    return {
        "total_loss": total_loss / num_batches,
        "detection_loss": (
            total_detection_loss / num_batches
        ),
        "localization_loss": (
            total_localization_loss / num_batches
        ),
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    use_amp=False,
):
    model.eval()

    total_loss = 0.0
    total_detection_loss = 0.0
    total_localization_loss = 0.0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        detection_targets = (
            masks.flatten(start_dim=1).max(dim=1).values
        )

        detection_targets = detection_targets.float().unsqueeze(1)

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            outputs = model(images)

            losses = criterion(
                detection_logits=outputs["detection_logits"],
                detection_targets=detection_targets,
                localization_logits=outputs["localization_logits"],
                localization_targets=masks,
            )

        total_loss += losses["total_loss"].item()

        total_detection_loss += (
            losses["detection_loss"].item()
        )

        total_localization_loss += (
            losses["localization_loss"].item()
        )

    num_batches = max(len(loader), 1)

    return {
        "total_loss": total_loss / num_batches,
        "detection_loss": (
            total_detection_loss / num_batches
        ),
        "localization_loss": (
            total_localization_loss / num_batches
        ),
    }


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    config,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "metrics": metrics,
        "config": vars(config),
    }

    torch.save(checkpoint, path)


def main():

    config = TrainingConfig()

    set_seed(config.seed)

    os.makedirs(
        config.checkpoint_dir,
        exist_ok=True,
    )

    device = torch.device(config.device)

    print("=" * 60)
    print("IMAGE TAMPERING DETECTION TRAINING")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")

    # -------------------------
    # Dataset
    # -------------------------

    train_loader = create_dataloader(
        config.train_images,
        config.train_masks,
        config.image_size,
        config.batch_size,
        config.num_workers,
        train=True,
    )

    validation_loader = create_dataloader(
        config.validation_images,
        config.validation_masks,
        config.image_size,
        config.batch_size,
        config.num_workers,
        train=False,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )

    # -------------------------
    # Model
    # -------------------------

    model = TamperingModel(
        use_srm=config.use_srm,
        use_cross_attention=config.use_cross_attention,
        use_localization=config.use_localization,
        fusion_channels=config.fusion_channels,
        pretrained=config.pretrained,
    )

    model = model.to(device)

    # -------------------------
    # Freeze backbone
    # -------------------------

    if config.freeze_backbone:

        for parameter in model.rgb_backbone.parameters():
            parameter.requires_grad = False

        print("RGB backbone frozen.")

    # -------------------------
    # Loss
    # -------------------------

    criterion = MultiTaskLoss(
        detection_weight=config.detection_loss_weight,
        localization_weight=config.localization_loss_weight,
        dice_weight=config.dice_loss_weight,
    )

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # -------------------------
    # Scheduler
    # -------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # -------------------------
    # AMP
    # -------------------------

    scaler = torch.cuda.amp.GradScaler(
        enabled=config.use_amp
    )

    best_validation_loss = float("inf")

    # -------------------------
    # Training loop
    # -------------------------

    for epoch in range(1, config.epochs + 1):

        print()
        print(
            f"Epoch {epoch}/{config.epochs}"
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=config.use_amp,
        )

        validation_metrics = validate_one_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=config.use_amp,
        )

        scheduler.step(
            validation_metrics["total_loss"]
        )

        print(
            f"Train Loss: "
            f"{train_metrics['total_loss']:.4f}"
        )

        print(
            f"Train Detection Loss: "
            f"{train_metrics['detection_loss']:.4f}"
        )

        print(
            f"Train Localization Loss: "
            f"{train_metrics['localization_loss']:.4f}"
        )

        print(
            f"Val Loss: "
            f"{validation_metrics['total_loss']:.4f}"
        )

        print(
            f"Val Detection Loss: "
            f"{validation_metrics['detection_loss']:.4f}"
        )

        print(
            f"Val Localization Loss: "
            f"{validation_metrics['localization_loss']:.4f}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if (
            validation_metrics["total_loss"]
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_metrics["total_loss"]
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    "train": train_metrics,
                    "validation": validation_metrics,
                },
                config=config,
                path=config.best_checkpoint,
            )

            print(
                "✓ Best model saved."
            )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best checkpoint: "
        f"{config.best_checkpoint}"
    )


if __name__ == "__main__":
    main()