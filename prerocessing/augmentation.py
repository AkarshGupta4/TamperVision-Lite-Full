import random

import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from .image_preprocessing import (
    load_image,
    load_mask,
    resize_image,
    resize_mask,
    image_to_tensor,
    mask_to_tensor,
    normalize_image,
)


class PairedTransform:
    """
    Applies synchronized transformations to an image
    and its segmentation mask.

    Spatial transformations are applied identically
    to both image and mask.

    Photometric transformations are applied only to image.
    """

    def __init__(
        self,
        image_size=224,
        train=True,
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.2,
        rotation_prob=0.2,
        max_rotation=10,
        color_jitter_prob=0.3,
        brightness=0.15,
        contrast=0.15,
    ):

        self.image_size = image_size
        self.train = train

        self.horizontal_flip_prob = (
            horizontal_flip_prob
        )

        self.vertical_flip_prob = (
            vertical_flip_prob
        )

        self.rotation_prob = rotation_prob
        self.max_rotation = max_rotation

        self.color_jitter_prob = (
            color_jitter_prob
        )

        self.brightness = brightness
        self.contrast = contrast

    def __call__(
        self,
        image,
        mask,
    ):
        """
        Args:
            image: PIL RGB image
            mask: PIL grayscale mask

        Returns:
            image_tensor
            mask_tensor
        """

        # --------------------------------------------------
        # Resize
        # --------------------------------------------------

        image = resize_image(
            image,
            self.image_size,
        )

        mask = resize_mask(
            mask,
            self.image_size,
        )

        # --------------------------------------------------
        # Spatial augmentation
        # --------------------------------------------------

        if self.train:

            # Horizontal flip
            if random.random() < self.horizontal_flip_prob:

                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() < self.vertical_flip_prob:

                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Small rotation
            if random.random() < self.rotation_prob:

                angle = random.uniform(
                    -self.max_rotation,
                    self.max_rotation,
                )

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )

            # --------------------------------------------------
            # Photometric augmentation
            # --------------------------------------------------
            # These are applied ONLY to image.
            # Mask must never receive brightness/contrast changes.
            # --------------------------------------------------

            if random.random() < self.color_jitter_prob:

                brightness_factor = random.uniform(
                    max(0.0, 1.0 - self.brightness),
                    1.0 + self.brightness,
                )

                contrast_factor = random.uniform(
                    max(0.0, 1.0 - self.contrast),
                    1.0 + self.contrast,
                )

                image = TF.adjust_brightness(
                    image,
                    brightness_factor,
                )

                image = TF.adjust_contrast(
                    image,
                    contrast_factor,
                )

        # --------------------------------------------------
        # Convert to tensors
        # --------------------------------------------------

        image = image_to_tensor(image)

        mask = mask_to_tensor(mask)

        # --------------------------------------------------
        # Normalize only image
        # --------------------------------------------------

        image = normalize_image(image)

        return image, mask


def create_train_transform(
    image_size=224,
):
    """
    Training augmentation pipeline.
    """

    return PairedTransform(
        image_size=image_size,
        train=True,
    )


def create_validation_transform(
    image_size=224,
):
    """
    Validation/test preprocessing.

    No random augmentation.
    """

    return PairedTransform(
        image_size=image_size,
        train=False,
    )