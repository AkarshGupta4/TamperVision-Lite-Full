from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


# ImageNet normalization because MobileNetV3
# will use ImageNet-pretrained weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(image_path):
    """
    Load an image and convert it to RGB.

    Args:
        image_path: Path to image.

    Returns:
        PIL RGB image.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        image = Image.open(image_path)

        # Convert grayscale/RGBA/etc. to RGB.
        image = image.convert("RGB")

    except Exception as exc:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        ) from exc

    return image


def load_mask(mask_path):
    """
    Load a ground-truth mask.

    Mask convention:
        0 = authentic pixel
        1 = tampered pixel

    Returns:
        PIL grayscale mask.
    """

    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: {mask_path}"
        )

    try:
        mask = Image.open(mask_path).convert("L")

    except Exception as exc:
        raise RuntimeError(
            f"Could not read mask: {mask_path}"
        ) from exc

    return mask


def resize_image(image, image_size):
    """
    Resize RGB image.

    Args:
        image: PIL image
        image_size: int or (height, width)

    Returns:
        resized PIL image
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    return TF.resize(
        image,
        image_size,
        interpolation=TF.InterpolationMode.BILINEAR,
    )


def resize_mask(mask, image_size):
    """
    Resize segmentation mask using NEAREST interpolation.

    IMPORTANT:
    Never use bilinear interpolation for masks because
    it can create invalid intermediate label values.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    return TF.resize(
        mask,
        image_size,
        interpolation=TF.InterpolationMode.NEAREST,
    )


def image_to_tensor(image):
    """
    Convert PIL image to float tensor.

    Output:
        C x H x W
        values in [0, 1]
    """

    return TF.to_tensor(image)


def mask_to_tensor(mask):
    """
    Convert mask to binary float tensor.

    Output:
        1 x H x W

    Values:
        0.0 = authentic
        1.0 = tampered
    """

    mask_array = np.array(mask)

    # Any non-zero ground-truth pixel is treated as tampered.
    binary_mask = (mask_array > 0).astype(np.float32)

    tensor = torch.from_numpy(binary_mask)

    return tensor.unsqueeze(0)


def normalize_image(image_tensor):
    """
    ImageNet normalization.

    Args:
        image_tensor:
            C x H x W in [0, 1]

    Returns:
        normalized tensor
    """

    return TF.normalize(
        image_tensor,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def preprocess_image(
    image_path,
    image_size=224,
):
    """
    Complete preprocessing for a single image.

    Returns:
        normalized image tensor
    """

    image = load_image(image_path)

    image = resize_image(
        image,
        image_size,
    )

    image = image_to_tensor(image)

    image = normalize_image(image)

    return image


def preprocess_image_and_mask(
    image_path,
    mask_path,
    image_size=224,
):
    """
    Preprocess image and corresponding mask.

    This function should be used when augmentation
    is not required.

    Returns:
        image_tensor: 3 x H x W
        mask_tensor: 1 x H x W
    """

    image = load_image(image_path)
    mask = load_mask(mask_path)

    image = resize_image(
        image,
        image_size,
    )

    mask = resize_mask(
        mask,
        image_size,
    )

    image = image_to_tensor(image)
    mask = mask_to_tensor(mask)

    image = normalize_image(image)

    return image, mask