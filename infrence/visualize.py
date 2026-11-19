from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_original_image(image_path):
    """
    Load original image as RGB.

    Args:
        image_path: Path to input image.

    Returns:
        PIL.Image
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        ) from exc

    return image


def probability_to_mask(
    probability_map,
    threshold=0.5,
):
    """
    Convert localization probability map into
    binary mask.

    Args:
        probability_map:
            H x W numpy array with values [0, 1]

        threshold:
            Pixel threshold.

    Returns:
        H x W uint8 mask.

    Values:
        0   = authentic
        255 = tampered
    """

    if probability_map.ndim == 3:
        probability_map = np.squeeze(
            probability_map
        )

    if probability_map.ndim != 2:
        raise ValueError(
            "Probability map must have shape H x W."
        )

    binary_mask = (
        probability_map >= threshold
    ).astype(np.uint8)

    binary_mask *= 255

    return binary_mask


def resize_mask_to_image(
    mask,
    image_size,
):
    """
    Resize mask using nearest-neighbor interpolation.

    This preserves binary mask values.
    """

    mask_image = Image.fromarray(
        mask,
        mode="L",
    )

    mask_image = mask_image.resize(
        image_size,
        Image.Resampling.NEAREST,
    )

    return np.array(mask_image)


def create_overlay(
    image,
    mask,
    alpha=0.45,
):
    """
    Create an overlay showing detected tampered regions.

    Args:
        image:
            PIL RGB image

        mask:
            H x W uint8 mask

        alpha:
            Overlay transparency.

    Returns:
        PIL RGB overlay image
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "image must be a PIL.Image."
        )

    image = image.convert("RGB")

    mask_image = Image.fromarray(
        mask,
        mode="L",
    )

    # Make sure mask has same dimensions as image.
    if mask_image.size != image.size:

        mask_image = mask_image.resize(
            image.size,
            Image.Resampling.NEAREST,
        )

    # Create a red overlay.
    overlay_color = Image.new(
        "RGB",
        image.size,
        (255, 0, 0),
    )

    # Only tampered pixels receive the overlay.
    result = Image.composite(
        overlay_color,
        image,
        mask_image,
    )

    # Blend with original image.
    result = Image.blend(
        image,
        result,
        alpha=alpha,
    )

    return result


def calculate_mask_statistics(mask):
    """
    Calculate basic statistics for predicted mask.

    Returns:
        dictionary
    """

    total_pixels = mask.size

    tampered_pixels = np.count_nonzero(
        mask > 0
    )

    tampered_ratio = (
        tampered_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    return {
        "total_pixels": int(total_pixels),
        "tampered_pixels": int(tampered_pixels),
        "tampered_ratio": float(tampered_ratio),
    }


def create_visualization(
    image,
    mask,
    overlay,
):
    """
    Create a single side-by-side visualization:

        Original | Mask | Overlay

    Returns:
        PIL RGB image
    """

    image = image.convert("RGB")
    overlay = overlay.convert("RGB")

    mask_image = Image.fromarray(
        mask,
        mode="L",
    ).convert("RGB")

    width, height = image.size

    canvas = Image.new(
        "RGB",
        (width * 3, height),
        "white",
    )

    canvas.paste(
        image,
        (0, 0),
    )

    canvas.paste(
        mask_image,
        (width, 0),
    )

    canvas.paste(
        overlay,
        (width * 2, 0),
    )

    return canvas


def add_labels(
    visualization,
    labels=(
        "Original",
        "Predicted Mask",
        "Overlay",
    ),
):
    """
    Add simple labels to visualization.
    """

    visualization = visualization.copy()

    draw = ImageDraw.Draw(
        visualization
    )

    width = visualization.width // 3

    for index, label in enumerate(labels):

        x = index * width + 10
        y = 10

        draw.rectangle(
            [
                x,
                y,
                x + 150,
                y + 28,
            ],
            fill=(0, 0, 0),
        )

        draw.text(
            (x + 5, y + 5),
            label,
            fill=(255, 255, 255),
        )

    return visualization


def save_prediction_visualization(
    image,
    mask,
    overlay,
    output_path,
):
    """
    Save Original + Mask + Overlay visualization.
    """

    visualization = create_visualization(
        image=image,
        mask=mask,
        overlay=overlay,
    )

    visualization = add_labels(
        visualization
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualization.save(
        output_path
    )

    return output_path