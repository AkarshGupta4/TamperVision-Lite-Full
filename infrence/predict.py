import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.tampering_model import TamperingModel
from preprocessing.image_preprocessing import (
    load_image,
    resize_image,
    image_to_tensor,
    normalize_image,
)

from .visualize import (
    probability_to_mask,
    create_overlay,
    save_prediction_visualization,
    calculate_mask_statistics,
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


class TamperingPredictor:
    """
    Inference wrapper for TamperingModel.

    Responsibilities:
        1. Load trained checkpoint.
        2. Preprocess input image.
        3. Run model.
        4. Calculate detection probability.
        5. Calculate localization probability.
        6. Generate binary mask.
    """

    def __init__(
        self,
        checkpoint_path,
        image_size=224,
        threshold=0.5,
        device=None,
        pretrained=False,
        use_srm=True,
        use_cross_attention=True,
        use_localization=True,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.image_size = image_size
        self.threshold = threshold

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        self.model = TamperingModel(
            pretrained=pretrained,
            use_srm=use_srm,
            use_cross_attention=use_cross_attention,
            use_localization=use_localization,
        )

        # --------------------------------------------------
        # Load checkpoint
        # --------------------------------------------------

        self._load_checkpoint()

        self.model.to(
            self.device
        )

        self.model.eval()

    def _load_checkpoint(self):
        """
        Load model checkpoint.

        Supports either:

            checkpoint = state_dict

        or:

            checkpoint = {
                "model_state_dict": ...
            }
        """

        if not self.checkpoint_path.exists():

            raise FileNotFoundError(
                "Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        if isinstance(
            checkpoint,
            dict,
        ) and "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

        # Handle checkpoints saved using DataParallel.
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):

                key = key[
                    len("module.") :
                ]

            cleaned_state_dict[
                key
            ] = value

        missing, unexpected = (
            self.model.load_state_dict(
                cleaned_state_dict,
                strict=False,
            )
        )

        if missing:

            print(
                "Warning: Missing model keys:"
            )

            for key in missing:
                print(
                    f"  - {key}"
                )

        if unexpected:

            print(
                "Warning: Unexpected model keys:"
            )

            for key in unexpected:
                print(
                    f"  - {key}"
                )

    def preprocess(self, image):
        """
        Preprocess a PIL image.

        Returns:
            1 x 3 x H x W tensor
        """

        image = resize_image(
            image,
            self.image_size,
        )

        tensor = image_to_tensor(
            image
        )

        tensor = normalize_image(
            tensor
        )

        tensor = tensor.unsqueeze(
            0
        )

        return tensor.to(
            self.device
        )

    @torch.no_grad()
    def predict(self, image):
        """
        Run inference on a PIL image.

        Returns:
            dictionary containing prediction results.
        """

        if not hasattr(
            image,
            "convert",
        ):

            raise TypeError(
                "image must be a PIL.Image."
            )

        image = image.convert(
            "RGB"
        )

        input_tensor = self.preprocess(
            image
        )

        outputs = self.model(
            input_tensor
        )

        # --------------------------------------------------
        # Detection
        # --------------------------------------------------

        detection_logits = outputs[
            "detection_logits"
        ]

        detection_probability = (
            torch.sigmoid(
                detection_logits
            )
            .item()
        )

        is_tampered = (
            detection_probability >= 0.5
        )

        # --------------------------------------------------
        # Localization
        # --------------------------------------------------

        localization_logits = outputs[
            "localization_logits"
        ]

        if localization_logits is not None:

            localization_probability = (
                torch.sigmoid(
                    localization_logits
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            mask = probability_to_mask(
                localization_probability,
                threshold=self.threshold,
            )

        else:

            localization_probability = None

            mask = np.zeros(
                (
                    image.height,
                    image.width,
                ),
                dtype=np.uint8,
            )

        # --------------------------------------------------
        # Resize mask to original image size
        # --------------------------------------------------

        if mask.shape != (
            image.height,
            image.width,
        ):

            mask_image = torch.from_numpy(
                mask
            ).float()

            mask_image = torch.nn.functional.interpolate(
                mask_image[
                    None,
                    None,
                    ...,
                ],
                size=(
                    image.height,
                    image.width,
                ),
                mode="nearest",
            )

            mask = (
                mask_image
                .squeeze()
                .numpy()
                .astype(np.uint8)
            )

        statistics = (
            calculate_mask_statistics(
                mask
            )
        )

        return {
            "detection_probability":
                float(
                    detection_probability
                ),

            "prediction":
                "Tampered"
                if is_tampered
                else "Authentic",

            "is_tampered":
                bool(is_tampered),

            "localization_probability":
                localization_probability,

            "mask":
                mask,

            "mask_statistics":
                statistics,
        }

    def predict_file(
        self,
        image_path,
        output_dir="results",
        save_outputs=True,
    ):
        """
        Run inference on an image file.
        """

        image_path = Path(
            image_path
        )

        if image_path.suffix.lower() not in (
            SUPPORTED_EXTENSIONS
        ):

            raise ValueError(
                "Unsupported image format: "
                f"{image_path.suffix}"
            )

        image = load_image(
            image_path
        )

        result = self.predict(
            image
        )

        overlay = create_overlay(
            image=image,
            mask=result["mask"],
        )

        result["image"] = image
        result["overlay"] = overlay

        # --------------------------------------------------
        # Save outputs
        # --------------------------------------------------

        if save_outputs:

            output_dir = Path(
                output_dir
            )

            masks_dir = (
                output_dir / "masks"
            )

            visualizations_dir = (
                output_dir
                / "visualizations"
            )

            metrics_dir = (
                output_dir / "metrics"
            )

            masks_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            visualizations_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            metrics_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Mask
            mask_path = (
                masks_dir
                / f"{image_path.stem}_mask.png"
            )

            mask_image = (
                torch.from_numpy(
                    result["mask"]
                )
                .numpy()
            )

            from PIL import Image

            Image.fromarray(
                mask_image,
                mode="L",
            ).save(
                mask_path
            )

            # Visualization
            visualization_path = (
                visualizations_dir
                / f"{image_path.stem}_result.png"
            )

            save_prediction_visualization(
                image=image,
                mask=result["mask"],
                overlay=overlay,
                output_path=visualization_path,
            )

            # JSON metrics
            metrics_path = (
                metrics_dir
                / f"{image_path.stem}_prediction.json"
            )

            metrics = {
                "image":
                    str(image_path),

                "prediction":
                    result["prediction"],

                "detection_probability":
                    result[
                        "detection_probability"
                    ],

                "mask_statistics":
                    result[
                        "mask_statistics"
                    ],

                "threshold":
                    self.threshold,

                "device":
                    str(self.device),
            }

            with open(
                metrics_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    metrics,
                    file,
                    indent=4,
                )

            result[
                "mask_path"
            ] = mask_path

            result[
                "visualization_path"
            ] = visualization_path

            result[
                "metrics_path"
            ] = metrics_path

        return result


def print_prediction(result):
    """
    Print human-readable prediction.
    """

    probability = (
        result[
            "detection_probability"
        ]
    )

    prediction = result[
        "prediction"
    ]

    statistics = result[
        "mask_statistics"
    ]

    print()
    print("=" * 50)
    print("IMAGE TAMPERING DETECTION")
    print("=" * 50)

    print(
        f"Prediction          : {prediction}"
    )

    print(
        "Tampering Probability: "
        f"{probability:.4f}"
    )

    print(
        "Tampering Percentage : "
        f"{statistics['tampered_ratio'] * 100:.2f}%"
    )

    print(
        "Tampered Pixels      : "
        f"{statistics['tampered_pixels']}"
    )

    print(
        "Total Pixels         : "
        f"{statistics['total_pixels']}"
    )

    print("=" * 50)

    if "mask_path" in result:

        print(
            f"Mask saved          : "
            f"{result['mask_path']}"
        )

        print(
            f"Visualization saved : "
            f"{result['visualization_path']}"
        )

        print(
            f"Metrics saved       : "
            f"{result['metrics_path']}"
        )

    print()


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Image Tampering Detection "
            "and Localization"
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help=(
            "Path to trained model "
            "checkpoint."
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Model input image size.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Pixel-level localization "
            "threshold."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device, e.g. cuda or cpu. "
            "Default: automatic."
        ),
    )

    return parser


def main():

    parser = build_parser()

    args = parser.parse_args()

    predictor = TamperingPredictor(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        threshold=args.threshold,
        device=args.device,
        pretrained=False,
        use_srm=True,
        use_cross_attention=True,
        use_localization=True,
    )

    result = predictor.predict_file(
        image_path=args.image,
        output_dir=args.output_dir,
        save_outputs=True,
    )

    print_prediction(
        result
    )


if __name__ == "__main__":
    main()