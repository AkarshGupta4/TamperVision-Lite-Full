import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.tampering_model import TamperingModel
from preprocessing.image_preprocessing import (
    load_image,
    load_mask,
    resize_image,
    resize_mask,
    image_to_tensor,
    mask_to_tensor,
    normalize_image,
)

from .detection_metrics import (
    calculate_detection_metrics,
    print_detection_metrics,
)

from .localization_metrics import (
    calculate_localization_metrics,
    print_localization_metrics,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


class ModelEvaluator:
    """
    Complete evaluation pipeline.

    Evaluates:

        Image-level detection
        Pixel-level localization
    """

    def __init__(
        self,
        checkpoint_path,
        image_size=224,
        detection_threshold=0.5,
        localization_threshold=0.5,
        device=None,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.image_size = image_size

        self.detection_threshold = (
            detection_threshold
        )

        self.localization_threshold = (
            localization_threshold
        )

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
            pretrained=False,
            use_srm=True,
            use_cross_attention=True,
            use_localization=True,
        )

        self._load_checkpoint()

        self.model.to(
            self.device
        )

        self.model.eval()

    def _load_checkpoint(self):
        """
        Load trained checkpoint.
        """

        if not self.checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

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
                "Warning: Missing checkpoint keys:"
            )

            for key in missing:

                print(
                    f"  - {key}"
                )

        if unexpected:

            print(
                "Warning: Unexpected checkpoint keys:"
            )

            for key in unexpected:

                print(
                    f"  - {key}"
                )

    def preprocess(
        self,
        image_path,
    ):
        """
        Load and preprocess image.
        """

        image = load_image(
            image_path
        )

        original_size = image.size

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

        tensor = tensor.to(
            self.device
        )

        return (
            tensor,
            original_size,
        )

    def preprocess_mask(
        self,
        mask_path,
    ):
        """
        Load ground-truth mask.

        Mask is resized using nearest-neighbor
        interpolation.
        """

        mask = load_mask(
            mask_path
        )

        mask = resize_mask(
            mask,
            self.image_size,
        )

        mask = mask_to_tensor(
            mask
        )

        return mask

    @torch.no_grad()
    def evaluate(
        self,
        image_dir,
        mask_dir,
        save_predictions=False,
        output_dir="results/evaluation",
    ):
        """
        Evaluate complete test dataset.

        Args:
            image_dir:
                Directory containing test images.

            mask_dir:
                Directory containing ground-truth masks.

            save_predictions:
                Whether to save predicted masks.

            output_dir:
                Where predictions/metrics are saved.
        """

        image_dir = Path(
            image_dir
        )

        mask_dir = Path(
            mask_dir
        )

        if not image_dir.exists():

            raise FileNotFoundError(
                f"Image directory not found: "
                f"{image_dir}"
            )

        if not mask_dir.exists():

            raise FileNotFoundError(
                f"Mask directory not found: "
                f"{mask_dir}"
            )

        # --------------------------------------------------
        # Find images
        # --------------------------------------------------

        image_paths = sorted(
            [
                path
                for path in image_dir.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            ]
        )

        if len(image_paths) == 0:

            raise RuntimeError(
                f"No images found in "
                f"{image_dir}"
            )

        if save_predictions:

            output_dir = Path(
                output_dir
            )

            prediction_dir = (
                output_dir
                / "predicted_masks"
            )

            prediction_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        # --------------------------------------------------
        # Storage
        # --------------------------------------------------

        detection_targets = []
        detection_probabilities = []

        localization_targets = []
        localization_predictions = []

        skipped_images = []

        # --------------------------------------------------
        # Evaluation loop
        # --------------------------------------------------

        for image_path in tqdm(
            image_paths,
            desc="Evaluating",
        ):

            # Find corresponding mask.
            mask_path = (
                mask_dir
                / f"{image_path.stem}.png"
            )

            # Also check same extension.
            if not mask_path.exists():

                alternative = (
                    mask_dir
                    / image_path.name
                )

                if alternative.exists():

                    mask_path = alternative

                else:

                    skipped_images.append(
                        {
                            "image": str(
                                image_path
                            ),
                            "reason": (
                                "Ground-truth "
                                "mask not found"
                            ),
                        }
                    )

                    continue

            try:

                image_tensor, _ = (
                    self.preprocess(
                        image_path
                    )
                )

                target_mask = (
                    self.preprocess_mask(
                        mask_path
                    )
                )

            except Exception as exc:

                skipped_images.append(
                    {
                        "image": str(
                            image_path
                        ),
                        "reason": str(
                            exc
                        ),
                    }
                )

                continue

            # --------------------------------------------------
            # Ground-truth detection label
            # --------------------------------------------------

            target_mask_np = (
                target_mask
                .squeeze(0)
                .numpy()
                > 0.5
            )

            has_tampering = (
                np.any(
                    target_mask_np
                )
            )

            detection_target = int(
                has_tampering
            )

            # --------------------------------------------------
            # Model inference
            # --------------------------------------------------

            outputs = self.model(
                image_tensor
            )

            detection_probability = (
                torch.sigmoid(
                    outputs[
                        "detection_logits"
                    ]
                )
                .item()
            )

            localization_probability = (
                torch.sigmoid(
                    outputs[
                        "localization_logits"
                    ]
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # --------------------------------------------------
            # Binary localization mask
            # --------------------------------------------------

            predicted_mask = (
                localization_probability
                >= self.localization_threshold
            )

            # --------------------------------------------------
            # Store detection results
            # --------------------------------------------------

            detection_targets.append(
                detection_target
            )

            detection_probabilities.append(
                detection_probability
            )

            # --------------------------------------------------
            # Store localization results
            # --------------------------------------------------

            localization_targets.append(
                target_mask_np
            )

            localization_predictions.append(
                predicted_mask
            )

            # --------------------------------------------------
            # Save predicted mask
            # --------------------------------------------------

            if save_predictions:

                prediction_image = Image.fromarray(
                    (
                        predicted_mask
                        .astype(np.uint8)
                        * 255
                    ),
                    mode="L",
                )

                prediction_path = (
                    prediction_dir
                    / f"{image_path.stem}_mask.png"
                )

                prediction_image.save(
                    prediction_path
                )

        # --------------------------------------------------
        # Check results
        # --------------------------------------------------

        if len(detection_targets) == 0:

            raise RuntimeError(
                "No valid images were evaluated."
            )

        # --------------------------------------------------
        # Detection metrics
        # --------------------------------------------------

        detection_metrics = (
            calculate_detection_metrics(
                targets=np.array(
                    detection_targets
                ),
                probabilities=np.array(
                    detection_probabilities
                ),
                threshold=self.detection_threshold,
            )
        )

        # --------------------------------------------------
        # Localization metrics
        # --------------------------------------------------

        localization_metrics = (
            calculate_localization_metrics(
                targets=np.stack(
                    localization_targets
                ),
                predictions=np.stack(
                    localization_predictions
                ),
            )
        )

        # --------------------------------------------------
        # Complete result
        # --------------------------------------------------

        results = {
            "detection": detection_metrics,
            "localization": localization_metrics,
            "dataset": {
                "image_directory": str(
                    image_dir
                ),
                "mask_directory": str(
                    mask_dir
                ),
                "num_images_found": len(
                    image_paths
                ),
                "num_images_evaluated": len(
                    detection_targets
                ),
                "num_images_skipped": len(
                    skipped_images
                ),
            },
            "configuration": {
                "image_size": self.image_size,
                "detection_threshold": (
                    self.detection_threshold
                ),
                "localization_threshold": (
                    self.localization_threshold
                ),
                "device": str(
                    self.device
                ),
            },
            "skipped_images": skipped_images,
        }

        # --------------------------------------------------
        # Print
        # --------------------------------------------------

        print_detection_metrics(
            detection_metrics
        )

        print_localization_metrics(
            localization_metrics
        )

        # --------------------------------------------------
        # Save JSON
        # --------------------------------------------------

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_path = (
            output_dir
            / "metrics.json"
        )

        with open(
            metrics_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
            )

        print()
        print(
            f"Evaluation results saved to:"
        )
        print(
            metrics_path
        )

        return results


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate image tampering "
            "detection and localization."
        )
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help=(
            "Directory containing test images."
        ),
    )

    parser.add_argument(
        "--masks",
        type=str,
        required=True,
        help=(
            "Directory containing ground-truth masks."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/"
            "best_model.pth"
        ),
        help="Path to trained checkpoint.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Model input image size.",
    )

    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold for image-level "
            "tampering prediction."
        ),
    )

    parser.add_argument(
        "--localization-threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold for pixel-level "
            "tampering prediction."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/evaluation"
        ),
        help="Evaluation output directory.",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predicted masks.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu.",
    )

    return parser


def main():

    parser = build_parser()

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        detection_threshold=(
            args.detection_threshold
        ),
        localization_threshold=(
            args.localization_threshold
        ),
        device=args.device,
    )

    evaluator.evaluate(
        image_dir=args.images,
        mask_dir=args.masks,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()