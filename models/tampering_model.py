import torch
import torch.nn as nn

from .mobilenetv3 import MobileNetV3FeatureExtractor
from .srm import SRMFeatureExtractor
from .fusion import MultiScaleFusion
from .decoder import LightweightLocalizationDecoder


class DetectionHead(nn.Module):
    """
    Image-level tampering detection head.

    Input:
        B x C x H x W

    Output:
        B x 1
    """

    def __init__(
        self,
        in_channels=128,
        hidden_dim=128,
    ):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Linear(
                in_channels,
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, x):

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        return self.classifier(x)


class TamperingModel(nn.Module):
    """
    Complete image tampering detection
    and localization model.

    Architecture:

        RGB
         |
         +--> MobileNetV3
         |
         |
        SRM
         |
         +--> Forensic Branch
         |
         v
      Multi-scale Fusion
         |
         +----------------+
         |                |
         v                v
    Detection         Localization
       Head              Decoder
         |                |
         v                v
      B x 1          B x 1 x H x W
    """

    def __init__(
        self,
        pretrained=True,
        use_srm=True,
        use_cross_attention=True,
        use_localization=True,
        fusion_channels=128,
    ):

        super().__init__()

        self.use_srm = use_srm
        self.use_cross_attention = use_cross_attention
        self.use_localization = use_localization

        # --------------------------------------------------
        # RGB Backbone
        # --------------------------------------------------

        self.rgb_backbone = MobileNetV3FeatureExtractor(
            pretrained=pretrained,
        )

        # --------------------------------------------------
        # SRM / Forensic Branch
        # --------------------------------------------------

        if use_srm:

            self.srm_branch = SRMFeatureExtractor()

        # --------------------------------------------------
        # Feature Fusion
        # --------------------------------------------------

        self.fusion = MultiScaleFusion(
            use_cross_attention=use_cross_attention,
            fusion_channels=fusion_channels,
        )

        # --------------------------------------------------
        # Detection Head
        # --------------------------------------------------

        self.detection_head = DetectionHead(
            in_channels=fusion_channels,
        )

        # --------------------------------------------------
        # Localization Decoder
        # --------------------------------------------------

        if use_localization:

            self.decoder = LightweightLocalizationDecoder(
                feature_channels=fusion_channels,
            )

    def forward(self, x):

        input_size = x.shape[-2:]

        # --------------------------------------------------
        # RGB Features
        # --------------------------------------------------

        rgb_features = self.rgb_backbone(x)

        # --------------------------------------------------
        # SRM Features
        # --------------------------------------------------

        if self.use_srm:

            srm_features = self.srm_branch(x)

        else:

            # If SRM is disabled, create zero features.
            srm_features = {}

            for name, feature in rgb_features.items():

                srm_features[name] = torch.zeros(
                    feature.shape[0],
                    self._get_srm_channels(name),
                    feature.shape[2],
                    feature.shape[3],
                    device=feature.device,
                    dtype=feature.dtype,
                )

        # --------------------------------------------------
        # Feature Fusion
        # --------------------------------------------------

        fused_features = self.fusion(
            rgb_features,
            srm_features,
        )

        # --------------------------------------------------
        # Detection
        # --------------------------------------------------

        detection_logits = self.detection_head(
            fused_features["deep"]
        )

        # --------------------------------------------------
        # Localization
        # --------------------------------------------------

        if self.use_localization:

            localization_logits = self.decoder(
                fused_features,
                output_size=input_size,
            )

        else:

            localization_logits = None

        return {
            "detection_logits": detection_logits,
            "localization_logits": localization_logits,
            "features": fused_features,
        }

    @staticmethod
    def _get_srm_channels(name):

        channels = {
            "low": 24,
            "mid": 40,
            "high": 80,
            "deep": 160,
        }

        return channels[name]


if __name__ == "__main__":

    print("Testing TamperingModel...")

    model = TamperingModel(
        pretrained=False,
        use_srm=True,
        use_cross_attention=True,
        use_localization=True,
    )

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    outputs = model(x)

    print(
        "Detection:",
        outputs["detection_logits"].shape,
    )

    print(
        "Localization:",
        outputs["localization_logits"].shape,
    )

    print("\nFused features:")

    for name, feature in outputs["features"].items():

        print(
            name,
            feature.shape,
        )