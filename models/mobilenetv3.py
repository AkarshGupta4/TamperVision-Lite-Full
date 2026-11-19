import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


class MobileNetV3FeatureExtractor(nn.Module):
    """
    MobileNetV3-Large feature extractor.

    Returns multi-scale feature maps:
        low   : B x 24  x 56 x 56
        mid   : B x 40  x 28 x 28
        high  : B x 80  x 14 x 14
        deep  : B x 960 x 7 x 7
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None

        backbone = mobilenet_v3_large(weights=weights)

        self.features = backbone.features

        # Selected MobileNetV3 feature stages.
        self.feature_indices = {
            "low": 2,
            "mid": 4,
            "high": 7,
            "deep": 16,
        }

        self.out_channels = {
            "low": 24,
            "mid": 40,
            "high": 80,
            "deep": 960,
        }

    def forward(self, x):

        outputs = {}

        for idx, layer in enumerate(self.features):
            x = layer(x)

            for name, target_idx in self.feature_indices.items():
                if idx == target_idx:
                    outputs[name] = x

        return outputs


if __name__ == "__main__":

    model = MobileNetV3FeatureExtractor(pretrained=False)

    x = torch.randn(2, 3, 224, 224)

    features = model(x)

    for name, feature in features.items():
        print(name, feature.shape)