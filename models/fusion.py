import torch
import torch.nn as nn

from .cross_attention import BidirectionalCrossAttention


class FeatureProjection(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):

        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.projection(x)


class MultiScaleFusion(nn.Module):
    """
    RGB + SRM multi-scale fusion.

    Feature dimensions are projected to common dimensions.

    low  : 56 x 56
    mid  : 28 x 28
    high : 14 x 14
    deep : 7 x 7
    """

    def __init__(
        self,
        use_cross_attention=True,
        fusion_channels=128,
    ):

        super().__init__()

        self.use_cross_attention = use_cross_attention

        # MobileNet channels.
        self.rgb_low = FeatureProjection(
            24,
            fusion_channels,
        )

        self.rgb_mid = FeatureProjection(
            40,
            fusion_channels,
        )

        self.rgb_high = FeatureProjection(
            80,
            fusion_channels,
        )

        self.rgb_deep = FeatureProjection(
            960,
            fusion_channels,
        )

        # SRM channels.
        self.srm_low = FeatureProjection(
            24,
            fusion_channels,
        )

        self.srm_mid = FeatureProjection(
            40,
            fusion_channels,
        )

        self.srm_high = FeatureProjection(
            80,
            fusion_channels,
        )

        self.srm_deep = FeatureProjection(
            160,
            fusion_channels,
        )

        if use_cross_attention:

            self.attention = BidirectionalCrossAttention(
                channels=fusion_channels,
                num_heads=4,
            )

        self.simple_fusion = nn.ModuleDict()

        for name in ["low", "mid", "high", "deep"]:

            self.simple_fusion[name] = nn.Sequential(
                nn.Conv2d(
                    fusion_channels * 2,
                    fusion_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True),
            )

    def forward(
        self,
        rgb_features,
        srm_features,
    ):

        rgb = {
            "low": self.rgb_low(
                rgb_features["low"]
            ),

            "mid": self.rgb_mid(
                rgb_features["mid"]
            ),

            "high": self.rgb_high(
                rgb_features["high"]
            ),

            "deep": self.rgb_deep(
                rgb_features["deep"]
            ),
        }

        srm = {
            "low": self.srm_low(
                srm_features["low"]
            ),

            "mid": self.srm_mid(
                srm_features["mid"]
            ),

            "high": self.srm_high(
                srm_features["high"]
            ),

            "deep": self.srm_deep(
                srm_features["deep"]
            ),
        }

        fused = {}

        for name in ["low", "mid", "high", "deep"]:

            rgb_feature = rgb[name]
            srm_feature = srm[name]

            if (
                name == "deep"
                and self.use_cross_attention
            ):
                fused[name] = self.attention(
                    rgb_feature,
                    srm_feature,
                )

            else:

                combined = torch.cat(
                    [
                        rgb_feature,
                        srm_feature,
                    ],
                    dim=1,
                )

                fused[name] = self.simple_fusion[
                    name
                ](combined)

        return fused