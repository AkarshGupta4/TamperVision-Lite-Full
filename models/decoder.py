import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x,
        skip,
    ):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.conv(x)


class LightweightLocalizationDecoder(nn.Module):
    """
    Lightweight segmentation decoder.

    Input:
        deep : B x 128 x 7 x 7
        high : B x 128 x 14 x 14
        mid  : B x 128 x 28 x 28
        low  : B x 128 x 56 x 56

    Output:
        B x 1 x H x W
    """

    def __init__(
        self,
        feature_channels=128,
    ):

        super().__init__()

        self.block1 = DecoderBlock(
            in_channels=feature_channels,
            skip_channels=feature_channels,
            out_channels=96,
        )

        self.block2 = DecoderBlock(
            in_channels=96,
            skip_channels=feature_channels,
            out_channels=64,
        )

        self.block3 = DecoderBlock(
            in_channels=64,
            skip_channels=feature_channels,
            out_channels=32,
        )

        self.final_conv = nn.Sequential(

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        features,
        output_size,
    ):

        x = features["deep"]

        x = self.block1(
            x,
            features["high"],
        )

        x = self.block2(
            x,
            features["mid"],
        )

        x = self.block3(
            x,
            features["low"],
        )

        # 56 x 56 -> original input resolution.
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        logits = self.final_conv(x)

        return logits


if __name__ == "__main__":

    model = LightweightLocalizationDecoder()

    features = {
        "low": torch.randn(
            2, 128, 56, 56
        ),
        "mid": torch.randn(
            2, 128, 28, 28
        ),
        "high": torch.randn(
            2, 128, 14, 14
        ),
        "deep": torch.randn(
            2, 128, 7, 7
        ),
    }

    output = model(
        features,
        output_size=(224, 224),
    )

    print(output.shape)