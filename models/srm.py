import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMFilter(nn.Module):
    """
    Fixed SRM-style high-pass filtering.

    Input:
        B x 3 x H x W

    Output:
        B x 3 x H x W
    """

    def __init__(self):
        super().__init__()

        kernel1 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ) / 4.0

        kernel2 = torch.tensor(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=torch.float32,
        ) / 12.0

        kernel3 = torch.tensor(
            [
                [0, 0, 1, 0, 0],
                [0, -1, -2, -1, 0],
                [1, -2, 4, -2, 1],
                [0, -1, -2, -1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=torch.float32,
        ) / 4.0

        kernels = torch.stack([kernel1, kernel2, kernel3])

        # Shape: 3 x 1 x 5 x 5
        self.register_buffer(
            "kernels",
            kernels.unsqueeze(1),
        )

    def forward(self, x):

        outputs = []

        for channel in range(3):
            channel_x = x[:, channel:channel + 1]

            filtered = F.conv2d(
                channel_x,
                self.kernels,
                padding=2,
            )

            # Average the three SRM responses.
            filtered = filtered.mean(dim=1, keepdim=True)

            outputs.append(filtered)

        return torch.cat(outputs, dim=1)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
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

    def forward(self, x):
        return self.block(x)


class SRMFeatureExtractor(nn.Module):
    """
    Lightweight forensic feature extractor.

    Returns:
        low   : B x 24  x 56 x 56
        mid   : B x 40  x 28 x 28
        high  : B x 80  x 14 x 14
        deep  : B x 160 x 7 x 7
    """

    def __init__(self):

        super().__init__()

        self.srm = SRMFilter()

        self.stem = ConvBlock(
            3,
            24,
            stride=2,
        )

        self.mid = ConvBlock(
            24,
            40,
            stride=2,
        )

        self.high = ConvBlock(
            40,
            80,
            stride=2,
        )

        self.deep = ConvBlock(
            80,
            160,
            stride=2,
        )

    def forward(self, x):

        x = self.srm(x)

        low = self.stem(x)
        mid = self.mid(low)
        high = self.high(mid)
        deep = self.deep(high)

        return {
            "low": low,
            "mid": mid,
            "high": high,
            "deep": deep,
        }


if __name__ == "__main__":

    model = SRMFeatureExtractor()

    x = torch.randn(2, 3, 224, 224)

    features = model(x)

    for name, feature in features.items():
        print(name, feature.shape)