import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMFilter(nn.Module):
    """
    Fixed SRM-style high-pass filter bank.

    Input:
        B x 3 x H x W

    Output:
        B x 9 x H x W

    Explanation:
        3 RGB channels × 3 filters = 9 response maps.
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

        kernels = torch.stack(
            [
                kernel1,
                kernel2,
                kernel3,
            ]
        )

        # Shape:
        # 3 filters x 1 channel x 5 x 5
        kernels = kernels.unsqueeze(1)

        self.register_buffer(
            "kernels",
            kernels,
        )

    def forward(self, x):
        """
        Args:
            x:
                B x 3 x H x W

        Returns:
            B x 9 x H x W
        """

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape B x C x H x W"
            )

        if x.shape[1] != 3:
            raise ValueError(
                "SRMFilter expects 3-channel RGB input."
            )

        responses = []

        for channel in range(3):

            channel_input = x[
                :,
                channel:channel + 1,
                :,
                :,
            ]

            filtered = F.conv2d(
                channel_input,
                self.kernels,
                padding=2,
            )

            responses.append(filtered)

        return torch.cat(
            responses,
            dim=1,
        )


class SRMPreprocessor(nn.Module):
    """
    SRM preprocessing wrapper.

    Optionally normalizes the magnitude of high-frequency
    responses to make them easier for subsequent layers
    to process.
    """

    def __init__(
        self,
        normalize=True,
    ):

        super().__init__()

        self.filter = SRMFilter()
        self.normalize = normalize

    def forward(self, x):

        features = self.filter(x)

        if self.normalize:

            mean = features.mean(
                dim=(2, 3),
                keepdim=True,
            )

            std = features.std(
                dim=(2, 3),
                keepdim=True,
            )

            features = (
                features - mean
            ) / (
                std + 1e-6
            )

        return features


if __name__ == "__main__":

    print("Testing SRM filter...")

    model = SRMFilter()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(x)

    print(
        "Input shape:",
        x.shape,
    )

    print(
        "Output shape:",
        output.shape,
    )