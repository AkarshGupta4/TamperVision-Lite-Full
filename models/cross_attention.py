import torch
import torch.nn as nn


class CrossAttention2D(nn.Module):
    """
    Lightweight 2D Cross-Attention.

    Query:
        B x C x H x W

    Context:
        B x C x H x W

    Output:
        B x C x H x W
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        dropout=0.1,
    ):

        super().__init__()

        assert channels % num_heads == 0, (
            "channels must be divisible by num_heads"
        )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.k_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.v_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.out_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):

        B, C, H, W = query.shape

        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # B x C x H x W
        # →
        # B x heads x HW x head_dim

        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        q = q.reshape(
            B,
            H * W,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.reshape(
            B,
            H * W,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.reshape(
            B,
            H * W,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        attention = attention / (self.head_dim ** 0.5)

        attention = torch.softmax(
            attention,
            dim=-1,
        )

        attention = self.dropout(attention)

        output = torch.matmul(
            attention,
            v,
        )

        output = output.transpose(
            1,
            2,
        ).contiguous()

        output = output.reshape(
            B,
            H * W,
            C,
        )

        output = output.transpose(
            1,
            2,
        )

        output = output.reshape(
            B,
            C,
            H,
            W,
        )

        output = self.out_proj(output)

        # Residual connection.
        output = output + query

        output = self.norm(output)

        return output


class BidirectionalCrossAttention(nn.Module):
    """
    RGB <-> SRM cross attention.

    Both directions are calculated and then fused.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
    ):

        super().__init__()

        self.rgb_to_srm = CrossAttention2D(
            channels=channels,
            num_heads=num_heads,
        )

        self.srm_to_rgb = CrossAttention2D(
            channels=channels,
            num_heads=num_heads,
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        rgb_features,
        srm_features,
    ):

        rgb_attended = self.rgb_to_srm(
            rgb_features,
            srm_features,
        )

        srm_attended = self.srm_to_rgb(
            srm_features,
            rgb_features,
        )

        fused = torch.cat(
            [
                rgb_attended,
                srm_attended,
            ],
            dim=1,
        )

        return self.fusion(fused)


if __name__ == "__main__":

    model = BidirectionalCrossAttention(
        channels=128,
        num_heads=4,
    )

    rgb = torch.randn(
        2, 128, 14, 14
    )

    srm = torch.randn(
        2, 128, 14, 14
    )

    output = model(
        rgb,
        srm,
    )

    print(output.shape)