# scripts/image_encoding/autoencoder.py

import math
import torch
import torch.nn as nn


def _make_gn(channels: int) -> nn.GroupNorm:
    groups = 8
    if channels < groups:
        groups = 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.gn1 = _make_gn(out_ch)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.gn2 = _make_gn(out_ch)

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0)

        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)

        identity = self.res_conv(identity)

        x = x + identity
        x = self.act2(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, final: bool = False):
        super().__init__()
        self.final = bool(final)

        if self.final:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
            self.gn1 = _make_gn(out_ch)
            self.act1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
            self.gn2 = _make_gn(out_ch)

            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

            self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.final:
            x = self.up(x)
            x = self.conv(x)
            return x

        identity = x

        x = self.up(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)

        identity = self.up(identity)
        identity = self.res_conv(identity)

        x = x + identity
        x = self.act2(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 128,
        image_size: int = 128,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.latent_dim = int(latent_dim)
        self.image_size = int(image_size)

        assert self.image_size > 0
        assert (self.image_size & (self.image_size - 1)) == 0

        num_down = int(math.log2(self.image_size)) - 4
        if num_down < 2:
            num_down = 2

        chs = [self.base_channels * (2 ** i) for i in range(num_down)]

        enc_blocks = []
        in_ch = self.in_channels
        for out_ch in chs:
            enc_blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.enc_blocks = nn.ModuleList(enc_blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.image_size, self.image_size)
            x = dummy
            for blk in self.enc_blocks:
                x = blk(x)
            self.feature_shape = x.shape[1:]
            self.feature_dim = int(x.view(1, -1).size(1))

        self.to_latent = nn.Linear(self.feature_dim, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, self.feature_dim)

        chs_dec = list(reversed(chs))
        dec_blocks = []
        in_ch = chs_dec[0]
        for out_ch in chs_dec[1:]:
            dec_blocks.append(DeconvBlock(in_ch, out_ch, final=False))
            in_ch = out_ch

        dec_blocks.append(
            DeconvBlock(
                in_ch,
                self.in_channels,
                final=True,
            )
        )
        self.dec_blocks = nn.ModuleList(dec_blocks)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.enc_blocks:
            x = blk(x)
        x = x.view(x.size(0), -1)
        z = self.to_latent(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        x = x.view(x.size(0), *self.feature_shape)
        for blk in self.dec_blocks:
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        logits = self.decode(z)
        return logits
