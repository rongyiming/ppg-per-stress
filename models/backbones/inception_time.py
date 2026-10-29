"""
InceptionTime 1D backbone — multi-scale parallel conv for time series.

Reference: Fawaz et al. 2020 "InceptionTime: Finding AlexNet for TSC"
Each InceptionBlock runs 3 parallel 1D conv branches (k=1, 3, 5) plus a
maxpool branch, concatenates, then projects back.

The 4-level output interface matches FinetuneModel's multi-level ResNet usage:
  f_levels = [l0_out, l1_out, l2_out, l3_out]  with channels [64, 128, 256, 512]
"""

import torch
from torch import nn


class InceptionBlock(nn.Module):
    """1D Inception module with bottleneck + 4 parallel branches."""

    def __init__(self, in_channels: int, out_channels: int,
                 bottleneck_channels: int = 32):
        super().__init__()
        self.out_channels = out_channels

        # Bottleneck: reduce input dimension
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        # 4 parallel branches
        self.branch1 = nn.Sequential(
            nn.Conv1d(bottleneck_channels, out_channels // 4, kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(bottleneck_channels, out_channels // 4, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(bottleneck_channels, out_channels // 4, kernel_size=5,
                      padding=2, bias=False),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(bottleneck_channels, out_channels // 4, kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

        # Residual projection if channel sizes differ
        self.match = None
        if in_channels != out_channels:
            self.match = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.bottleneck(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        if self.match is not None:
            identity = self.match(identity)
        x = x + identity
        return self.act(x)


class InceptionTime(nn.Module):
    """InceptionTime backbone with 4-level feature pyramid output.

    Level layout (125Hz, 1250-sample input):
      - stem:   Conv1d(1, 64, k=7, s=2, p=3) → BN → ReLU → MaxPool(k=3, s=2)
      - level0: 3 × InceptionBlock(64→64),      313 time steps
      - level1: Down(k=3,s=2) + 3 × InceptionBlock(128→128), 157
      - level2: Down(k=3,s=2) + 3 × InceptionBlock(256→256),  79
      - level3: Down(k=3,s=2) + 3 × InceptionBlock(512→512),  40

    Returns f_levels = [l0, l1, l2, l3] compatible with FinetuneModel multi-level.
    """

    def __init__(self, input_channels: int = 1,
                 base_channels: int = 64,
                 blocks_per_level: int = 3):
        super().__init__()

        # Stem: same as ResNet1D conv1 for fair comparison
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Level 0 – 64 channels
        self.level0_blocks = nn.ModuleList([
            InceptionBlock(base_channels, base_channels)
            for _ in range(blocks_per_level)
        ])

        # Level 1 – 128 channels (with stride-2 downsampling)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.level1_blocks = nn.ModuleList([
            InceptionBlock(base_channels * 2, base_channels * 2)
            for _ in range(blocks_per_level)
        ])

        # Level 2 – 256 channels
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.level2_blocks = nn.ModuleList([
            InceptionBlock(base_channels * 4, base_channels * 4)
            for _ in range(blocks_per_level)
        ])

        # Level 3 – 512 channels
        self.down3 = nn.Conv1d(base_channels * 4, base_channels * 8,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.level3_blocks = nn.ModuleList([
            InceptionBlock(base_channels * 8, base_channels * 8)
            for _ in range(blocks_per_level)
        ])

        self.output_channels = base_channels * 8  # 512

    def forward(self, x):
        """Return [l0, l1, l2, l3] feature pyramid."""
        x = self.stem(x)

        # Level 0
        for blk in self.level0_blocks:
            x = blk(x)
        l0 = x

        # Level 1
        x = self.down1(x)
        for blk in self.level1_blocks:
            x = blk(x)
        l1 = x

        # Level 2
        x = self.down2(x)
        for blk in self.level2_blocks:
            x = blk(x)
        l2 = x

        # Level 3
        x = self.down3(x)
        for blk in self.level3_blocks:
            x = blk(x)
        l3 = x

        return [l0, l1, l2, l3]
