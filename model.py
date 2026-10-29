"""
model.py -- Multi-Scale Dilated Residual CNN (MS-DRCNN) for Colorization.
Preserves full spatial resolution end-to-end without downsampling or skip connections.

Interface:
    Input  : (B, 1, H, W)  -- L channel, normalized [0, 1]
    Output : (B, 2, H, W)  -- predicted ab channels, in [-1, 1]
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with dilated convolutions.
    Conv(k=3, p=dilation, dilation=dilation) -> BN -> ReLU
    -> Conv(k=3, p=dilation, dilation=dilation) -> BN
    -> add input -> ReLU
    """

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class ColorizationUNet(nn.Module):
    """
    Multi-Scale Dilated Residual CNN (MS-DRCNN) for LAB colorization.
    Kept under the class name `ColorizationUNet` to satisfy interface contracts.

    Architecture:
      1. Stem: Conv2d(1 -> base_channels, k=3, p=1) + BN + ReLU
      2. Four parallel dilated residual branches:
         - Dilation rates: 1, 2, 4, 8
         - Each has 2x ResidualBlock(channels=base_channels, dilation=d)
      3. Fusion: concat 4 branches -> Conv2d(4*base_channels -> base_channels, k=1) + BN + ReLU
      4. Refinement: 2x ResidualBlock(channels=base_channels, dilation=1)
      5. Head: Conv2d(base_channels -> 2, k=1) -> Tanh()
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        b = base_channels

        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, b, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

        # 2. Four parallel dilated residual branches (dilations 1, 2, 4, 8)
        self.branch1 = nn.Sequential(
            ResidualBlock(b, dilation=1),
            ResidualBlock(b, dilation=1),
        )
        self.branch2 = nn.Sequential(
            ResidualBlock(b, dilation=2),
            ResidualBlock(b, dilation=2),
        )
        self.branch3 = nn.Sequential(
            ResidualBlock(b, dilation=4),
            ResidualBlock(b, dilation=4),
        )
        self.branch4 = nn.Sequential(
            ResidualBlock(b, dilation=8),
            ResidualBlock(b, dilation=8),
        )

        # 3. Fusion: concat 4 branches -> Conv 1x1 -> BN -> ReLU
        self.fusion = nn.Sequential(
            nn.Conv2d(4 * b, b, kernel_size=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

        # 4. Refinement: 2x ResidualBlock at dilation=1
        self.refinement = nn.Sequential(
            ResidualBlock(b, dilation=1),
            ResidualBlock(b, dilation=1),
        )

        # 5. Head: Conv 1x1 -> 2 channels -> Tanh
        self.head = nn.Sequential(
            nn.Conv2d(b, 2, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor[B, 1, H, W] in [0, 1]

        Returns:
            Tensor[B, 2, H, W] in [-1, 1]
        """
        # Stem
        feat = self.stem(x)

        # Parallel branches
        b1 = self.branch1(feat)
        b2 = self.branch2(feat)
        b3 = self.branch3(feat)
        b4 = self.branch4(feat)

        # Concat along channel dim & fuse
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        fused = self.fusion(concat)

        # Local refinement
        refined = self.refinement(fused)

        # Output head
        out = self.head(refined)
        return out


def test_model():
    """
    Instantiates ColorizationUNet with a dummy input and asserts output shape.
    Tests on CPU; no GPU required.
    """
    print("Running model tests...")

    BATCH = 2
    H = W = 256

    model = ColorizationUNet(base_channels=64)
    model.eval()

    dummy_input = torch.randn(BATCH, 1, H, W)  # (B, 1, H, W)

    with torch.no_grad():
        output = model(dummy_input)  # (B, 2, H, W)

    expected = (BATCH, 2, H, W)
    assert output.shape == expected, (
        f"Shape mismatch: expected {expected}, got {tuple(output.shape)}"
    )

    # Output should be in [-1, 1] due to Tanh
    assert float(output.min()) >= -1.0 - 1e-5, (
        f"Output min {output.min()} outside [-1, 1]"
    )
    assert float(output.max()) <= 1.0 + 1e-5, (
        f"Output max {output.max()} outside [-1, 1]"
    )

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [PASS] Output shape: {tuple(output.shape)}")
    print(f"  [PASS] Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  [INFO] Trainable parameters: {n_params:,}")
    print()
    print(f"[PASS] Model test passed! Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    test_model()
