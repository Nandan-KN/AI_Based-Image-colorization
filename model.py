"""
model.py -- Colorization U-Net
Encoder-decoder with skip connections.
Input  : (B, 1, H, W)  -- L channel, normalized [0, 1]
Output : (B, 2, H, W)  -- predicted ab channels, in [-1, 1]
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling step: MaxPool2d(2) then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upsampling step: ConvTranspose2d(stride=2) then DoubleConv.
    Concatenates the matching encoder skip feature map.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # in_channels = channels from previous decoder level
        # after concat with skip: in_channels + skip_channels = in_channels * 2 // 2 + in_channels // 2
        # We keep it simple: upsample halves the channels, then concat doubles them again.
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                        kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)                           # (B, C//2, H*2, W*2)

        # Handle odd spatial dimensions (H or W not divisible by 2)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            x = nn.functional.pad(x, [dw // 2, dw - dw // 2,
                                       dh // 2, dh - dh // 2])

        x = torch.cat([skip, x], dim=1)          # (B, C, H, W)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class ColorizationUNet(nn.Module):
    """
    4-level U-Net for image colorization.

    Architecture (channels):
        Encoder:  1  -> 64 -> 128 -> 256 -> 512
        Bottleneck:       512 -> 1024
        Decoder: 1024 -> 512 -> 256 -> 128 -> 64
        Head:      64 -> 2  (ab channels)
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        b = base_channels

        # Encoder
        self.enc1 = DoubleConv(1,      b)        # (B,  64, H,    W)
        self.enc2 = Down(b,     b * 2)           # (B, 128, H/2,  W/2)
        self.enc3 = Down(b * 2, b * 4)           # (B, 256, H/4,  W/4)
        self.enc4 = Down(b * 4, b * 8)           # (B, 512, H/8,  W/8)

        # Bottleneck
        self.bottleneck = Down(b * 8, b * 16)    # (B,1024, H/16, W/16)

        # Decoder
        self.dec4 = Up(b * 16, b * 8)            # (B, 512, H/8,  W/8)
        self.dec3 = Up(b * 8,  b * 4)            # (B, 256, H/4,  W/4)
        self.dec2 = Up(b * 4,  b * 2)            # (B, 128, H/2,  W/2)
        self.dec1 = Up(b * 2,  b)                # (B,  64, H,    W)

        # Output head: predict 2 ab channels, tanh to keep in [-1, 1]
        self.head = nn.Sequential(
            nn.Conv2d(b, 2, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor[B, 1, H, W]  -- L channel

        Returns:
            Tensor[B, 2, H, W]  -- predicted ab channels
        """
        # Encode
        s1 = self.enc1(x)           # (B,  64, H,    W)
        s2 = self.enc2(s1)          # (B, 128, H/2,  W/2)
        s3 = self.enc3(s2)          # (B, 256, H/4,  W/4)
        s4 = self.enc4(s3)          # (B, 512, H/8,  W/8)

        # Bottleneck
        b  = self.bottleneck(s4)    # (B,1024, H/16, W/16)

        # Decode with skip connections
        d4 = self.dec4(b,  s4)      # (B, 512, H/8,  W/8)
        d3 = self.dec3(d4, s3)      # (B, 256, H/4,  W/4)
        d2 = self.dec2(d3, s2)      # (B, 128, H/2,  W/2)
        d1 = self.dec1(d2, s1)      # (B,  64, H,    W)

        return self.head(d1)        # (B,   2, H,    W)


# ---------------------------------------------------------------------------
# Verification test (Phase 3)
# ---------------------------------------------------------------------------

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

    dummy_input = torch.randn(BATCH, 1, H, W)   # (B, 1, H, W)

    with torch.no_grad():
        output = model(dummy_input)              # (B, 2, H, W)

    expected = (BATCH, 2, H, W)
    assert output.shape == expected, \
        f"Shape mismatch: expected {expected}, got {tuple(output.shape)}"

    # Output should be in [-1, 1] due to Tanh
    assert float(output.min()) >= -1.0 - 1e-5, \
        f"Output min {output.min()} outside [-1, 1]"
    assert float(output.max()) <=  1.0 + 1e-5, \
        f"Output max {output.max()} outside [-1, 1]"

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [PASS] Output shape: {tuple(output.shape)}")
    print(f"  [PASS] Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  [INFO] Trainable parameters: {n_params:,}")
    print()
    print(f"[PASS] Model test passed! Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    test_model()
