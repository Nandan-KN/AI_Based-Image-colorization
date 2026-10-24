"""
dataset.py — Colorization Dataset
Loads RGB images from a directory, converts to LAB color space,
and returns (L, ab) tensors for training the colorization U-Net.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import color


class ColorizationDataset(Dataset):
    """
    Custom Dataset for image colorization.

    Each item returns:
        L  : Tensor[1, H, W]  — normalized to [0, 1]
        ab : Tensor[2, H, W]  — normalized to [-1, 1]
    """

    def __init__(self, image_dir: str, image_size: int = 256):
        """
        Args:
            image_dir  : Path to folder containing RGB images.
            image_size : Target spatial size (images resized to H=W=image_size).
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in '{image_dir}'.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]

        # Load image, convert to RGB (handles grayscale / RGBA inputs)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        return self._to_lab_tensors(np.array(img, dtype=np.float32))

    @staticmethod
    def _to_lab_tensors(rgb_np: np.ndarray):
        """
        Converts a (H, W, 3) float32 RGB array (values 0-255) to LAB tensors.

        Returns:
            L_tensor  : Tensor[1, H, W]  — L channel, normalized [0, 1]
            ab_tensor : Tensor[2, H, W]  — ab channels, normalized [-1, 1]
        """
        # skimage expects float64 RGB in [0, 1]
        rgb_01 = rgb_np / 255.0

        # Convert RGB → LAB
        # L  ∈ [0, 100]
        # a  ∈ [-128, 127]
        # b  ∈ [-128, 127]
        lab = color.rgb2lab(rgb_01).astype(np.float32)

        L  = lab[:, :, 0]   # (H, W)
        ab = lab[:, :, 1:]  # (H, W, 2)

        # Normalize to model-friendly ranges
        L_norm  = L / 100.0                     # [0, 1]
        ab_norm = ab / 128.0                     # [-1, 1]  (approx)

        # Convert to tensors: (C, H, W)
        L_tensor  = torch.from_numpy(L_norm).unsqueeze(0)          # (1, H, W)
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1)     # (2, H, W)

        return L_tensor, ab_tensor


def get_dataloader(
    image_dir: str,
    image_size: int = 256,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience factory: returns a DataLoader for ColorizationDataset.

    Args:
        image_dir   : Path to folder of RGB images.
        image_size  : Resize target (square).
        batch_size  : Samples per batch.
        shuffle     : Whether to shuffle each epoch.
        num_workers : DataLoader worker processes (0 = main process).

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = ColorizationDataset(image_dir, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Utility: lab tensors → RGB PIL Image (used by app.py)
# ---------------------------------------------------------------------------

def lab_tensors_to_rgb(L_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts (1, H, W) L tensor and (2, H, W) ab tensor back to an (H, W, 3)
    uint8 RGB numpy array.

    Args:
        L_tensor  : Tensor[1, H, W], values in [0, 1]
        ab_tensor : Tensor[2, H, W], values in [-1, 1]

    Returns:
        rgb_uint8 : np.ndarray of shape (H, W, 3), dtype uint8
    """
    # Detach and move to CPU
    L_np  = L_tensor.squeeze(0).detach().cpu().numpy()   # (H, W)
    ab_np = ab_tensor.detach().cpu().numpy()              # (2, H, W)

    # De-normalize
    L_denorm  = L_np * 100.0                             # [0, 100]
    ab_denorm = ab_np * 128.0                             # [-128, 128]

    # Stack to (H, W, 3)
    H, W = L_denorm.shape
    lab = np.zeros((H, W, 3), dtype=np.float32)
    lab[:, :, 0]  = L_denorm
    lab[:, :, 1:] = ab_denorm.transpose(1, 2, 0)

    # Convert LAB → RGB (clip to [0, 1])
    rgb_01    = np.clip(color.lab2rgb(lab), 0.0, 1.0)
    rgb_uint8 = (rgb_01 * 255).astype(np.uint8)
    return rgb_uint8


# ---------------------------------------------------------------------------
# Verification test (Phase 2)
# ---------------------------------------------------------------------------

def test_dataset():
    """
    Tests the dataset pipeline with purely synthetic data (no real images).
    Asserts output tensor shapes and value ranges without touching disk.
    """
    print("Running dataset tests...")

    BATCH  = 4
    H = W  = 256

    # ---- Test 1: _to_lab_tensors shape and range -------------------------
    dummy_rgb = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8).astype(np.float32)
    L_t, ab_t = ColorizationDataset._to_lab_tensors(dummy_rgb)

    assert L_t.shape  == (1, H, W), f"Expected L shape (1,{H},{W}), got {L_t.shape}"
    assert ab_t.shape == (2, H, W), f"Expected ab shape (2,{H},{W}), got {ab_t.shape}"
    assert float(L_t.min())  >= -0.01, f"L min out of range: {L_t.min()}"
    assert float(L_t.max())  <=  1.01, f"L max out of range: {L_t.max()}"
    assert float(ab_t.min()) >= -1.01, f"ab min out of range: {ab_t.min()}"
    assert float(ab_t.max()) <=  1.01, f"ab max out of range: {ab_t.max()}"
    print(f"  [PASS] _to_lab_tensors - L: {tuple(L_t.shape)}  ab: {tuple(ab_t.shape)}")

    # ---- Test 2: Batched tensor simulation (mimics DataLoader output) -----
    L_batch  = torch.stack([L_t  for _ in range(BATCH)])   # (B, 1, H, W)
    ab_batch = torch.stack([ab_t for _ in range(BATCH)])   # (B, 2, H, W)

    assert L_batch.shape  == (BATCH, 1, H, W), \
        f"Batched L shape mismatch: {L_batch.shape}"
    assert ab_batch.shape == (BATCH, 2, H, W), \
        f"Batched ab shape mismatch: {ab_batch.shape}"
    print(f"  [PASS] Batched tensors - L: {tuple(L_batch.shape)}  ab: {tuple(ab_batch.shape)}")

    # ---- Test 3: Round-trip LAB -> RGB reconstruction ---------------------
    rgb_out = lab_tensors_to_rgb(L_t, ab_t)
    assert rgb_out.shape == (H, W, 3),   f"RGB output shape mismatch: {rgb_out.shape}"
    assert rgb_out.dtype == np.uint8,    f"RGB dtype mismatch: {rgb_out.dtype}"
    print(f"  [PASS] lab_tensors_to_rgb - shape: {rgb_out.shape}  dtype: {rgb_out.dtype}")

    print()
    print(f"[PASS] Dataset test passed! L: {tuple(L_batch.shape)}  ab: {tuple(ab_batch.shape)}")


if __name__ == "__main__":
    test_dataset()
