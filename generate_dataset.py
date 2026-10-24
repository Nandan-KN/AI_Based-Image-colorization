"""
generate_dataset.py -- Creates 500 synthetic training images (no internet needed).
Saves them as PNG to data/train/.
"""

import os
import numpy as np
from PIL import Image

def generate_synthetic_images(dest_dir: str, num_images: int = 500, image_size: int = 256):
    os.makedirs(dest_dir, exist_ok=True)
    existing = [f for f in os.listdir(dest_dir) if f.endswith(".png")]
    if len(existing) >= num_images:
        print(f"[DATA] {len(existing)} images already exist in '{dest_dir}', skipping.")
        return

    rng = np.random.default_rng(seed=1234)
    print(f"[DATA] Generating {num_images} synthetic RGB images -> '{dest_dir}' ...")

    for i in range(num_images):
        # Combine gradients + noise to get variety in hue/saturation cues
        H = W = image_size
        r_base = rng.integers(30, 230)
        g_base = rng.integers(30, 230)
        b_base = rng.integers(30, 230)

        grad_h = np.linspace(0, 1, H)[:, None]
        grad_w = np.linspace(0, 1, W)[None, :]

        r = np.clip(r_base * grad_h + rng.integers(0, 60, (H, W)), 0, 255).astype(np.uint8)
        g = np.clip(g_base * grad_w + rng.integers(0, 60, (H, W)), 0, 255).astype(np.uint8)
        b = np.clip(b_base * (1 - grad_h) + rng.integers(0, 60, (H, W)), 0, 255).astype(np.uint8)

        rgb = np.stack([r, g, b], axis=-1)
        pil = Image.fromarray(rgb, mode="RGB")
        pil.save(os.path.join(dest_dir, f"synth_{i:05d}.png"))

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_images}")

    print(f"[DATA] Done. {num_images} images saved.")

if __name__ == "__main__":
    dest = os.path.join(os.path.dirname(__file__), "data", "train")
    generate_synthetic_images(dest, num_images=500)
