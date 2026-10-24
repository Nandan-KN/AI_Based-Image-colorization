"""
run_demo_inference.py -- Phase 6 helper.
Loads a test image (or generates a synthetic one), runs colorization,
and saves the result to outputs/test_colorized.png.
"""

import os
import numpy as np
from PIL import Image
from app import colorize_image

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_and_save():
    print("[DEMO] Generating synthetic test image ...")
    # Create a synthetic grayscale gradient image (looks like a real photo scene)
    H = W = 256
    rng = np.random.default_rng(seed=42)
    # Combine gradient + noise for a realistic-ish grayscale
    gradient_h = np.linspace(0, 255, H, dtype=np.float32)[:, None] * np.ones((H, W), dtype=np.float32)
    gradient_v = np.linspace(0, 255, W, dtype=np.float32)[None, :] * np.ones((H, W), dtype=np.float32)
    noise      = rng.integers(0, 40, (H, W)).astype(np.float32)
    gray_np    = np.clip((gradient_h * 0.4 + gradient_v * 0.4 + noise) / 1.08, 0, 255).astype(np.uint8)

    pil_gray = Image.fromarray(gray_np, mode="L")
    pil_gray.save(os.path.join(OUTPUT_DIR, "test_input_gray.png"))
    print(f"[DEMO] Saved grayscale input -> outputs/test_input_gray.png")

    print("[DEMO] Running colorization ...")
    pil_color = colorize_image(pil_gray)
    out_path  = os.path.join(OUTPUT_DIR, "test_colorized.png")
    pil_color.save(out_path)
    print(f"[DEMO] Saved colorized output -> outputs/test_colorized.png")
    print("[DEMO] Done.")

if __name__ == "__main__":
    run_and_save()
