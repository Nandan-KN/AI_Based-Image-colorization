"""
app.py -- Gradio demo for the LAB colorization U-Net.
Designed for Hugging Face Spaces deployment.

Model runs with RANDOM WEIGHTS (no checkpoint loaded).
This demonstrates the inference pipeline and color space mechanics.
For production colorization, train on a real photo dataset and
replace this file with one that loads a trained checkpoint.

Test inference only (no UI):
    python -c "from app import *; test_inference()"
"""

import numpy as np
import torch
from PIL import Image

from dataset import lab_tensors_to_rgb
from model import ColorizationUNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_IMAGE_SIZE = 256


# ---------------------------------------------------------------------------
# Model (random weights, loaded once)
# ---------------------------------------------------------------------------

_model_cache: ColorizationUNet | None = None

def _get_model() -> ColorizationUNet:
    global _model_cache
    if _model_cache is None:
        print("[APP] Initializing model with random weights ...")
        model = ColorizationUNet(base_channels=64)
        model.eval()
        _model_cache = model
        print("[APP] Model ready.")
    return _model_cache


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _pil_to_L_tensor(pil_img: Image.Image, image_size: int = _IMAGE_SIZE) -> torch.Tensor:
    """Converts a PIL image to a normalized L-channel tensor (1, 1, H, W)."""
    from skimage import color as sk_color

    img    = pil_img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    rgb_np = np.array(img, dtype=np.float32) / 255.0
    lab    = sk_color.rgb2lab(rgb_np).astype(np.float32)
    L_np   = lab[:, :, 0] / 100.0                           # [0, 1]
    return torch.from_numpy(L_np).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)


def colorize_image(pil_img: Image.Image) -> Image.Image:
    """
    Main inference function.
    Takes any PIL image, returns a colorized RGB PIL image (256x256).
    Uses randomly initialized weights — output color is architecturally
    driven but not learned.
    """
    model    = _get_model()
    L_tensor = _pil_to_L_tensor(pil_img)   # (1, 1, H, W)

    with torch.no_grad():
        ab_pred = model(L_tensor)           # (1, 2, H, W)

    rgb_np = lab_tensors_to_rgb(
        L_tensor.squeeze(0),               # (1, H, W)
        ab_pred.squeeze(0),                # (2, H, W)
    )
    return Image.fromarray(rgb_np)


# ---------------------------------------------------------------------------
# Verification test
# ---------------------------------------------------------------------------

def test_inference():
    """Synthetic test: dummy grayscale PIL -> colorized RGB PIL."""
    print("Running inference test ...")
    H = W = _IMAGE_SIZE
    rng    = np.random.default_rng(seed=0)
    gray   = rng.integers(0, 256, (H, W), dtype=np.uint8)
    pil_in = Image.fromarray(gray, mode="L")

    pil_out = colorize_image(pil_in)

    assert isinstance(pil_out, Image.Image), f"Expected PIL Image, got {type(pil_out)}"
    assert pil_out.mode == "RGB",            f"Expected RGB, got {pil_out.mode}"
    assert pil_out.size == (W, H),           f"Expected ({W},{H}), got {pil_out.size}"

    print(f"  [PASS] Output size: {pil_out.size}  Mode: {pil_out.mode}")
    print("[PASS] Inference test passed!")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _gradio_colorize(pil_img: Image.Image) -> Image.Image:
    if pil_img is None:
        return None
    try:
        return colorize_image(pil_img)
    except Exception as e:
        print(f"[APP] Inference error: {e}")
        raise


def launch_demo():
    import gradio as gr

    demo = gr.Interface(
        fn=_gradio_colorize,
        inputs=gr.Image(type="pil", label="Upload a grayscale (or color) image"),
        outputs=gr.Image(type="pil", label="Colorized output"),
        title="AI Image Colorization (Architecture Demo)",
        description=(
            "Upload a grayscale image. The **U-Net** predicts color using the **LAB color space**.\n\n"
            "> This demo uses **randomly initialized weights** to showcase the pipeline architecture.\n"
            "> To get meaningful colorization, train the model on real photos using `train.py`.\n\n"
            "- Model: 4-level U-Net encoder-decoder (31M params)\n"
            "- Input: L channel → Output: predicted ab channels → Recombined to RGB\n"
            "- No checkpoint loaded — weights are random"
        ),
        allow_flagging="never",
        examples=[],
    )
    demo.launch()


if __name__ == "__main__":
    launch_demo()
