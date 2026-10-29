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

import os
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
# Model (loads custom checkpoint if present, else random weights)
# ---------------------------------------------------------------------------

_model_cache: ColorizationUNet | None = None

def _get_model() -> ColorizationUNet:
    global _model_cache
    if _model_cache is None:
        model = ColorizationUNet(base_channels=64)
        ckpt_loaded = False
        for ckpt_name in ["colorizer_custom.pth", "colorizer.pth"]:
            ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_name)
            if os.path.exists(ckpt_path):
                try:
                    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                    model.load_state_dict(state)
                    print(f"[APP] Loaded checkpoint: {ckpt_path}")
                    ckpt_loaded = True
                    break
                except Exception as e:
                    print(f"[APP] Warning loading {ckpt_path}: {e}")
        if not ckpt_loaded:
            print("[APP] Initializing model with random weights ...")
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
        title="AI Image Colorization (MS-DRCNN)",
        description=(
            "Upload a grayscale image. The **Multi-Scale Dilated Residual CNN (MS-DRCNN)** "
            "predicts color in the **LAB color space** preserving full spatial resolution.\n\n"
            "- Architecture: MS-DRCNN (757K params, dilations 1, 2, 4, 8)\n"
            "- Checkpoint: Automatically loads `colorizer_custom.pth` if present\n"
            "- Channels: L (grayscale) → predicted ab → RGB reconstruction"
        ),
        flagging_mode="never",
        examples=[],
    )
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    launch_demo()
