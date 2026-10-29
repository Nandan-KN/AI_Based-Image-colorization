# AI Image Colorization

A PyTorch implementation of automatic image colorization using a **Multi-Scale Dilated Residual CNN (MS-DRCNN)** in the **LAB color space**.  
The model takes a grayscale **L channel** as input and predicts the **a** and **b** color channels while preserving full spatial resolution end-to-end without downsampling or skip connections.

---

## Architecture: Multi-Scale Dilated Residual CNN (MS-DRCNN)

```
Input (L channel, 1×H×W)
        │
   [Stem Conv] Conv2d(1 -> 64, k=3, p=1) + BatchNorm + ReLU
        │
   ┌────┴──────────────────────────┬──────────────────────────┬──────────────────────────┐
   │                               │                          │                          │
[Branch 1: d=1]             [Branch 2: d=2]            [Branch 3: d=4]            [Branch 4: d=8]
2x ResidualBlock            2x ResidualBlock           2x ResidualBlock           2x ResidualBlock
   │                               │                          │                          │
   └────┬──────────────────────────┴──────────────────────────┴──────────────────────────┘
        │
  [Channel Concat] (4 × 64 = 256 channels)
        │
  [Fusion Conv] Conv2d(256 -> 64, k=1) + BatchNorm + ReLU
        │
  [Local Refinement] 2x ResidualBlock (d=1, 64 channels)
        │
  [Output Head] Conv2d(64 -> 2, k=1) -> Tanh()
        │
Output (ab channels, 2×H×W in [-1, 1])
        │
  Recombine: L + ab -> LAB -> RGB
```

### Specifications
| Hyperparameter | Value |
|---|---|
| Trainable Parameters | 757,186 (~0.76M) |
| Spatial Resolution | 256×256 preserved end-to-end (no downsampling) |
| Loss Function | Mean Squared Error (MSE) on ab channels |
| Optimizer | Adam (lr=1e-3) |
| Channel Ranges | L ∈ [0, 1] ; ab ∈ [-1, 1] |

---

## Project Structure

```
colorizer/
├── dataset.py              # ColorizationDataset + LAB conversion utilities
├── model.py                # Multi-Scale Dilated Residual CNN (MS-DRCNN)
├── train.py                # Training loop with batch logging + smoke test
├── app.py                  # Gradio demo (loads checkpoint if present)
├── generate_dataset.py     # Generates synthetic training images
├── run_demo_inference.py   # Runs inference on test image -> outputs/
├── data/
│   └── train/              # Training images (generated locally)
├── outputs/
│   ├── test_input_gray.png # Grayscale input image
│   └── test_colorized.png  # Colorized output image
├── colorizer_custom.pth    # Trained MS-DRCNN checkpoint (~3.08 MB)
└── requirements.txt        # Pinned dependencies
```

---

## Setup

```bash
git clone https://github.com/Nandan-KN/AI_Based-Image-colorization.git
cd AI_Based-Image-colorization

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Verification Gates

```bash
# Gate 1 — Model shape, range, and parameter count
python -c "from model import *; test_model()"

# Gate 2 — Smoke test (1 epoch on synthetic data)
python train.py --smoke-test

# Gate 3 — Inference pipeline integration
python -c "from app import *; test_inference()"

# Gate 4 — Training run
python train.py --epochs 3 --batch-size 8 --max-images 200 --checkpoint colorizer_custom.pth

# Gate 5 — End-to-end demo inference
python run_demo_inference.py
```

---

## Running the Gradio Demo

```bash
python app.py
```

Opens an interactive web app to upload grayscale images and view the predicted colorized output in real time.

---

## Training Results

Trained on 200 images for 3 epochs:

| Epoch | Train Loss (MSE) | Val Loss (MSE) |
|:---:|:---:|:---:|
| 1 | 0.096528 | 0.163564 |
| 2 | 0.035569 | 0.260624 |
| **3** | **0.028515** | **0.065218** |

- **Best Val Loss:** `0.065218`
- **Output Sample:** generated in `outputs/test_colorized.png`
