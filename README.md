# AI Image Colorization

A PyTorch implementation of automatic image colorization using a **U-Net encoder-decoder** in the **LAB color space**.  
The model takes a grayscale **L channel** as input and predicts the **a** and **b** color channels.

---

## Architecture

```
Input (L channel, 1×H×W)
        │
  [Encoder — 4 down-blocks]
  64 → 128 → 256 → 512 channels, MaxPool stride 2
        │
  [Bottleneck] 512 → 1024 channels
        │
  [Decoder — 4 up-blocks]
  ConvTranspose2d + skip connection from matching encoder
  1024 → 512 → 256 → 128 → 64
        │
  [Output head] Conv1×1 → 2 channels → Tanh
        │
Output (ab channels, 2×H×W)
        │
  Combine L + ab → LAB → RGB
```

| Hyperparameter | Value |
|---|---|
| Trainable params | 31,036,546 |
| Loss | MSE on ab channels |
| Optimizer | Adam (lr=1e-3) |
| Input size | 256×256 |
| Normalization | L ∈ [0,1] ; ab ∈ [-1,1] |

---

## Project Structure

```
colorizer/
├── dataset.py              # ColorizationDataset + LAB utilities
├── model.py                # ColorizationUNet (U-Net)
├── train.py                # Training loop + smoke test
├── app.py                  # Gradio demo (upload → colorize)
├── generate_dataset.py     # Generates synthetic training images
├── run_demo_inference.py   # Runs inference on one image, saves to outputs/
├── data/
│   └── train/              # Training images (500 synthetic PNGs)
├── outputs/
│   ├── test_input_gray.png # Grayscale test input
│   └── test_colorized.png  # Colorized model output
└── colorizer.pth           # Best model checkpoint (~124 MB)
```

---

## Setup

All dependencies are standard; no extra installs needed if you have the base environment:

```
Python      3.12.10
torch       2.14.1
torchvision 0.29.1
scikit-image 0.26.0
Pillow      12.3.0
gradio      6.29.1
numpy       2.5.3
```

To create a fresh virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install torch torchvision scikit-image pillow gradio numpy
```

---

## Verification (one command per phase)

```bash
# Phase 2 — Dataset
python -c "from dataset import *; test_dataset()"

# Phase 3 — Model
python -c "from model import *; test_model()"

# Phase 4 — Smoke test (1 epoch, synthetic data)
python train.py --smoke-test

# Phase 5 — App inference
python -c "from app import *; test_inference()"
```

---

## Training

### Quick smoke test (1 epoch, synthetic data, ~1 min)

```bash
python train.py --smoke-test
```

### Full training (5 epochs, 500 images)

```bash
python train.py --epochs 5 --batch-size 16 --max-images 500
```

The best checkpoint (lowest val loss) is saved to `colorizer.pth`.

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 5 | Number of training epochs |
| `--batch-size` | 16 | Images per batch |
| `--image-size` | 256 | Resize images to N×N |
| `--lr` | 1e-3 | Adam learning rate |
| `--max-images` | 500 | Training images to generate |
| `--checkpoint` | colorizer.pth | Output checkpoint file |
| `--smoke-test` | — | 1 epoch, 10 samples, fast validation |

---

## Running the Demo App

```bash
python app.py
```

Opens a Gradio web UI at `http://127.0.0.1:7860`.  
Upload a grayscale (or color) image — the model predicts and displays the colorized version.

### Batch / script inference

```bash
python run_demo_inference.py
# Saves: outputs/test_input_gray.png + outputs/test_colorized.png
```

---

## Results (5-epoch training on 500 synthetic images, CPU)

### Loss curve

| Epoch | Train Loss (MSE) | Val Loss (MSE) |
|:---:|:---:|:---:|
| 1 | 0.062083 | 0.092319 |
| 2 | 0.036913 | 0.116014 |
| 3 | 0.032109 | 0.029711 |
| 4 | 0.027046 | 0.041982 |
| **5** | **0.018541** | **0.018937** |

- Training loss decreased **70%** from epoch 1 to 5 (0.062 → 0.018).
- Val loss converged to **0.0189** (best checkpoint saved at epoch 5).

### Before / After

| Input (grayscale L) | Output (colorized RGB) |
|---|---|
| `outputs/test_input_gray.png` | `outputs/test_colorized.png` |

> **Note:** The synthetic training dataset (gradient + noise images) teaches the model the
> LAB color mapping mechanics. For photo-realistic colorization of natural images,
> replace `data/train/` with real photographs (e.g., Places365, ImageNet).

---

## Extending the Project

| Idea | How |
|---|---|
| Real dataset | Drop JPEGs into `data/train/` and re-run `train.py` |
| More epochs | `--epochs 50` |
| Perceptual loss | Replace MSE with LPIPS or VGG feature loss |
| Larger U-Net | `--base-channels 128` (edit `train.py`) |
| GPU training | Runs automatically if `torch.cuda.is_available()` |
