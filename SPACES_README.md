---
title: AI Image Colorization
emoji: 🎨
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: apache-2.0
---

# AI Image Colorization

Upload a grayscale image and the U-Net model predicts plausible colors using the **LAB color space**.

- **Input**: Grayscale (L channel, 1×256×256)
- **Output**: Colorized RGB image (256×256)
- **Model**: 4-level U-Net encoder-decoder, 31M parameters, trained with MSE loss on ab channels

Trained on 500 synthetic images for 5 epochs. For best results with real photos, the model benefits from fine-tuning on a real photograph dataset (e.g., Places365).
