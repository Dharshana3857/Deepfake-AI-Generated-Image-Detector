# Deepfake-AI-Generated-Image-Detector
Dual-stream CNN (spatial + frequency-domain) for detecting AI-generated faces, with Grad-CAM explainability and a live Streamlit dashboard. 98.88% accuracy / 0.9995 AUC on StyleGAN-generated faces.
<div align="center">

# 🔮 Deepfake / AI-Generated Image Detector

**A dual-stream deep learning system for detecting AI-generated face images, combining spatial and frequency-domain analysis with Grad-CAM explainability.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

</div>

---

## Overview

This project implements a binary image classifier that distinguishes **real photographs** from **AI-generated (deepfake) faces**. Rather than relying on a single CNN over raw pixels, it uses a **dual-stream architecture**:

- A pretrained **EfficientNet-B0** branch analyzes the raw image for spatial artifacts — blending boundaries, unnatural lighting, warped geometry.
- A custom CNN branch analyzes the image's **FFT (frequency-domain) spectrum**, picking up periodic artifacts left behind by GAN upsampling — a signal invisible to the human eye but statistically detectable.

The two feature streams are fused into a single classifier head, and every prediction is paired with a **Grad-CAM heatmap**, so the model's reasoning can be visually inspected rather than trusted blindly.

The trained model is served through an interactive **Streamlit dashboard** for live testing and demonstration.

---

## Results

Trained on the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset (real FFHQ photographs vs. StyleGAN-generated faces), evaluated on a held-out test set of 30,000 images:

| Metric | Score |
|---|---|
| Accuracy | **98.88%** |
| Precision | 98.30% |
| Recall | 99.49% |
| F1-score | 98.89% |
| ROC-AUC | **0.9995** |

---

## Architecture
