"""
Basic sanity tests -- run with:
    pytest tests/
"""
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import DualStreamDeepfakeDetector
from dataset import compute_fft_magnitude
import numpy as np


def test_model_forward_shape():
    model = DualStreamDeepfakeDetector(pretrained=False, freeze_early_layers=False)
    model.eval()
    rgb = torch.randn(2, 3, 224, 224)
    freq = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        out = model(rgb, freq)
    assert out.shape == (2, 2)


def test_fft_magnitude_range():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    mag = compute_fft_magnitude(img)
    assert mag.shape == (64, 64)
    assert mag.min() >= 0.0 - 1e-6
    assert mag.max() <= 1.0 + 1e-6


def test_gradcam_target_layer_exists():
    model = DualStreamDeepfakeDetector(pretrained=False, freeze_early_layers=False)
    layer = model.get_gradcam_target_layer()
    assert layer is not None
