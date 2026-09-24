"""
Run inference on a single image.

Usage:
    python src/predict.py --image path/to/photo.jpg
    python src/predict.py --image path/to/photo.jpg --gradcam --output outputs/explained.png
"""
import os
import sys
import argparse

import cv2
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import build_transforms, compute_fft_magnitude
from models import build_model
from gradcam import GradCAM, overlay_heatmap
from utils import load_config, get_device, load_checkpoint

LABELS = ["REAL", "FAKE"]


def preprocess(image_path, image_size):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized_for_display = cv2.resize(img_rgb, (image_size, image_size))

    gray = cv2.cvtColor(img_resized_for_display, cv2.COLOR_RGB2GRAY)
    freq = compute_fft_magnitude(gray)
    freq_tensor = torch.from_numpy(freq).unsqueeze(0).unsqueeze(0).float()

    transform = build_transforms(image_size, train=False)
    rgb_tensor = transform(image=img_rgb)["image"].unsqueeze(0).float()

    return rgb_tensor, freq_tensor, img_resized_for_display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--output", default="outputs/gradcam_result.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg["inference"]["default_checkpoint"]
    device = get_device()

    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    rgb_tensor, freq_tensor, display_img = preprocess(args.image, cfg["data"]["image_size"])
    rgb_tensor, freq_tensor = rgb_tensor.to(device), freq_tensor.to(device)

    with torch.no_grad():
        outputs = model(rgb_tensor, freq_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(probs.argmax())

    print(f"Prediction: {LABELS[pred_idx]}  (confidence: {probs[pred_idx]:.2%})")
    print(f"  P(real)={probs[0]:.4f}   P(fake)={probs[1]:.4f}")

    if args.gradcam:
        cam_engine = GradCAM(model, model.get_gradcam_target_layer())
        cam, class_idx = cam_engine.generate(rgb_tensor, freq_tensor)
        overlay = overlay_heatmap(display_img, cam)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        cv2.imwrite(args.output, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM visualization saved to {args.output}")


if __name__ == "__main__":
    main()
