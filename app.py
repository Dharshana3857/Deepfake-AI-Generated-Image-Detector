"""
Interactive demo UI.

Usage:
    streamlit run app/app.py
"""
import os
import sys
import numpy as np
import cv2
import torch
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset import build_transforms, compute_fft_magnitude
from models import build_model
from gradcam import GradCAM, overlay_heatmap
from utils import load_config, get_device, load_checkpoint

st.set_page_config(page_title="Deepfake Image Detector", page_icon="🕵️", layout="centered")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


@st.cache_resource
def load_model():
    cfg = load_config(CONFIG_PATH)
    device = get_device()
    model = build_model(cfg).to(device)
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", cfg["inference"]["default_checkpoint"])
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model, map_location=device)
        loaded = True
    else:
        loaded = False
    model.eval()
    return model, cfg, device, loaded


def predict(model, cfg, device, image_rgb):
    size = cfg["data"]["image_size"]
    img_resized = cv2.resize(image_rgb, (size, size))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    freq = compute_fft_magnitude(gray)
    freq_tensor = torch.from_numpy(freq).unsqueeze(0).unsqueeze(0).float().to(device)

    transform = build_transforms(size, train=False)
    rgb_tensor = transform(image=image_rgb)["image"].unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(rgb_tensor, freq_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    cam_engine = GradCAM(model, model.get_gradcam_target_layer())
    cam, class_idx = cam_engine.generate(rgb_tensor, freq_tensor)
    overlay = overlay_heatmap(img_resized, cam)

    return probs, overlay, freq


st.title("🕵️ Deepfake / AI-Generated Image Detector")
st.caption("Dual-stream CNN: spatial (EfficientNet) + frequency-domain (FFT) branch, with Grad-CAM explainability.")

model, cfg, device, loaded = load_model()

if not loaded:
    st.warning(
        "No trained checkpoint found yet at `checkpoints/best_model.pt`. "
        "Predictions below use an untrained model and are meaningless until you train one:\n\n"
        "```\npython generate_sample_data.py\npython src/train.py\n```"
    )

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Running dual-stream inference..."):
            probs, overlay, freq_spectrum = predict(model, cfg, device, img_rgb)

        pred_label = "FAKE" if probs[1] > probs[0] else "REAL"
        confidence = max(probs)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", pred_label, f"{confidence:.1%} confidence")
        with col2:
            st.write(f"P(real) = {probs[0]:.3f}")
            st.write(f"P(fake) = {probs[1]:.3f}")

        st.subheader("Grad-CAM: what the model looked at")
        st.image(overlay, use_column_width=True)

        st.subheader("Frequency-domain spectrum (FFT input)")
        st.image(freq_spectrum, use_column_width=True, clamp=True)

st.divider()
st.caption(
    "This is a college/research demo, not a production forensics tool. "
    "Accuracy depends entirely on the dataset used for training."
)
