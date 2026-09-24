"""
Generates a small, fully OFFLINE demo dataset so you can run the entire
pipeline (train -> evaluate -> predict -> Streamlit app) immediately,
with zero downloads and zero external datasets.

It builds "real" images from scikit-image's built-in sample photos, and
generates "fake" counterparts by injecting periodic upsampling-style
artifacts in the frequency domain -- the same kind of spectral fingerprint
that real GAN/diffusion-generated images exhibit. This lets the dual-stream
(RGB + FFT) model actually learn something meaningful on toy data, instead
of training on random noise.

IMPORTANT: this demo dataset is only for smoke-testing the pipeline end to
end. For a real project result, replace data/raw/real and data/raw/fake with
an actual dataset -- see download_dataset.py / README.md for options
(140k Real and Fake Faces, FaceForensics++, DFDC, etc).

Usage:
    python generate_sample_data.py --count 150
"""
import os
import argparse
import numpy as np
import cv2
from skimage import data as skdata
from skimage.transform import resize, rotate
from skimage.util import img_as_ubyte

OUT_ROOT = os.path.join("data", "raw")


def base_images():
    """A handful of built-in scikit-image sample photos (ships with the package, no internet)."""
    candidates = [
        skdata.astronaut(), skdata.chelsea(), skdata.coffee(), skdata.rocket(),
        skdata.hubble_deep_field(), skdata.horse(), skdata.cat() if hasattr(skdata, "cat") else skdata.chelsea(),
    ]
    imgs = []
    for im in candidates:
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        if im.shape[-1] == 4:
            im = im[..., :3]
        imgs.append(img_as_ubyte(im))
    return imgs


def random_augment(img, size=256):
    img = resize(img, (size, size), anti_aliasing=True)
    angle = np.random.uniform(-25, 25)
    img = rotate(img, angle, mode="edge")
    if np.random.rand() < 0.5:
        img = img[:, ::-1, :]
    # random crop jitter
    pad = int(size * 0.1)
    y0 = np.random.randint(0, pad)
    x0 = np.random.randint(0, pad)
    img = img[y0:y0 + size - pad, x0:x0 + size - pad]
    img = resize(img, (size, size), anti_aliasing=True)
    return img_as_ubyte(np.clip(img, 0, 1))


def inject_gan_style_artifacts(img):
    """
    Simulates transposed-convolution / upsampling checkerboard artifacts by
    adding structured periodic noise in the frequency domain, plus a mild
    smoothing pass (GANs/diffusion models tend to over-smooth high-frequency
    texture). This is a pedagogical stand-in for real generator fingerprints,
    NOT an actual GAN.
    """
    out = img.astype(np.float32)
    h, w, c = out.shape

    # mild blur -> mimics over-smoothing common in generated images
    out = cv2.GaussianBlur(out, (3, 3), 0.6)

    # periodic checkerboard-style pattern injected per channel via FFT
    for ch in range(c):
        f = np.fft.fft2(out[:, :, ch])
        fshift = np.fft.fftshift(f)

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cy, cx = h // 2, w // 2
        period = np.random.uniform(6, 10)
        ring = np.sin(2 * np.pi * (np.abs(yy - cy) + np.abs(xx - cx)) / period)
        boost = 1.0 + 0.06 * ring

        fshift *= boost
        f_ishift = np.fft.ifftshift(fshift)
        out[:, :, ch] = np.real(np.fft.ifft2(f_ishift))

    out = np.clip(out, 0, 255).astype(np.uint8)

    # slight color/contrast shift, another common generator tell
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= np.random.uniform(0.85, 1.15)
    hsv[:, :, 2] *= np.random.uniform(0.9, 1.1)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150, help="images per class")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    real_dir = os.path.join(OUT_ROOT, "real")
    fake_dir = os.path.join(OUT_ROOT, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    sources = base_images()
    print(f"Generating {args.count} real + {args.count} fake demo images...")

    for i in range(args.count):
        src = sources[i % len(sources)]
        real_img = random_augment(src, size=args.size)
        cv2.imwrite(os.path.join(real_dir, f"real_{i:04d}.jpg"),
                    cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR))

        fake_src = sources[(i + 1) % len(sources)]
        fake_base = random_augment(fake_src, size=args.size)
        fake_img = inject_gan_style_artifacts(fake_base)
        cv2.imwrite(os.path.join(fake_dir, f"fake_{i:04d}.jpg"),
                    cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR))

    print(f"Done. Wrote images to '{real_dir}' and '{fake_dir}'.")
    print("This is a SMOKE-TEST dataset only. For real results, swap in an actual")
    print("dataset -- run: python download_dataset.py --help")


if __name__ == "__main__":
    main()
