"""
Dataset for dual-stream deepfake detection.

Each sample yields:
  - rgb tensor        (3, H, W)  -> normal spatial stream
  - frequency tensor  (1, H, W)  -> log-magnitude FFT spectrum

Why frequency domain?
GAN / diffusion upsampling layers leave periodic checkerboard-style artifacts
that are hard to see in pixel space but show up clearly as peaks in the
Fourier spectrum. Feeding both streams to the network lets it learn spatial
*and* spectral tell-tale signs of synthetic images.
"""
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_fft_magnitude(gray_img: np.ndarray) -> np.ndarray:
    """Log-scaled, min-max normalized magnitude spectrum of a grayscale image."""
    f = np.fft.fft2(gray_img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    mn, mx = magnitude.min(), magnitude.max()
    magnitude = (magnitude - mn) / (mx - mn + 1e-8)
    return magnitude.astype(np.float32)


def build_transforms(image_size=224, train=True):
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.ImageCompression(quality_range=(60, 100), p=0.3),  # simulate re-compression
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def list_samples(root_dir):
    """
    Expects:
        root_dir/real/*.jpg|png
        root_dir/fake/*.jpg|png
    Returns list of (path, label) with label 0=real, 1=fake
    """
    samples = []
    for label, cls in enumerate(["real", "fake"]):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                samples.append((os.path.join(cls_dir, fname), label))
    random.Random(42).shuffle(samples)
    return samples


def split_samples(samples, val_split=0.15, test_split=0.15):
    n = len(samples)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    return train, val, test


class DeepfakeDataset(Dataset):
    def __init__(self, samples, image_size=224, train=True):
        self.samples = samples
        self.image_size = image_size
        self.transform = build_transforms(image_size, train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # Corrupt file fallback: return a black frame instead of crashing training
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.image_size, self.image_size))
        freq = compute_fft_magnitude(gray)
        freq_tensor = torch.from_numpy(freq).unsqueeze(0).float()

        augmented = self.transform(image=img)
        rgb_tensor = augmented["image"].float()

        return rgb_tensor, freq_tensor, torch.tensor(label, dtype=torch.long)
