"""
Evaluate a trained checkpoint on the held-out test split.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt
"""
import os
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import DeepfakeDataset, list_samples, split_samples
from models import build_model
from utils import load_config, get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg["inference"]["default_checkpoint"]
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    samples = list_samples(cfg["data"]["root_dir"])
    _, _, test_s = split_samples(samples, cfg["data"]["val_split"], cfg["data"]["test_split"])

    test_ds = DeepfakeDataset(test_s, cfg["data"]["image_size"], train=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for rgb, freq, labels in test_loader:
            rgb, freq = rgb.to(device), freq.to(device)
            outputs = model(rgb, freq)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels, all_probs = map(np.array, (all_preds, all_labels, all_probs))

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # only one class present in tiny test sets

    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"], zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=150)
        plt.close()

    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
