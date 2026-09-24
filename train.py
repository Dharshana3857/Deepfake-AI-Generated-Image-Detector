"""
Train the dual-stream deepfake detector.

Usage:
    python src/train.py --config config.yaml
    python src/train.py --epochs 10 --batch_size 16     # override individual settings
"""
import os
import sys
import argparse
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import DeepfakeDataset, list_samples, split_samples
from models import build_model
from utils import load_config, set_seed, get_device, save_checkpoint, EarlyStopping


def run_epoch(model, loader, criterion, optimizer, device, scaler, train=True):
    model.train(mode=train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for rgb, freq, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            rgb, freq, labels = rgb.to(device), freq.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            if scaler is not None and train:
                with torch.cuda.amp.autocast():
                    outputs = model(rgb, freq)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(rgb, freq)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_n += labels.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs: cfg["train"]["epochs"] = args.epochs
    if args.batch_size: cfg["train"]["batch_size"] = args.batch_size
    if args.lr: cfg["train"]["lr"] = args.lr
    if args.data_root: cfg["data"]["root_dir"] = args.data_root

    set_seed(cfg["data"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    samples = list_samples(cfg["data"]["root_dir"])
    if len(samples) < 10:
        print(f"ERROR: only found {len(samples)} images under '{cfg['data']['root_dir']}'.")
        print("Run 'python generate_sample_data.py' first, or add your own images under")
        print("data/raw/real/ and data/raw/fake/. See README for real-dataset options.")
        sys.exit(1)

    train_s, val_s, test_s = split_samples(
        samples, cfg["data"]["val_split"], cfg["data"]["test_split"]
    )
    print(f"Samples -> train: {len(train_s)}  val: {len(val_s)}  test: {len(test_s)}")

    image_size = cfg["data"]["image_size"]
    train_ds = DeepfakeDataset(train_s, image_size, train=True)
    val_ds = DeepfakeDataset(val_s, image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                               shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                             shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg["train"]["scheduler_patience"]
    )
    scaler = torch.cuda.amp.GradScaler() if (cfg["train"]["mixed_precision"] and device.type == "cuda") else None
    early_stop = EarlyStopping(patience=cfg["train"]["early_stopping_patience"], mode="max")

    os.makedirs(os.path.dirname(cfg["train"]["log_file"]), exist_ok=True)
    log_rows = []

    best_acc = 0.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)
        scheduler.step(val_acc)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.4f} | {dt:.1f}s")

        log_rows.append([epoch, train_loss, train_acc, val_loss, val_acc, dt])

        improved = early_stop.step(val_acc)
        if improved:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_acc,
                             os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pt"))
            print(f"  -> new best model saved (val_acc={best_acc:.4f})")

        if early_stop.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    with open(cfg["train"]["log_file"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
        writer.writerows(log_rows)

    print(f"Training complete. Best val_acc={best_acc:.4f}. "
          f"Checkpoint: {cfg['train']['checkpoint_dir']}/best_model.pt")


if __name__ == "__main__":
    main()
