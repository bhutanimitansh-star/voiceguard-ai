"""
VoiceGuard AI - Training Script
==================================
Trains VoiceGuardNet (CNN + BiLSTM + Attention) on the cached, preprocessed
dataset produced by prepare_dataset.py.

Includes:
    - Stratified K-Fold cross-validation
    - Early stopping on validation loss
    - ReduceLROnPlateau learning-rate scheduling
    - Confusion matrix, ROC-AUC, Precision/Recall/F1 reporting
    - TensorBoard logging
    - Best-fold checkpoint export to models/checkpoints/

Usage:
    python train.py --manifest ../../dataset/processed/cached_manifest.csv
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(8)
torch.set_num_interop_threads(2)
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_fscore_support,
    classification_report,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import TRAIN_CONFIG, MODEL_DIR, DEVICE, CHECKPOINT_PATH
from models.architecture import build_model
from training.dataset import VoiceGuardDataset


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for mel, aux, labels in tqdm(loader, desc="train", leave=False):
        mel, aux, labels = mel.to(device), aux.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(mel, aux)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    for mel, aux, labels in tqdm(loader, desc="eval", leave=False):
        mel, aux, labels = mel.to(device), aux.to(device), labels.to(device)
        logits, _ = model(mel, aux)
        loss = criterion(logits, labels)
        total_loss += loss.item() * mel.size(0)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # P(AI Voice)

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


def compute_metrics(y_true, y_pred, y_prob):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # e.g. only one class present in a small batch
    cm = confusion_matrix(y_true, y_pred)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
    }


def run_training(manifest_path: str, config: dict):
    set_seed(config["seed"])

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # ------------------------------------------------------------
    # Load the three fixed dataset splits
    # ------------------------------------------------------------
    train_ds = VoiceGuardDataset(
        manifest_path,
        train=True,
        split="training"
    )

    val_ds = VoiceGuardDataset(
        manifest_path,
        train=False,
        split="validation"
    )

    test_ds = VoiceGuardDataset(
        manifest_path,
        train=False,
        split="testing"
    )
   

    print(f"Training samples:   {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Testing samples:    {len(test_ds)}")

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------
    train_loader = DataLoader(
    train_ds,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=0,
)
    val_loader = DataLoader(
    val_ds,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=0,
)

    test_loader = DataLoader(
    test_ds,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=0,
)


    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["lr_scheduler_factor"],
        patience=config["lr_scheduler_patience"],
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(config["num_epochs"]):

        print(f"\n===== Epoch {epoch + 1}/{config['num_epochs']} =====")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"F1={val_metrics['f1']:.4f} | "
            f"ROC-AUC={val_metrics['roc_auc']:.4f}"
        )

        # --------------------------------------------------------
        # Save best validation model
        # --------------------------------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

            patience_counter = 0

            print("✓ New best validation model")

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{config['early_stopping_patience']})"
            )

            if patience_counter >= config["early_stopping_patience"]:

                print("Early stopping triggered.")
                break

    # ------------------------------------------------------------
    # Load best validation model
    # ------------------------------------------------------------
    model.load_state_dict(best_state)

    # ------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------
    print("\n===== FINAL TEST EVALUATION =====")

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f"Test Loss:    {test_loss:.4f}")
    print(f"Precision:    {test_metrics['precision']:.4f}")
    print(f"Recall:       {test_metrics['recall']:.4f}")
    print(f"F1:           {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    # ------------------------------------------------------------
    # Save production checkpoint
    # ------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "test_metrics": test_metrics,
        },
        CHECKPOINT_PATH
    )

    print(
        f"\n✓ Best trained model saved to:\n"
        f"{CHECKPOINT_PATH}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VoiceGuard AI model")
    parser.add_argument("--manifest", type=str,
                         default="../../data/processed/cached_manifest.csv")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--k_folds", type=int, default=TRAIN_CONFIG["k_folds"])
    args = parser.parse_args()

    cfg = dict(TRAIN_CONFIG)
    cfg["num_epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["k_folds"] = args.k_folds

    run_training(args.manifest, cfg)
