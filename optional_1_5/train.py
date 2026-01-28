import yaml
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.pytorch

from dataset import TCGAMultimodalDataset
from model import LateFusionMLP


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(y.numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )


def main(config_path):
    cfg = load_config(config_path)

    # --- MLflow ---
    mlflow.set_experiment(cfg["experiment"]["name"])

    with mlflow.start_run():
        # Log config
        mlflow.log_params({
            "lr": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["batch_size"],
            "epochs": cfg["training"]["epochs"],
            "dropout": cfg["model"]["dropout"],
            "hidden_dims": str(cfg["model"]["hidden_dims"]),
        })

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Dataset ---
        train_ds = TCGAMultimodalDataset(
            split="train",
            cfg=cfg
        )
        val_ds = TCGAMultimodalDataset(
            split="val",
            cfg=cfg
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
        )

        # --- Model ---
        model = LateFusionMLP(
            input_dim=train_ds.input_dim,
            hidden_dims=cfg["model"]["hidden_dims"],
            dropout=cfg["model"]["dropout"],
            num_classes=train_ds.num_classes,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()

        # --- Training loop ---
        for epoch in range(cfg["training"]["epochs"]):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device
            )

            preds, targets = eval_epoch(model, val_loader, device)

            macro_f1 = f1_score(targets, preds, average="macro")
            micro_f1 = f1_score(targets, preds, average="micro")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
            }, step=epoch)

            print(
                f"[Epoch {epoch}] "
                f"Loss={train_loss:.4f} "
                f"Macro-F1={macro_f1:.4f} "
                f"Micro-F1={micro_f1:.4f}"
            )

        # --- Confusion Matrix ---
        cm = confusion_matrix(targets, preds)
        np.save("confusion_matrix.npy", cm)
        mlflow.log_artifact("confusion_matrix.npy")

        # --- Save best model ---
        Path("artifacts").mkdir(exist_ok=True)
        model_path = "artifacts/best_model.pt"
        torch.save(model.state_dict(), model_path)

        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="modeling/config.yaml",
    )
    args = parser.parse_args()

    main(args.config)