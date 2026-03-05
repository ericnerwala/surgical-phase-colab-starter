#!/usr/bin/env python3
import sys
from pathlib import Path

# Allow running scripts without requiring `pip install -e .` first
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from surgical_phase.data.dataset import FrameDataset, SequenceDataset
from surgical_phase.engine.trainer import train_loop
from surgical_phase.models.backbones import FrameClassifier
from surgical_phase.models.temporal import CNNLSTMPhaseModel, TemporalConvNetPhaseModel, TransformerPhaseModel
from surgical_phase.utils.io import load_yaml
from surgical_phase.utils.repro import seed_everything


def make_model(cfg, num_classes):
    name = cfg["model"]["name"]
    if name == "frame_cnn":
        return FrameClassifier(num_classes=num_classes, pretrained=cfg["model"].get("pretrained_cnn", True))
    if name == "cnn_lstm":
        return CNNLSTMPhaseModel(
            num_classes=num_classes,
            hidden_dim=cfg["model"].get("hidden_dim", 256),
            layers=cfg["model"].get("layers", 1),
            pretrained_cnn=cfg["model"].get("pretrained_cnn", True),
        )
    if name == "tcn":
        return TemporalConvNetPhaseModel(
            num_classes=num_classes,
            hidden_dim=cfg["model"].get("hidden_dim", 256),
            levels=cfg["model"].get("levels", 4),
            pretrained_cnn=cfg["model"].get("pretrained_cnn", True),
        )
    if name == "transformer":
        return TransformerPhaseModel(
            num_classes=num_classes,
            d_model=cfg["model"].get("d_model", 256),
            nhead=cfg["model"].get("nhead", 8),
            layers=cfg["model"].get("layers", 4),
            pretrained_cnn=cfg["model"].get("pretrained_cnn", True),
        )
    raise ValueError(f"Unknown model {name}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(cfg["data"]["manifest_csv"])
    num_classes = int(df.phase.max() + 1)

    train_df = df[df.split == "train"].copy()
    val_df = df[df.split == "val"].copy()
    if len(val_df) == 0:
        val_df = train_df.sample(frac=cfg["data"].get("val_frac", 0.2), random_state=cfg.get("seed", 42))
        train_df = train_df.drop(val_df.index)

    if cfg["model"]["name"] == "frame_cnn":
        train_ds = FrameDataset(train_df, image_size=cfg["data"].get("image_size", 224), augment=True)
        val_ds = FrameDataset(val_df, image_size=cfg["data"].get("image_size", 224), augment=False)
    else:
        train_ds = SequenceDataset(
            train_df,
            seq_len=cfg["data"].get("seq_len", 32),
            image_size=cfg["data"].get("image_size", 224),
            stride=cfg["data"].get("train_stride", 8),
            augment=True,
        )
        val_ds = SequenceDataset(
            val_df,
            seq_len=cfg["data"].get("seq_len", 32),
            image_size=cfg["data"].get("image_size", 224),
            stride=cfg["data"].get("eval_stride", 16),
            augment=False,
        )

    num_workers = cfg["train"].get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=False, num_workers=num_workers)

    # Diagnostics: split/phase coverage
    print({"rows_total": len(df), "rows_train": len(train_df), "rows_val": len(val_df), "num_classes": num_classes})
    train_phases = set(train_df.phase.unique().tolist())
    val_phases = set(val_df.phase.unique().tolist())
    missing_in_train = sorted(list(val_phases - train_phases))
    if missing_in_train:
        print("WARNING: phases present in val but missing in train:", missing_in_train)

    model = make_model(cfg, num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"].get("lr", 1e-4), weight_decay=cfg["train"].get("wd", 1e-4))

    # Optional class-weighted CE + label smoothing for imbalance robustness
    use_class_weights = cfg["train"].get("use_class_weights", True)
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    if use_class_weights:
        counts = np.bincount(train_df.phase.to_numpy(), minlength=num_classes).astype(float)
        counts[counts == 0] = 1.0
        weights = 1.0 / counts
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print("class_counts:", counts.tolist())
        print("class_weights:", [round(float(w), 4) for w in class_weights.detach().cpu()])
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    out_dir = Path(cfg["train"].get("output_dir", "outputs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    full_cfg = {
        **cfg,
        "output_dir": str(out_dir),
        "epochs": cfg["train"].get("epochs", 5),
    }
    train_loop(model, train_loader, val_loader, optim, criterion, full_cfg, device, num_classes)


if __name__ == "__main__":
    main()
