from __future__ import annotations

import json

import torch
from tqdm import tqdm

from .metrics import compute_metrics


def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y, *_ in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return compute_metrics(y_true, y_pred, num_classes)


def save_metrics(path: str, metrics: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
