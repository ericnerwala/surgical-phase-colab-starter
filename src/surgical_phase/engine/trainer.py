from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import compute_metrics


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    losses, y_true, y_pred = [], [], []
    for batch in tqdm(loader, leave=False):
        x, y = batch[0].to(device), batch[1].to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return sum(losses) / max(1, len(losses)), y_true, y_pred


def train_loop(model, train_loader, val_loader, optimizer, criterion, cfg, device, num_classes):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1
    history = []
    for epoch in range(cfg["epochs"]):
        tr_loss, tr_y, tr_p = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_y, va_p = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        tr_m = compute_metrics(tr_y, tr_p, num_classes)
        va_m = compute_metrics(va_y, va_p, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_accuracy": tr_m["accuracy"],
            "val_accuracy": va_m["accuracy"],
            "train_macro_f1": tr_m["macro_f1"],
            "val_macro_f1": va_m["macro_f1"],
        }
        history.append(row)
        print(row)

        if va_m["macro_f1"] > best_val:
            best_val = va_m["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": va_m}, out_dir / "best.pt")
            (out_dir / "best_val_metrics.json").write_text(json.dumps(va_m, indent=2))

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
