#!/usr/bin/env python3
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from surgical_phase.data.dataset import FrameDataset, SequenceDataset
from surgical_phase.engine.evaluate import evaluate, save_metrics
from scripts.train import make_model
from surgical_phase.utils.io import load_yaml


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default="outputs/metrics_test.json")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    df = pd.read_csv(cfg["data"]["manifest_csv"])
    sub = df[df.split == args.split].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows in split={args.split}")
    num_classes = int(df.phase.max() + 1)

    if cfg["model"]["name"] == "frame_cnn":
        ds = FrameDataset(sub, image_size=cfg["data"].get("image_size", 224), augment=False)
    else:
        ds = SequenceDataset(
            sub,
            seq_len=cfg["data"].get("seq_len", 32),
            image_size=cfg["data"].get("image_size", 224),
            stride=cfg["data"].get("eval_stride", 16),
            augment=False,
        )
    loader = DataLoader(ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(cfg, num_classes).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    m = evaluate(model, loader, device=device, num_classes=num_classes)
    save_metrics(args.out, m)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
