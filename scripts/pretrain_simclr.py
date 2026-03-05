#!/usr/bin/env python3
import sys
from pathlib import Path

# Allow running scripts without requiring `pip install -e .` first
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from torch.utils.data import DataLoader

from surgical_phase.data.dataset import FrameDataset
from surgical_phase.models.ssl import SimCLRFrameModel, nt_xent
from surgical_phase.utils.io import load_yaml
from surgical_phase.utils.repro import seed_everything
import pandas as pd


def aug_twice(x):
    noise1 = torch.randn_like(x) * 0.03
    noise2 = torch.randn_like(x) * 0.03
    return (x + noise1).clamp(-3, 3), (x + noise2).clamp(-3, 3)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/simclr_pretrain.pt")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))
    df = pd.read_csv(cfg["data"]["manifest_csv"])
    train_df = df[df.split == "train"]
    ds = FrameDataset(train_df, image_size=cfg["data"].get("image_size", 224), augment=True)
    loader = DataLoader(ds, batch_size=cfg["ssl"].get("batch_size", 64), shuffle=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimCLRFrameModel(pretrained=cfg["ssl"].get("pretrained_cnn", False), proj_dim=cfg["ssl"].get("proj_dim", 128)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["ssl"].get("lr", 3e-4), weight_decay=1e-6)

    model.train()
    for epoch in range(cfg["ssl"].get("epochs", 5)):
        running = 0.0
        for x, *_ in loader:
            x = x.to(device)
            x1, x2 = aug_twice(x)
            z1, z2 = model(x1), model(x2)
            loss = nt_xent(z1, z2, temperature=cfg["ssl"].get("temperature", 0.1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        print({"epoch": epoch, "loss": running / max(1, len(loader))})

    torch.save({"encoder": model.encoder.state_dict()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
