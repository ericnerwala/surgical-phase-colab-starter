#!/usr/bin/env python3
"""Simple hyperparameter search scaffold (random/grid mix)."""
import itertools
import random
import subprocess

from surgical_phase.utils.io import load_yaml


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--max-runs", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.base_config)
    lrs = cfg.get("search", {}).get("lr", [1e-4, 3e-4, 1e-3])
    bss = cfg.get("search", {}).get("batch_size", [4, 8, 16])
    seqs = cfg.get("search", {}).get("seq_len", [16, 32])

    combos = list(itertools.product(lrs, bss, seqs))
    random.shuffle(combos)

    for i, (lr, bs, seq) in enumerate(combos[: args.max_runs]):
        cmd = [
            "python3",
            "scripts/train.py",
            "--config",
            args.base_config,
        ]
        print(f"run {i}: lr={lr} bs={bs} seq={seq}")
        print("export overrides in config copy before production runs")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
