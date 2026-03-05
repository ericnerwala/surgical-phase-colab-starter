# Surgical Phase Recognition — Colab Experimentation Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericnerwala/surgical-phase-colab-starter/blob/main/notebooks/02_training_pipeline_colab.ipynb)

A modular, reproducible experimentation repo for **surgical phase classification** with practical Colab constraints.

## What was in the starter (inspected assumptions)

Original repo contained:
- `notebooks/01_dataset_setup_and_eda.ipynb` for CholecT50 challenge validation download + EDA.

Data assumptions discovered from challenge labels:
- Labels are JSON per video (`labels/VIDxx.json`).
- `annotations` are frame-keyed dicts.
- Phase id can be parsed from annotation entries (last element in annotation vector).
- Frames are in `videos/VIDxx/*.png` with occasional duplicate-style naming (`000123 (1).png`).

## What this upgraded pipeline now includes

### Models (modular)
- **CNN + LSTM baseline** (`cnn_lstm`)
- **Temporal ConvNet (1D dilated conv)** (`tcn`)
- **Transformer temporal encoder** (`transformer`)
- Optional frame-level baseline (`frame_cnn`)

### Training/evaluation features
- Train / val / test split support via manifest
- Metrics:
  - accuracy
  - macro-F1
  - per-phase F1
  - confusion matrix
- Seed control (`seed_everything`)
- Checkpointing (`best.pt`)
- JSON logs (`history.json`, `best_val_metrics.json`, eval outputs)

### Experiment system
- YAML config-based runs in `configs/`
- Reproducible scripts in `scripts/`
- Colab training notebook: `notebooks/02_training_pipeline_colab.ipynb`
- Hyperparameter search scaffold: `scripts/search.py`
- Optional self-supervised pretraining scaffold (SimCLR): `scripts/pretrain_simclr.py`

## Repository structure

- `src/surgical_phase/data/`
  - `manifest.py`: build manifest from challenge labels
  - `dataset.py`: frame and sequence datasets
- `src/surgical_phase/models/`
  - `backbones.py`, `temporal.py`, `ssl.py`
- `src/surgical_phase/engine/`
  - `trainer.py`, `evaluate.py`, `metrics.py`
- `scripts/`
  - `build_manifest.py`, `train.py`, `eval.py`, `search.py`, `pretrain_simclr.py`
- `configs/`
  - `base.yaml`, `tcn.yaml`, `transformer.yaml`, `frame_cnn.yaml`
- `docs_research_2023_2026.md`

## Quickstart (local or Colab)

```bash
git clone https://github.com/ericnerwala/surgical-phase-colab-starter.git
cd surgical-phase-colab-starter
pip install -r requirements.txt
pip install -e .
```

### 1) Build manifest from challenge validation set

```bash
python scripts/build_manifest.py \
  --data-root /content/data/cholect50-challenge-val \
  --out-csv data/processed/manifest.csv \
  --phase-map-out data/processed/phase_map.json \
  --val-videos VID73 \
  --test-videos VID75
```

### 2) Train models

```bash
# CNN+LSTM baseline
python scripts/train.py --config configs/base.yaml

# TCN
python scripts/train.py --config configs/tcn.yaml

# Transformer
python scripts/train.py --config configs/transformer.yaml
```

### 3) Evaluate on test split

```bash
python scripts/eval.py --config configs/base.yaml --checkpoint outputs/cnn_lstm_base/best.pt --split test --out outputs/cnn_lstm_base/test_metrics.json
python scripts/eval.py --config configs/tcn.yaml --checkpoint outputs/tcn_base/best.pt --split test --out outputs/tcn_base/test_metrics.json
python scripts/eval.py --config configs/transformer.yaml --checkpoint outputs/transformer_base/best.pt --split test --out outputs/transformer_base/test_metrics.json
```

### 4) (Optional) SSL pretraining scaffold

```bash
python scripts/pretrain_simclr.py --config configs/base.yaml --out outputs/simclr_pretrain.pt
```

### 5) Hyperparameter search scaffold

```bash
python scripts/search.py --base-config configs/base.yaml --max-runs 6 --dry-run
```

## Experiment matrix (recommended first run)

| Experiment | Config | Seq len | Epochs | Goal |
|---|---|---:|---:|---|
| Baseline temporal | `configs/base.yaml` | 32 | 8 | reference CNN+LSTM |
| Temporal conv | `configs/tcn.yaml` | 32 | 8 | stronger local temporal context |
| Transformer temporal | `configs/transformer.yaml` | 32 | 8 | long-range temporal reasoning |
| SSL + finetune (optional) | `base + pretrain_simclr.py` | N/A | 5 + 8 | low-label robustness |

## Notes for Colab medium compute

- Start with batch size 8, seq_len 32.
- If OOM: lower `seq_len` to 16 or batch size to 4.
- Prefer frozen or lightly finetuned CNN encoder for faster experiments.
- Use overlapping windows for better boundary performance without large models.

## Research summary

See: `docs_research_2023_2026.md` for concise 2023–2026 paper notes and actionable directions.
