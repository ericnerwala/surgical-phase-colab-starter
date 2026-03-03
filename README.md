# Surgical Phase Recognition — Colab Starter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericnerwala/surgical-phase-colab-starter/blob/main/notebooks/01_dataset_setup_and_eda.ipynb)

Public starter repo for surgical AI experimentation on **CholecT50** with a Colab-first workflow.

## What this repo includes

- `notebooks/01_dataset_setup_and_eda.ipynb`
  - Colab environment setup
  - pulls CholecT50 helper repo
  - optional download for the public challenge validation set
  - basic dataset examination (file counts, label distribution, sample frames)

## Why this is useful

This gives you a clean, reproducible starting point to show lab instructors/professors:

- practical familiarity with surgical video datasets
- clean dataset handling practices
- fast baseline-readiness for modeling work

## Run in Google Colab

1. Open the notebook in Colab.
2. Runtime → Change runtime type → GPU.
3. Run cells top to bottom.

## Dataset notes

- CholecT50 main release requires request access via the official form (linked in notebook).
- The challenge validation set is publicly downloadable and useful for quick EDA pipeline tests.

## Quick local sanity check (completed)

Using the public challenge validation zip:

- clips: **5** (`VID68`, `VID70`, `VID73`, `VID74`, `VID75`)
- total extracted files: **1326**
- label files found: **5 JSON** + `label_mapping.txt`

## Next steps

- Add baseline model training notebook (`02_baseline_phase_model.ipynb`)
- Add experiment tracking (Weights & Biases)
- Add reproducibility checklist and mini-report template
