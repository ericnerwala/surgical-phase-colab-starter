# Surgical Phase Recognition Research Notes (2023–2026)

This short review focuses on practical ideas that can be implemented in a Colab-friendly pipeline.

## Key papers and actionable takeaways

1. **LoViT: Long Video Transformer for Surgical Phase Recognition** (2023)  
   - Link: https://arxiv.org/abs/2305.08989  
   - Actionable: long-context temporal modeling helps phase boundaries; for medium compute, use shorter windows + overlap + positional encodings.

2. **Metrics Matter in Surgical Phase Recognition** (2023)  
   - Link: https://arxiv.org/abs/2305.13961  
   - Actionable: report more than accuracy; always include macro-F1, per-phase F1, and confusion matrix due to class imbalance and boundary errors.

3. **Friends Across Time: Multi-Scale Action Segmentation Transformer for Surgical Phase Recognition** (2024)  
   - Link: https://arxiv.org/abs/2401.11644  
   - Actionable: multi-scale temporal features improve robustness; in practice, evaluate multiple sequence lengths (16/32/48) and/or multi-dilation TCN.

4. **Robust Surgical Phase Recognition From Annotation Efficient Supervision** (2024)  
   - Link: https://arxiv.org/abs/2406.18481  
   - Actionable: label efficiency matters; support low-label splits + self-supervised pretraining to improve sample efficiency.

5. **ViTALS: Vision Transformer for Action Localization in Surgical Nephrectomy** (2024)  
   - Link: https://arxiv.org/abs/2405.02571  
   - Actionable: temporal design (dilated and hierarchical processing) is often more important than heavier image backbones for workflow tasks.

6. **SurgX: Neuron-Concept Association for Explainable Surgical Phase Recognition** (2025)  
   - Link: https://arxiv.org/abs/2507.15418  
   - Actionable: add explainability diagnostics after baseline performance (saliency/concept probes) to support clinical trust.

7. **CurConMix+: Unified Spatio-Temporal Framework for Hierarchical Surgical Workflow Understanding** (2026)  
   - Link: https://arxiv.org/abs/2601.12312  
   - Actionable: jointly modeling phase/action/tool hierarchy is promising; next step after stable phase-only baselines.

## Recommended practical roadmap (for this repo)

- **Step 1 (done in code):** robust baselines across model families (CNN+LSTM, TCN, Transformer).
- **Step 2:** run fair comparison with fixed data splits and seeds.
- **Step 3:** pretrain encoder (SimCLR) on unlabeled/weakly labeled frames, then finetune temporal heads.
- **Step 4:** run low-label experiments (10%, 25%, 50% labels) and compare macro-F1 gains.
- **Step 5:** add boundary-aware loss/smoothing and calibration diagnostics.

## Notes

- arXiv links are included for quick access/reproducibility.
- For publication-grade review, confirm final venue versions (MICCAI/MedIA/TMI) and exact dataset protocols.
