# SangoProject

Extremely low-resource English–Sango machine translation experiments comparing three
parameter-efficient fine-tuning (PEFT) methods — LoRA, DoRA, and OFT — applied to
facebook/nllb-200-distilled-600M.

**Paper:** "Comparing LoRA, DoRA, and OFT for Extremely Low-Resource English–Sango
Machine Translation: Adaptation Performance and Out-of-Domain Retention"
Author: Rakhun Kim (Hongik University)

This is a follow-up study to an earlier English–Sango MT paper targeting JKSCI
(separate repository/data).

## Repository structure

- `00_environment/` — environment manifests (Python, PyTorch, Transformers, PEFT,
  SacreBLEU versions; see `Appendix_Environment_Table` in the paper)
- `01_data_build/` — data pipeline scripts (LASER filtering, dedup, Direct-9K sampling,
  adaptation-dev/test split construction)
- `02_config/` — CONFIG LOCK v2: fixed hyperparameters (Adafactor lr=1e-4, warmup
  ratio 0.1, FP32, max_steps=3375) applied identically across all 9 runs
- `03_runs/` — LoRA / DoRA / OFT × seeds {42,43,44}, each with `best_adapter/`,
  `final_adapter/`, and `history_adapters/step_*` checkpoints (tracked via Git LFS)
- `04_evaluation/` — checkpoint evaluation scripts (chrF/BLEU/TER via sacrebleu,
  char_order=6, beta=2)
- `05_analysis/` — bootstrap significance testing (10,000 resamples, Holm correction),
  sensitivity analyses (best-vs-final, A-only, dual-reference)
- `06_figures/` — Figure 2 (forest plot), Figure 3 (trajectories), Supplementary S1/S2
- `07_data_package/` — minimal reproducibility package referenced in the paper's Data
  Availability Statement (aggregated CSVs + `analysis.py` + README; sentence-level
  raw hypotheses/references are intentionally excluded — see Data Availability
  Statement in the paper)

## Reproducing results

1. Set up the environment per `00_environment/` (Python 3.12, PyTorch 2.11.0+cu128,
   Transformers 5.15.0, Datasets 5.0.1, PEFT 0.20.0, SacreBLEU 2.6.0).
2. Rebuild the data splits with scripts in `01_data_build/` (note: `allenai/nllb` is a
   script-based HF dataset loader and requires `datasets==3.6.0` specifically for the
   build step — see script docstrings).
3. Run training with `02_config/` settings; all three PEFT methods use identical
   hyperparameters except method-specific ones (LoRA rank/alpha, DoRA, OFT block size).
4. Evaluate checkpoints with scripts in `04_evaluation/`.

## Data availability

Full sentence-level adaptation-test and FLORES+ hypothesis/reference data are not
included in this repository. An aggregated results package (bootstrap results,
per-checkpoint scores, efficiency metrics) is provided in `07_data_package/` and
described in the paper's Data Availability Statement.
