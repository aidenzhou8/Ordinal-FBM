# Adaptive Benchmarking for Ordinal and Continuous Scores

This project extends [Fluid Benchmarking](https://github.com/allenai/fluid-benchmarking) (Hofmann et al., COLM 2025) so adaptive testing is not limited to **binary** (correct/incorrect) outcomes.

## What was added

| Capability | Description |
|------------|-------------|
| **Ordinal IRT** | **GRM** (Graded Response Model) and **GPCM** (Generalized Partial Credit Model) for multi-level scores (e.g. Likert-style or binned metrics). |
| **Continuous scores** | Two continuous formulations: a baseline Gaussian-on-sigmoid model (`continuous`) and a **heteroskedastic** model aligned with [continuous-cat](https://github.com/trismik/continuous-cat) (`continuous_cat`): variance \(\sigma^2 \propto \mu(1-\mu)\). |
| **Same API** | `evaluation.fluid_benchmarking()`, `full_ability()`, and related helpers branch on IRT type; MFI item selection uses the appropriate Fisher information. |
| **Fitting** | `irt/fit_ordinal_irt_model.py` fits ordinal/continuous models from JSONL responses; `irt/ordinal_models.py` implements Pyro SVI models. |

## Data flow (high level)

1. **LM evaluation** produces scores per item (binary, ordinal levels, or \([0,1]\) continuous).
2. **IRT parameters** are loaded from CSV + metadata (ordinal/continuous) or Hugging Face (2PL).
3. **Adaptive loop** selects items by maximum Fisher information and updates ability estimates.

See [ARCHITECTURE.md](ARCHITECTURE.md) for module-level detail.

## Relationship to upstream

- **Upstream**: [allenai/fluid-benchmarking](https://github.com/allenai/fluid-benchmarking) — binary 2PL, Hugging Face data, paper experiments.
- **This fork/extension**: ordinal/continuous paths in `fluid_benchmarking/`, Pyro models in `irt/`, scripts and HarmBench notes preserved.

When citing work, cite the original Fluid Benchmarking paper; you may additionally describe this repository as an extension for ordinal and continuous IRT.
