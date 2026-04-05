# Architecture

## Package layout

| Path | Role |
|------|------|
| `fluid_benchmarking/config.py` | `IRT_MODEL_TYPE`, `SCORE_VALUES`, `DATA_DIR` (local benchmarks). |
| `fluid_benchmarking/datasets.py` | `load_irt_model` (2PL), `load_ordinal_irt_model`, `load_lm_eval_results` (binary vs ordinal mapping), `align_ordinal_irt_to_items`. |
| `fluid_benchmarking/irt_utils.py` | Fisher information: 2PL, GRM, GPCM, continuous, `continuous_cat`. |
| `fluid_benchmarking/estimators.py` | `ability_estimate` (2PL), `ability_estimate_ordinal` (GRM/GPCM/continuous). |
| `fluid_benchmarking/engine.py` | `run_fluid_benchmarking`, `_is_ordinal_irt`, MFI selection. |
| `fluid_benchmarking/evaluation.py` | `fluid_benchmarking`, `full_ability`, `iterate_evals`. |
| `irt/ordinal_models.py` | Pyro: GRM, GPCM, `ContinuousSafetyModel`, `ContinuousCategoricalModel`. |
| `irt/fit_ordinal_irt_model.py` | CLI to fit ordinal/continuous IRT from JSONL. |
| `scripts/run_experiments.py` | `--irt-model-type 2pl|grm|gpcm|continuous|continuous_cat`. |
| `scripts/verify_pipeline.py` | Smoke tests for the extended pipeline. |

## IRT model representation

- **2PL**: NumPy array shape `(n_items, 2)` with columns `[a, b]`.
- **Ordinal / continuous**: dict with `model_type`, `a`, and type-specific fields (`thresholds`, `steps`, `diff`, `sigma` for naive continuous).

Detection: `isinstance(irt_model, dict) and "model_type" in irt_model`.

## Verification

```bash
python scripts/verify_pipeline.py
```
