# Adaptive Benchmarking for Ordinal / Continuous LLM Evaluation

**Extending [Fluid Benchmarking](https://github.com/allenai/fluid-benchmarking) beyond binary outcomes**

[Python 3.10+](https://www.python.org/downloads/)
[License](LICENSE.md)
[Upstream](https://github.com/allenai/fluid-benchmarking)

**Repository:** [https://github.com/aidenzhou8/Ordinal-FBM](https://github.com/aidenzhou8/Ordinal-FBM)

---

## Summary

[Fluid Benchmarking](https://arxiv.org/abs/2509.11106) uses **Item Response Theory (IRT)** and **Computerized Adaptive Testing (CAT)** to adaptively evaluate language models, achieving reduced costs and enhanced evaluation quality. The original release is constrained to **binary** responses and **2PL** IRT.

This project **extends** that stack so adaptive evaluation works for a **much larger set of benchmarks** that use **ordinal** or **continuous** scores: for instance, graded safety tasks, Likert scale questions, or metrics in $[0,1]$.


| Feature             | Models                                                                                                                                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ordinal IRT         | **GRM** (Graded Response), **GPCM** (Generalized Partial Credit)                                                                                                                                                                                                                                                           |
| Continuous IRT      | **continuous**: $[0,1]$ scores as noisy observations around a predicted mean; the spread of noise is **fixed**. **continuous_cat**: same setup, but **noise is increased when the prediction is near 0.5 and decreased near 0 or 1**, adhering to the formulation of [Balkir et al.](https://arxiv.org/html/2601.13885v1). |
| Same evaluation API | `fluid_benchmarking()`, `full_ability()`, `iterate_evals()`— unchanged API, IRT type is inferred from the model object                                                                                                                                                                                                     |


---

## Installation

```bash
git clone https://github.com/aidenzhou8/fluid-benchmarking-ordinal.git
cd fluid-benchmarking-ordinal
```

**Base** (matches upstream Fluid Benchmarking):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

**With 2PL fitting (py-irt)**:

```bash
pip install -e ".[irt]"
```

**With ordinal / continuous Pyro models** (GRM, GPCM, continuous, continuous_cat):

```bash
pip install -e ".[ordinal]"
```

---

## Quick start

Load a 2PL model from the [Hugging Face dataset](https://huggingface.co/datasets/allenai/fluid-benchmarking), then run adaptive benchmarking:

```python
from fluid_benchmarking import evaluation

abilities_fb, items_fb = evaluation.fluid_benchmarking(
    lm_responses=lm_responses,
    irt_model=irt_model,  # (n_items, 2): columns [a, b]
    start_ability=0.0,
    n_max=100,
    estimation_method="map",
)
```

Same API for **ordinal or continuous** IRT: pass a **dict** `irt_model` (`model_type` plus item parameters). To start,

```python
import numpy as np
from fluid_benchmarking import evaluation

irt_model = {
    "model_type": "continuous_cat",
    "a": np.array([1.0, 1.0, 1.0]),
    "diff": np.array([0.0, 0.5, -0.2]),
}
lm_responses = np.array([0.7, 0.55, 0.9])  # [0, 1] scores, one per item

abilities_fb, items_fb = evaluation.fluid_benchmarking(
    lm_responses=lm_responses,
    irt_model=irt_model,
    start_ability=0.0,
    n_max=100,
    estimation_method="map",
)
```

After you fit item parameters (see **Fitting IRT models**), point the loaders at your data root:

1. Set the environment variable `FLUID_BENCHMARKING_DATA_DIR` to the directory that contains `irt_models/` and `lm_eval_results/`.
2. Call `datasets.load_ordinal_irt_model(benchmark_name, model_type)` to read `{benchmark}_{model_type}_items.csv` and `_metadata.json`, then `datasets.align_ordinal_irt_to_items(irt_model, item_ids)` so rows match your LM evaluation CSV.
3. Load LM scores with `datasets.load_lm_eval_results(..., binary=False)`; for GRM/GPCM, pass `score_values` so raw scores map to category indices. Continuous modes use raw scores in [0,1].

The `irt_model` dict always includes `model_type` and `a`; ordinal models add `thresholds` or `steps`, naive continuous adds `diff` and `sigma`, and **continuous_cat** adds `diff` only (heteroskedastic variance is implied).

For benchmarks across LMs and checkpoints, use `scripts/run_experiments.py` with `--irt-model-type` set to `2pl`, `grm`, `gpcm`, `continuous`, or `continuous_cat`, and `--score-values` when using ordinal types that need explicit score levels.

---

## Fitting IRT models


| Script                         | Use                                                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `irt/fit_irt_model.py`         | Binary **2PL** (py-irt), upstream-compatible                                                          |
| `irt/fit_ordinal_irt_model.py` | **GRM**, **GPCM**, **continuous**, **continuous_cat** from JSONL (`subject_id`, `responses` per item) |


Example:

```bash
python irt/fit_ordinal_irt_model.py \
  --input_path responses.jsonl \
  --model_type continuous_cat \
  --output_dir data/irt_models
```

Outputs `{stem}_{model_type}_items.csv` and `_metadata.json` for use with `load_ordinal_irt_model`.

---

## Credits

- **Fluid Benchmarking** — Valentin Hofmann et al., *Second Conference on Language Modeling (COLM), 2025*. [Paper](https://arxiv.org/abs/2509.11106) · [Blog](https://allenai.org/blog/fluid-benchmarking) · [Dataset](https://huggingface.co/datasets/allenai/fluid-benchmarking).
- **Continuous heteroskedastic formulation** — [Balkir et al.](https://arxiv.org/html/2601.13885v1), *Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores* (arXiv:2601.13885).

This repository is an **independent extension**. It is not endorsed by AI2. Cite the original work when using Fluid Benchmarking methods.

```bibtex
@inproceedings{hofmann2025fluid,
  title={Fluid Language Model Benchmarking},
  author={Valentin Hofmann and David Heineman and Ian Magnusson and Kyle Lo and Jesse Dodge and Maarten Sap and Pang Wei Koh and Chun Wang and Hannaneh Hajishirzi and Noah A. Smith},
  booktitle={Second Conference on Language Modeling},
  year={2025}
}
```

---

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md).
