<div align="center">

# Adaptive Benchmarking for Ordinal and Continuous LLM Evaluation

**Extending [Fluid Benchmarking](https://github.com/allenai/fluid-benchmarking) beyond binary outcomes**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![Upstream](https://img.shields.io/badge/upstream-allenai%2Ffluid--benchmarking-111)](https://github.com/allenai/fluid-benchmarking)

**Repository:** https://github.com/aidenzhou8/fluid-benchmarking-ordinal

</div>

---

## Summary

[Fluid Benchmarking](https://arxiv.org/abs/2509.11106) uses **Item Response Theory (IRT)** and **maximum Fisher information (MFI)** to adaptively evaluate language models with fewer items than a full benchmark. The original release assumes **binary** (correct/incorrect) responses and **2PL** IRT.

This project **extends** that stack so adaptive evaluation works for a **much larger set of benchmarks** that use **ordinal** or **continuous** scores—for example multi-level safety ratings, Likert-style judgments, or metrics in \([0,1]\) (e.g. ROUGE, normalized judge scores).

**What you get**

| Feature | Models |
|--------|--------|
| Ordinal IRT | **GRM** (Graded Response), **GPCM** (Generalized Partial Credit) |
| Continuous IRT | Homoskedastic Gaussian-on-sigmoid (`continuous`) and **heteroskedastic** continuous-cat style (`continuous_cat`) |
| Same evaluation API | `fluid_benchmarking()`, `full_ability()`, `iterate_evals()` — unchanged call pattern; IRT type is inferred from the model object |

See **[docs/PROJECT.md](docs/PROJECT.md)** for design detail and **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for file-level map.

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

Or everything: `pip install -e ".[irt,ordinal]"`.

---

## Quick start

### Binary 2PL (original behavior)

Unchanged from upstream: load 2PL parameters from the [Hugging Face dataset](https://huggingface.co/datasets/allenai/fluid-benchmarking) and run adaptive benchmarking.

```python
from fluid_benchmarking import evaluation

abilities_fb, items_fb = evaluation.fluid_benchmarking(
    lm_responses=lm_responses,
    irt_model=irt_model,  # (n_items, 2) array [a, b]
    start_ability=0.0,
    n_max=100,
    estimation_method="map",
)
```

### Ordinal or continuous IRT

1. Set `FLUID_BENCHMARKING_DATA_DIR` to a directory containing `irt_models/` and `lm_eval_results/`.
2. Fit or export ordinal/continuous item parameters (see **Fitting IRT** below).
3. Pass a **dict** IRT model (with `model_type`) instead of a 2PL array.

```bash
export FLUID_BENCHMARKING_DATA_DIR=/path/to/your/data
python scripts/run_experiments.py \
  --irt-model-type grm \
  --score-values 0,0.25,0.5,0.75,1 \
  --benchmarks harmbench \
  --lms your_lm
```

Supported `--irt-model-type` values: `2pl`, `grm`, `gpcm`, `continuous`, `continuous_cat`.

---

## Fitting IRT models

| Script | Use |
|--------|-----|
| `irt/fit_irt_model.py` | Binary **2PL** (py-irt), upstream-compatible |
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

## Verify the pipeline

```bash
python scripts/verify_pipeline.py
```

---

## HarmBench and local data

HarmBench-style workflows (local CSVs, optional placeholder IRT) are described in **`data/harmbench/README.md`** and **`scripts/prepare_harmbench_data.py`**.

---

## Reproducing upstream paper experiments

The **`scripts/run_experiments.py`** flow for standard benchmarks and Hugging Face data follows the original Fluid Benchmarking paper; use `--irt-model-type 2pl` for the published binary setup.

---

## Project structure (extension highlights)

```
fluid_benchmarking/   # Core: evaluation, engine, datasets, IRT utils
irt/
  ordinal_models.py     # GRM, GPCM, continuous, continuous_cat (Pyro)
  fit_ordinal_irt_model.py
scripts/
  run_experiments.py
  verify_pipeline.py
docs/
  PROJECT.md
  ARCHITECTURE.md
```

---

## Credits

- **Fluid Benchmarking** — Valentin Hofmann et al., *Second Conference on Language Modeling (COLM), 2025*. [Paper](https://arxiv.org/abs/2509.11106) · [Blog](https://allenai.org/blog/fluid-benchmarking) · [Dataset](https://huggingface.co/datasets/allenai/fluid-benchmarking).
- **Continuous heteroskedastic formulation** — aligned with ideas from [continuous-cat](https://github.com/trismik/continuous-cat) (Balkir et al., arXiv:2601.13885).

This repository is an **independent extension**; it is not endorsed by Ai2. Cite the original work when using Fluid Benchmarking methods.

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

Apache 2.0 — see [LICENSE.md](LICENSE.md) (same as upstream).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
