#!/usr/bin/env python3
"""
Prepare HarmBench data for Fluid Benchmarking.

This script:
1. Downloads HarmBench behaviors from Hugging Face
2. Creates IRT model template (placeholder a, b - run fit_irt_model after collecting responses)
3. Creates directory structure for LM eval results
4. Outputs instructions for converting HarmBench evaluation output

Usage:
    python scripts/prepare_harmbench_data.py --out-dir data/harmbench

Then:
    - Run HarmBench evaluation on your LMs (https://github.com/centerforaisafety/HarmBench)
    - Convert HarmBench "attack success" to binary: refusal=1, compliance=0
    - Save as CSV: rows=harmbench_0, harmbench_1, ...; columns=checkpoints
    - Fit IRT: python irt/fit_irt_model.py --input_path harmbench_responses.jsonl
    - Set FLUID_BENCHMARKING_DATA_DIR=data/harmbench and run experiments

Ordinal: fit with fit_ordinal_irt_model.py (grm/gpcm/continuous), run with --score-values and --irt-model-type.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/harmbench"),
        help="Output directory for HarmBench data",
    )
    parser.add_argument(
        "--placeholder-irt",
        action="store_true",
        help="Create placeholder IRT model (a=1, b=0) for testing without fitting",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "irt_models").mkdir(exist_ok=True)
    (out_dir / "lm_eval_results" / "harmbench").mkdir(parents=True, exist_ok=True)

    # Load HarmBench behaviors
    try:
        from datasets import load_dataset
        ds = load_dataset("AlignmentResearch/HarmBench", split="train")
        behaviors = ds.to_pandas()
    except Exception as e:
        print(f"Could not load HarmBench from HuggingFace: {e}")
        print("Creating placeholder with 400 items (HarmBench standard size)")
        behaviors = pd.DataFrame({"BehaviorID": [f"harmbench_{i}" for i in range(400)]})

    n_items = len(behaviors)
    item_ids = [f"harmbench_{i}" for i in range(n_items)]

    # IRT model: placeholder or template
    irt_path = out_dir / "irt_models" / "harmbench.csv"
    if args.placeholder_irt:
        irt_model = pd.DataFrame(
            {"a": np.ones(n_items), "b": np.zeros(n_items)},
            index=item_ids,
        )
        irt_model.to_csv(irt_path)
        print(f"Wrote placeholder IRT model to {irt_path}")
    else:
        # Template for py-irt input format
        template_path = out_dir / "harmbench_irt_template.jsonl"
        with open(template_path, "w") as f:
            f.write('{"subject_id": "lm_1", "responses": {"harmbench_0": 1, "harmbench_1": 0}}\n')
        print(f"IRT template (py-irt format) at {template_path}")
        print("  Add one line per (model, checkpoint) with responses for each harmbench_N")
        print("  Then: python irt/fit_irt_model.py --input_path harmbench_responses.jsonl")
        print("  Copy output to {out_dir}/irt_models/harmbench.csv")

    # LM eval CSV template
    template_csv = out_dir / "lm_eval_results" / "harmbench" / "template.csv"
    template_df = pd.DataFrame(
        {c: np.nan for c in ["checkpoint_1", "checkpoint_2"]},
        index=item_ids,
    )
    template_df.to_csv(template_csv)
    print(f"\nLM eval template at {template_csv}")
    print("  Replace with your data: rows=harmbench_0..harmbench_N, columns=checkpoints")
    print("  Values: 1=refusal (good), 0=compliance (bad)")

    # Instructions
    readme = out_dir / "README.md"
    readme.write_text("""# HarmBench data for Fluid Benchmarking

## Directory structure
- irt_models/harmbench.csv - IRT parameters (a, b) per item
- lm_eval_results/harmbench/{lm}.csv - LM responses per model

## Data format

### LM eval CSV
- Rows: harmbench_0, harmbench_1, ... harmbench_N (item IDs)
- Columns: checkpoint names (e.g. stage1-step1000)
- Values: 1 = model refused (correct), 0 = model complied (incorrect)

### IRT model CSV
- Index: harmbench_0, harmbench_1, ...
- Columns: a (discrimination), b (difficulty)

## Pipeline
1. Run HarmBench evaluation: https://github.com/centerforaisafety/HarmBench
2. Convert attack success → refusal (1 - ASR) per behavior
3. Save as lm_eval_results/harmbench/{lm}.csv
4. Fit IRT from combined responses (see irt/fit_irt_model.py)
5. export FLUID_BENCHMARKING_DATA_DIR=$(pwd)
6. python scripts/run_experiments.py --benchmarks harmbench --lms your_lm
""")
    print(f"\nInstructions at {readme}")
    print(f"\nTo use: export FLUID_BENCHMARKING_DATA_DIR={out_dir.absolute()}")
    print("Then: python scripts/run_experiments.py --benchmarks harmbench --lms <your_lms>")


if __name__ == "__main__":
    main()
