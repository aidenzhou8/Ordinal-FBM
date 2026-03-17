"""
Utilities for incorporating HarmBench (https://www.harmbench.org/) into Fluid Benchmarking.

HarmBench is an AI safety benchmark for automated red teaming. It evaluates whether LLMs
refuse harmful requests. For Fluid Benchmarking: refusal = 1 (correct), compliance = 0 (incorrect).

Data pipeline:
1. Run HarmBench evaluation on your LMs (see https://github.com/centerforaisafety/HarmBench)
2. Convert HarmBench results to Fluid format using prepare_harmbench_data.py
3. Fit IRT model from responses (many LMs × behaviors)
4. Set FLUID_BENCHMARKING_DATA_DIR and run experiments
"""

from typing import Any

import pandas as pd


def load_harmbench_behaviors() -> pd.DataFrame:
    """Load HarmBench behaviors from Hugging Face (AlignmentResearch/HarmBench)."""
    from datasets import load_dataset

    ds = load_dataset("AlignmentResearch/HarmBench", split="train")
    return ds.to_pandas()


def behavior_id_to_fluid_id(behavior_id: str, index: int) -> str:
    """
    Map HarmBench BehaviorID to Fluid Benchmarking item ID format.
    Fluid expects item IDs like harmbench_0, harmbench_1 for id2benchmark to work.
    """
    return f"harmbench_{index}"


def create_fluid_item_ids(behaviors: pd.DataFrame) -> dict[str, str]:
    """Create mapping from HarmBench behavior IDs to fluid format (harmbench_0, ...)."""
    if "BehaviorID" in behaviors.columns:
        ids = behaviors["BehaviorID"].tolist()
    else:
        ids = [str(i) for i in range(len(behaviors))]
    return {bid: f"harmbench_{i}" for i, bid in enumerate(ids)}
