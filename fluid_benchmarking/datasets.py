import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from fluid_benchmarking import config


def _scores_to_categories(values: np.ndarray, score_values: List[float], atol: float = 1e-5) -> np.ndarray:
    """Map raw scores to category indices (0..K-1) by nearest match."""
    arr = np.asarray(score_values, dtype=float)
    flat = np.asarray(values, dtype=float).ravel()
    idx = np.argmin(np.abs(flat[:, None] - arr), axis=1)
    if not np.allclose(np.take(arr, idx), flat, atol=atol):
        raise ValueError("Scores do not match score_values.")
    return idx.reshape(values.shape).astype(np.int64)


def load_irt_model(
    repo_id: str,
    filename: str,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Load 2PL IRT model (a, b). Uses DATA_DIR if set and benchmark is provided."""
    if config.DATA_DIR is not None and benchmark is not None:
        local_path = config.DATA_DIR / "irt_models" / f"{benchmark}.csv"
        if local_path.exists():
            return pd.read_csv(local_path, index_col=0)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    return pd.read_csv(path, index_col=0)


def load_ordinal_irt_model(benchmark: str, model_type: str = "grm") -> dict:
    """Load ordinal/continuous IRT from {DATA_DIR}/irt_models/{benchmark}_{model_type}_items.csv + _metadata.json."""
    if config.DATA_DIR is None:
        raise FileNotFoundError("DATA_DIR must be set to load ordinal IRT models.")
    base = config.DATA_DIR / "irt_models" / f"{benchmark}_{model_type}"
    items_path = Path(f"{base}_items.csv")
    meta_path = Path(f"{base}_metadata.json")
    if not items_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Ordinal IRT files not found: {items_path} and {meta_path}")
    items_df = pd.read_csv(items_path)
    meta = json.loads(meta_path.read_text())
    item_ids = items_df["item_id"].tolist()
    a = items_df["a"].to_numpy(dtype=float)
    result = {
        "model_type": model_type,
        "a": a,
        "score_values": meta.get("score_values", [0.0, 0.25, 0.5, 0.75, 1.0] if model_type not in ("continuous", "continuous_cat") else [0.0, 1.0]),
        "item_ids": item_ids,
    }
    b_cols = sorted(
        [c for c in items_df.columns if c.startswith("b") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    step_cols = sorted(
        [c for c in items_df.columns if c.startswith("step") and c[4:].isdigit()],
        key=lambda x: int(x[4:]),
    )
    if model_type == "continuous":
        if "b" in items_df.columns:
            result["diff"] = items_df["b"].to_numpy(dtype=float)
        if "sigma" in items_df.columns:
            result["sigma"] = float(items_df["sigma"].iloc[0])
        elif "sigma" in meta:
            result["sigma"] = float(meta["sigma"])
    elif model_type == "continuous_cat":
        if "b" in items_df.columns:
            result["diff"] = items_df["b"].to_numpy(dtype=float)
        result["score_values"] = meta.get("score_values", [0.0, 1.0])
    elif b_cols:
        result["thresholds"] = items_df[b_cols].to_numpy(dtype=float)
    if step_cols:
        result["steps"] = items_df[step_cols].to_numpy(dtype=float)
    return result


def align_ordinal_irt_to_items(irt_model: dict, item_ids: list) -> dict:
    """Reorder IRT params to match item_ids."""
    id_to_idx = {x: i for i, x in enumerate(irt_model["item_ids"])}
    order = [id_to_idx[i] for i in item_ids]
    out = {
        "model_type": irt_model["model_type"],
        "a": irt_model["a"][order],
        "score_values": irt_model.get("score_values", [0.0, 1.0]),
        "item_ids": item_ids,
    }
    if "thresholds" in irt_model:
        out["thresholds"] = irt_model["thresholds"][order]
    if "steps" in irt_model:
        out["steps"] = irt_model["steps"][order]
    if "diff" in irt_model:
        out["diff"] = irt_model["diff"][order]
    if "sigma" in irt_model:
        out["sigma"] = irt_model["sigma"]
    return out


def load_lm_eval_results(
    repo_id: str,
    filename: str,
    binary: bool = True,
    benchmark: str | None = None,
    lm: str | None = None,
    score_values: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Load LM eval results. binary=True: >=0.5->1; score_values: map to categories for GRM/GPCM."""
    if config.DATA_DIR is not None and benchmark is not None and lm is not None:
        local_path = config.DATA_DIR / "lm_eval_results" / benchmark / f"{lm}.csv"
        if local_path.exists():
            eval_results = pd.read_csv(local_path, index_col=0)
            return _convert_eval_results(eval_results, binary, score_values)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    eval_results = pd.read_csv(path, index_col=0)
    return _convert_eval_results(eval_results, binary, score_values)


def _convert_eval_results(eval_results: pd.DataFrame, binary: bool, score_values: Optional[List[float]] = None) -> pd.DataFrame:
    if binary:
        return eval_results.ge(0.5).astype(int)
    if score_values is not None:
        return pd.DataFrame(
            _scores_to_categories(eval_results.values, score_values),
            index=eval_results.index,
            columns=eval_results.columns,
        )
    return eval_results


def load_open_llm_leaderboard_results() -> Any:
    path = hf_hub_download(
        repo_id="allenai/fluid-benchmarking",
        repo_type="dataset",
        filename=f"data/open_llm_leaderboard_results.json",
    )
    with open(path, "r") as f:
        return json.load(f)
    

def load_id_to_item_map() -> Any:
    path = hf_hub_download(
        repo_id="allenai/fluid-benchmarking",
        repo_type="dataset",
        filename=f"data/id_to_item_map.json",
    )
    with open(path, "r") as f:
        return json.load(f)
