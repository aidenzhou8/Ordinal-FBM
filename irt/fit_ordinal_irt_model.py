#!/usr/bin/env python3
"""Fit GRM/GPCM/continuous from JSONL: {"subject_id": "...", "responses": {"item": score, ...}}."""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyro
import torch

from ordinal_models import IrtFitResult, create_model

DEFAULT_SCORE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def canonicalize_score(score: float, score_values: List[float], atol: float = 1e-5) -> Tuple[float, int]:
    arr = np.asarray(score_values, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(score))))
    if not np.isclose(arr[idx], float(score), atol=atol):
        raise ValueError(f"Unexpected score {score}; nearest allowed score is {arr[idx]}.")
    return float(arr[idx]), idx


def load_jsonl_responses(input_path: Path, score_values: List[float], model_type: str = "grm") -> pd.DataFrame:
    rows = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            subject_id = str(rec["subject_id"])
            responses = rec.get("responses", rec)
            for item_id, raw_score in responses.items():
                if raw_score is None or (isinstance(raw_score, float) and np.isnan(raw_score)):
                    continue
                raw = float(raw_score)
                if model_type in ("continuous", "continuous_cat"):
                    if not (0 <= raw <= 1):
                        raise ValueError(f"Continuous model expects scores in [0,1], got {raw}")
                    rows.append({"subject_id": subject_id, "item_id": str(item_id), "score": raw, "category": 0})
                else:
                    score_val, category = canonicalize_score(raw, score_values)
                    rows.append({
                        "subject_id": subject_id,
                        "item_id": str(item_id),
                        "score": score_val,
                        "category": category,
                    })
    if not rows:
        raise ValueError(f"No valid responses found in {input_path}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="JSONL file with subject_id and responses")
    parser.add_argument("--model_type", type=str, default="grm", choices=["grm", "gpcm", "continuous", "continuous_cat"])
    parser.add_argument("--score_values", type=str, default=",".join(map(str, DEFAULT_SCORE_VALUES)))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    input_path = Path(args.input_path).expanduser().resolve()
    score_values = [float(x) for x in args.score_values.split(",")]
    if args.model_type in ("continuous", "continuous_cat"):
        score_values = [0.0, 1.0]  # placeholder for create_model

    long_df = load_jsonl_responses(input_path, score_values, model_type=args.model_type)

    subject_codes, subject_ids = pd.factorize(long_df["subject_id"], sort=True)
    item_codes, item_ids = pd.factorize(long_df["item_id"], sort=True)

    model = create_model(
        args.model_type,
        num_items=len(item_ids),
        num_subjects=len(subject_ids),
        score_values=score_values,
        device=args.device,
    )

    obs = long_df["score"].to_numpy(dtype=np.float32) if args.model_type in ("continuous", "continuous_cat") else long_df["category"].to_numpy(dtype=np.int64)
    learned = model.fit(
        subjects=np.asarray(subject_codes, dtype=np.int64),
        items=np.asarray(item_codes, dtype=np.int64),
        obs=obs,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        verbose=args.verbose,
    )

    result = IrtFitResult(
        model_type=args.model_type,
        score_values=score_values if args.model_type not in ("continuous", "continuous_cat") else [0.0, 1.0],
        ability=np.asarray(learned["ability"]),
        item_ids=list(item_ids),
        subject_ids=list(subject_ids),
        disc=np.asarray(learned["disc"]) if "disc" in learned else None,
        diff=np.asarray(learned["diff"]) if "diff" in learned else None,
        thresholds=np.asarray(learned["thresholds"]) if "thresholds" in learned else None,
        steps=np.asarray(learned["steps"]) if "steps" in learned else None,
        sigma=float(learned["sigma"]) if "sigma" in learned else None,
        expected_score=np.asarray(learned["expected_score"]) if "expected_score" in learned else None,
    )

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().with_name("params")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    prefix = out_dir / f"{stem}_{result.model_type}"

    result.item_frame().to_csv(f"{prefix}_items.csv", index=False)
    result.subject_frame().to_csv(f"{prefix}_subjects.csv", index=False)
    long_df.to_csv(f"{prefix}_long.csv", index=False)

    metadata = {
        "model_type": result.model_type,
        "score_values": result.score_values,
        "n_subjects": len(result.subject_ids),
        "n_items": len(result.item_ids),
        "item_csv": str(prefix) + "_items.csv",
        "subject_csv": str(prefix) + "_subjects.csv",
    }
    Path(f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
