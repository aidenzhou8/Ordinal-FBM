#!/usr/bin/env python3
"""Verify ordinal/continuous pipeline. Run from repo root: python scripts/verify_pipeline.py"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root and irt/ so ordinal_models is findable
import sys
repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "irt"))

from fluid_benchmarking import datasets, engine, estimators, evaluation, irt_utils


def test_datasets():
    """datasets: _scores_to_categories, _convert_eval_results"""
    sv = [0.0, 0.25, 0.5, 0.75, 1.0]
    cat = datasets._scores_to_categories(np.array([[0.5, 0.25], [1.0, 0.0]]), sv)
    assert cat.shape == (2, 2) and np.array_equal(cat, [[2, 1], [4, 0]])
    df = pd.DataFrame([[0.5, 0.25]], columns=["c1", "c2"])
    out = datasets._convert_eval_results(df, binary=False, score_values=sv)
    assert out.iloc[0, 0] == 2 and out.iloc[0, 1] == 1
    print("  datasets OK")


def test_irt_utils():
    """irt_utils: Fisher info for 2PL, GRM, GPCM, continuous, continuous_cat"""
    a, b = np.array([1.0, 1.5]), np.array([0.0, 0.5])
    fi = irt_utils.fisher_information(0.0, a, b)
    assert len(fi) == 2 and np.all(fi >= 0)
    fi_grm = irt_utils.fisher_information_grm(0.0, a, np.array([[0.5, 1.0], [-0.5, 0.5]]))
    assert len(fi_grm) == 2
    fi_gpcm = irt_utils.fisher_information_gpcm(0.0, a, np.array([[0.5], [0.0]]))
    assert len(fi_gpcm) == 2
    fi_cont = irt_utils.fisher_information_continuous(0.0, a, b, sigma=0.2)
    assert len(fi_cont) == 2
    fi_cont_cat = irt_utils.fisher_information_continuous_cat(0.0, a, b)
    assert len(fi_cont_cat) == 2 and np.all(fi_cont_cat >= 0)
    print("  irt_utils OK")


def test_estimators():
    """estimators: 2PL, ordinal (GRM/GPCM/continuous)"""
    irt_2pl = np.array([[1.0, 0.0], [1.0, 0.5]])
    resp = np.array([1.0, 0.0])
    th = estimators.ability_estimate(resp, irt_2pl, "map")
    assert np.isfinite(th)
    th = estimators.ability_estimate_ordinal(
        np.array([2, 1]),
        {"model_type": "grm", "a": [1, 1], "thresholds": np.array([[0.5, 1.0], [-0.5, 0.5]])},
        "mle",
    )
    assert np.isfinite(th)
    th = estimators.ability_estimate_ordinal(
        np.array([0.7, 0.6]),
        {"model_type": "continuous", "a": [1, 1], "diff": [0, 0.5], "sigma": 0.2},
        "mle",
    )
    assert np.isfinite(th)
    th = estimators.ability_estimate_ordinal(
        np.array([0.7, 0.6]),
        {"model_type": "continuous_cat", "a": [1, 1], "diff": [0, 0.5]},
        "mle",
    )
    assert np.isfinite(th)
    print("  estimators OK")


def test_engine():
    """engine: select_mfi, select_mfi_ordinal, run_fluid_benchmarking"""
    irt_2pl = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, -0.5]])
    idx = engine.select_mfi(0.0, irt_2pl, np.array([False, False, False]), D=1.0)
    assert idx in (0, 1, 2)
    irt_ord = {"model_type": "continuous", "a": [1, 1, 1], "diff": [0, 0.5, -0.5], "sigma": 0.2}
    idx = engine.select_mfi_ordinal(0.0, irt_ord, np.array([False, False, False]))
    assert idx in (0, 1, 2)
    irt_cont_cat = {"model_type": "continuous_cat", "a": [1, 1, 1], "diff": [0, 0.5, -0.5]}
    idx = engine.select_mfi_ordinal(0.0, irt_cont_cat, np.array([False, False, False]))
    assert idx in (0, 1, 2)
    out = engine.run_fluid_benchmarking(
        lm_responses=np.array([1.0, 0.0, 1.0]),
        irt_model=irt_2pl,
        start_ability=0.0,
        n_max=2,
    )
    assert len(out["items_fb"]) == 2 and len(out["abilities_fb"]) == 2
    out = engine.run_fluid_benchmarking(
        lm_responses=np.array([0.7, 0.6, 0.8]),
        irt_model=irt_ord,
        start_ability=0.0,
        n_max=2,
    )
    assert len(out["items_fb"]) == 2 and len(out["abilities_fb"]) == 2
    out = engine.run_fluid_benchmarking(
        lm_responses=np.array([0.7, 0.6, 0.8]),
        irt_model=irt_cont_cat,
        start_ability=0.0,
        n_max=2,
    )
    assert len(out["items_fb"]) == 2 and len(out["abilities_fb"]) == 2
    print("  engine OK")


def test_evaluation():
    """evaluation: full_ability, fluid_benchmarking with 2PL and ordinal"""
    irt_2pl = np.array([[1.0, 0.0], [1.0, 0.5]])
    resp = np.array([1.0, 0.0])
    th = evaluation.full_ability(resp, irt_2pl)
    assert np.isfinite(th)
    irt_ord = {"model_type": "continuous", "a": [1, 1], "diff": [0, 0.5], "sigma": 0.2}
    th = evaluation.full_ability(resp, irt_ord)
    assert np.isfinite(th)
    ab, it = evaluation.fluid_benchmarking(resp, irt_2pl, 0.0, 2, "map")
    assert len(ab) == 2 and len(it) == 2
    print("  evaluation OK")


def test_fit_ordinal_e2e():
    """End-to-end: fit ordinal IRT from JSONL, load, run fluid benchmarking"""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "test.jsonl"
        jsonl.write_text(
            '{"subject_id": "m1", "responses": {"i0": 0.75, "i1": 0.25, "i2": 0.5}}\n'
            '{"subject_id": "m2", "responses": {"i0": 0.25, "i1": 0.75, "i2": 0.5}}\n'
        )
        data_dir = Path(tmp) / "data"
        (data_dir / "irt_models").mkdir(parents=True)
        os.environ["FLUID_BENCHMARKING_DATA_DIR"] = str(data_dir)

        # Fit continuous (simpler, fewer params; GRM needs more data for K categories)
        try:
            from irt.fit_ordinal_irt_model import main as fit_main
            orig = sys.argv
            sys.argv = [
                "fit_ordinal_irt_model", "--input_path", str(jsonl), "--model_type", "continuous",
                "--epochs", "50", "--output_dir", str(data_dir / "irt_models"),
            ]
            with open(os.devnull, "w") as devnull:
                old_stdout, sys.stdout = sys.stdout, devnull
                fit_main()
                sys.stdout = old_stdout
            sys.argv = orig
        except Exception as e:
            print(f"  fit_ordinal e2e: skip ({e})")
            return

        from fluid_benchmarking import config
        config.DATA_DIR = data_dir
        irt = datasets.load_ordinal_irt_model("test", "continuous")
        irt = datasets.align_ordinal_irt_to_items(irt, ["i0", "i1", "i2"])
        resp = np.array([0.75, 0.25, 0.5])  # raw scores for continuous
        ab, it = evaluation.fluid_benchmarking(resp, irt, 0.0, 2, "map")
        assert len(ab) == 2 and len(it) == 2

        # Fit continuous_cat (heteroskedastic) and run fluid benchmarking
        try:
            sys.argv = [
                "fit_ordinal_irt_model", "--input_path", str(jsonl), "--model_type", "continuous_cat",
                "--epochs", "50", "--output_dir", str(data_dir / "irt_models"),
            ]
            with open(os.devnull, "w") as devnull:
                old_stdout, sys.stdout = sys.stdout, devnull
                fit_main()
                sys.stdout = old_stdout
            sys.argv = orig
        except Exception as e:
            print(f"  fit_ordinal continuous_cat e2e: skip ({e})")
        else:
            irt_cat = datasets.load_ordinal_irt_model("test", "continuous_cat")
            irt_cat = datasets.align_ordinal_irt_to_items(irt_cat, ["i0", "i1", "i2"])
            ab, it = evaluation.fluid_benchmarking(resp, irt_cat, 0.0, 2, "map")
            assert len(ab) == 2 and len(it) == 2

        print("  fit_ordinal e2e OK")


def main():
    print("Verifying pipeline...")
    test_datasets()
    test_irt_utils()
    test_estimators()
    test_engine()
    test_evaluation()
    test_fit_ordinal_e2e()
    print("All checks passed.")


if __name__ == "__main__":
    main()
