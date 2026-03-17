import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tqdm

from fluid_benchmarking import config, datasets, evaluation, indexing


def run_experiments(
    benchmarks: Iterable[str],
    lms: Iterable[str],
    seed: int = 0,
    *,
    score_values=None,
    irt_model_type=None,
) -> pd.DataFrame:
    irt_model_type = (irt_model_type or config.IRT_MODEL_TYPE).lower()
    use_ordinal = irt_model_type in ("grm", "gpcm", "continuous", "continuous_cat")
    if use_ordinal and config.DATA_DIR is None:
        raise ValueError("Ordinal/continuous mode requires DATA_DIR.")
        if irt_model_type in ("grm", "gpcm"):
            score_values = score_values or config.SCORE_VALUES or [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            score_values = None  # continuous uses raw floats

    lm_eval_results = {}
    for lm in lms:
        kwargs = {}
        if use_ordinal:
            kwargs["binary"] = False
            if score_values is not None:
                kwargs["score_values"] = score_values
        lm_eval_results[lm] = datasets.load_lm_eval_results(
            config.HF_REPO_ID,
            config.LM_EVAL_RESULTS_PATH.format(lm),
            **kwargs,
        )

    open_llm_leaderboard_results = datasets.load_open_llm_leaderboard_results()

    random.seed(seed)

    rows = []
    for benchmark in tqdm.tqdm(list(benchmarks), desc="Benchmarks"):

        if use_ordinal:
            irt_model_benchmark = datasets.load_ordinal_irt_model(
                benchmark, model_type=irt_model_type
            )
        else:
            irt_model_benchmark = datasets.load_irt_model(
                config.HF_REPO_ID,
                config.IRT_MODELS_PATH.format(benchmark),
                benchmark=benchmark,
            )
            assert irt_model_benchmark.columns[0] == "a"
            assert irt_model_benchmark.columns[1] == "b"

        if use_ordinal:
            n_items = len(irt_model_benchmark["a"])
        else:
            n_items = len(irt_model_benchmark)

        samples_dict = {}
        for n_samples in config.N_SAMPLES_LIST:
            if n_samples > n_items:
                continue
            samples_dict[n_samples] = np.array(
                random.sample(range(n_items), n_samples)
            )

        if benchmark in open_llm_leaderboard_results:
            start_ability = float(
                np.mean(list(open_llm_leaderboard_results[benchmark]["ability"].values()))
            )
        else:
            start_ability = 0.0

        for lm in tqdm.tqdm(
            list(lms),
            desc=f"{benchmark} • LMs",
            leave=False
        ):
            if config.DATA_DIR is not None and (config.DATA_DIR / "lm_eval_results" / benchmark / f"{lm}.csv").exists():
                load_kw = dict(
                    repo_id=config.HF_REPO_ID,
                    filename=config.LM_EVAL_RESULTS_PATH.format(lm),
                    benchmark=benchmark,
                    lm=lm,
                )
                if use_ordinal:
                    load_kw["binary"] = False
                    if score_values is not None:
                        load_kw["score_values"] = score_values
                lm_eval_results_benchmark = datasets.load_lm_eval_results(**load_kw)
            else:
                lm_eval_results_benchmark = indexing.filter_benchmark(
                    lm_eval_results[lm],
                    benchmark,
                )

            if use_ordinal:
                irt_model_benchmark = datasets.align_ordinal_irt_to_items(
                    irt_model_benchmark,
                    lm_eval_results_benchmark.index.tolist(),
                )
            else:
                assert (lm_eval_results_benchmark.index == irt_model_benchmark.index).all()

            for checkpoint in tqdm.tqdm(
                list(lm_eval_results_benchmark.columns),
                desc=f"{lm} • checkpoints",
                leave=False,
            ):
                lm_responses = np.array(lm_eval_results_benchmark[checkpoint])
                row = evaluation.iterate_evals(
                    lm_responses=lm_responses,
                    methods=config.METHODS,
                    irt_model=irt_model_benchmark,
                    estimation_method_irt=config.ESTIMATION_METHOD_IRT,
                    samples_dict=samples_dict,
                    start_ability_fb=start_ability,
                )
                row["benchmark"] = benchmark
                row["lm"] = lm
                row["checkpoint"] = checkpoint
                rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=config.BENCHMARKS,
        help="Benchmarks to evaluate.",
    )
    parser.add_argument(
        "--lms",
        nargs="+",
        default=config.LMS,
        help="LMs to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subset sampling.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/experiments.jsonl"),
        help="Output file.",
    )
    parser.add_argument(
        "--score-values",
        type=str,
        default=None,
        help="Comma-separated score values for ordinal mode (e.g. 0,0.25,0.5,0.75,1).",
    )
    parser.add_argument(
        "--irt-model-type",
        type=str,
        default=None,
        choices=["2pl", "grm", "gpcm", "continuous", "continuous_cat"],
        help="IRT model type. Use grm/gpcm with --score-values for ordinal; continuous uses raw scores.",
    )
    args = parser.parse_args()

    score_values = None
    if args.score_values:
        score_values = [float(x) for x in args.score_values.split(",")]
    irt_model_type = args.irt_model_type or config.IRT_MODEL_TYPE

    df = run_experiments(
        benchmarks=args.benchmarks,
        lms=args.lms,
        seed=args.seed,
        score_values=score_values,
        irt_model_type=irt_model_type,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.out, orient="records", lines=True)
    df.to_pickle(args.out.with_suffix(".pkl"))
    print(f"Wrote {len(df):,} rows to {args.out} and {args.out.with_suffix('.pkl')}.")


if __name__ == "__main__":
    main()
