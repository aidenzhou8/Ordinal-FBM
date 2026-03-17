import os
from pathlib import Path

# Hugging Face dataset repo ID
HF_REPO_ID = "allenai/fluid-benchmarking"

# Path templates
LM_EVAL_RESULTS_PATH = "data/lm_eval_results/{}.csv"
IRT_MODELS_PATH = "data/irt_models/{}.csv"

# Supported LMs, benchmarks, and methods
LMS = [
    "amber-7b",
    "k2-65b",
    "olmo1-7b",
    "olmo2-7b",
    "pythia-7b",
    "pythia-3b",
]

BENCHMARKS = [
    "arc_challenge",
    "gsm8k",
    "hellaswag",
    "truthfulqa_mc2",
    "winogrande",
    "mmlu",
    "harmbench",
]

# Optional: local data directory for benchmarks not in the default HF dataset.
# If set (e.g. via env FLUID_BENCHMARKING_DATA_DIR), load IRT and LM eval from:
#   {data_dir}/irt_models/{benchmark}.csv
#   {data_dir}/lm_eval_results/{benchmark}/{lm}.csv
# Use for harmbench - see scripts/prepare_harmbench_data.py
_DATA_DIR = os.environ.get("FLUID_BENCHMARKING_DATA_DIR")
DATA_DIR = Path(_DATA_DIR) if _DATA_DIR else None

METHODS = [
    "full_accuracy",
    "full_ability",
    "random_accuracy",
    "random_ability",
    "fluid_benchmarking",
]

IRT_METHODS = [
    "full_ability", 
    "random_ability", 
    "fluid_benchmarking",
]

# Default parameters for IRT-based methods and Fluid Benchmarking
ESTIMATION_METHOD_IRT = "map"

# Ordinal: SCORE_VALUES + IRT_MODEL_TYPE in (grm, gpcm). Continuous: IRT_MODEL_TYPE="continuous"
SCORE_VALUES = None
IRT_MODEL_TYPE = "2pl"

# Default evaluation sample sizes
N_SAMPLES_LIST = (
    list(range(1, 10)) + 
    list(range(10, 100, 10)) + 
    list(range(100, 600, 100))
)
