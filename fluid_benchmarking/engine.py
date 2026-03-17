from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np

from fluid_benchmarking import estimators, irt_utils


def _is_ordinal_irt(irt_model: Any) -> bool:
    return isinstance(irt_model, dict) and "model_type" in irt_model


def select_mfi(theta: float, irt_model: np.ndarray, used_mask: np.ndarray, D: float) -> int:
    """Select item with max Fisher information at theta (2PL)."""
    a = irt_model[:, 0]
    b = irt_model[:, 1]
    fi = irt_utils.fisher_information(theta, a, b, D=D)
    fi_masked = np.where(~used_mask, fi, -np.inf)
    idx = int(np.argmax(fi_masked))
    if not np.isfinite(fi_masked[idx]):
        raise RuntimeError("No available items to select. All items administered?")
    return idx


def select_mfi_ordinal(theta: float, irt_model: Dict[str, Any], used_mask: np.ndarray, D: float = 1.0) -> int:
    """Select item with max Fisher information (GRM/GPCM/continuous/continuous_cat)."""
    a = np.asarray(irt_model["a"], dtype=float)
    model_type = irt_model["model_type"].lower()
    if model_type == "grm":
        thresholds = np.asarray(irt_model["thresholds"], dtype=float)
        fi = irt_utils.fisher_information_grm(theta, a, thresholds, D=D)
    elif model_type == "gpcm":
        steps = np.asarray(irt_model["steps"], dtype=float)
        fi = irt_utils.fisher_information_gpcm(theta, a, steps, D=D)
    elif model_type == "continuous_cat":
        b = np.asarray(irt_model["diff"], dtype=float)
        fi = irt_utils.fisher_information_continuous_cat(theta, a, b, D=D)
    else:
        b = np.asarray(irt_model["diff"], dtype=float)
        sigma = float(irt_model["sigma"])
        fi = irt_utils.fisher_information_continuous(theta, a, b, sigma, D=D)
    fi_masked = np.where(~used_mask, fi, -np.inf)
    idx = int(np.argmax(fi_masked))
    if not np.isfinite(fi_masked[idx]):
        raise RuntimeError("No available items to select. All items administered?")
    return idx


def run_fluid_benchmarking(
    *,

    # Core inputs
    lm_responses: np.ndarray,
    irt_model: Union[np.ndarray, Dict[str, Any]],
    start_ability: float = 0.0,
    n_max: int = 100,

    # Ability estimation
    method: Literal["map", "mle", "MAP", "MLE"] = "map",
    D: float = 1.0,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
    tol: float = 1e-6,
    max_iter: int = 100,

    # Optional override for the estimator (defaults based on irt_model type)
    estimator: Optional[Callable[..., float]] = None,
) -> Dict[str, Any]:
    """MFI item selection + ability estimation. Supports 2PL (array) or ordinal/continuous (dict)."""

    method = method.lower()
    if method not in {"map", "mle"}:
        raise ValueError("method must be 'map' or 'mle'.")

    ordinal = _is_ordinal_irt(irt_model)
    if ordinal:
        n_items = len(irt_model["a"])
        select_fn = lambda th, um: select_mfi_ordinal(th, irt_model, um, D=D)
        if estimator is None:
            estimator = estimators.ability_estimate_ordinal
        model_type = irt_model["model_type"].lower()
        if model_type in ("continuous", "continuous_cat"):
            def _valid_response(r):
                return np.isfinite(r) and 0 <= r <= 1
        else:
            def _valid_response(r):
                return np.isfinite(r) and r == int(r) and r >= 0
    else:
        irt_model = np.asarray(irt_model, dtype=float)
        if irt_model.ndim != 2 or irt_model.shape[1] != 2:
            raise ValueError("irt_model must be an (n_items, 2) array with columns [a, b].")
        n_items = irt_model.shape[0]
        select_fn = lambda th, um: select_mfi(th, irt_model, um, D=D)
        if estimator is None:
            estimator = estimators.ability_estimate
        def _valid_response(r):
            return r == 0.0 or r == 1.0

    if lm_responses.shape != (n_items,):
        raise ValueError("lm_responses must be shape (n_items,) aligned with irt_model.")

    if n_max < 1:
        return {"abilities_fb": [], "items_fb": []}

    used_mask = np.zeros(n_items, dtype=bool)
    items = []
    abilities = []
    lm_responses_running = np.full(n_items, np.nan, dtype=float)

    idx0 = select_fn(start_ability, used_mask)
    used_mask[idx0] = True
    items.append(idx0)

    r0 = lm_responses[idx0]
    if not _valid_response(r0):
        raise ValueError(f"Invalid response for item {idx0}.")
    lm_responses_running[idx0] = float(r0)

    est_kw = dict(
        lm_responses=lm_responses_running,
        irt_model=irt_model,
        method=method,
        D=D,
        mu0=mu0,
        sigma0=sigma0,
        theta0=start_ability,
        theta_range=theta_range,
        tol=tol,
        max_iter=max_iter,
    )
    th = float(estimator(**est_kw))
    abilities.append(th)

    while len(items) < n_max and len(items) < n_items:
        idx = select_fn(abilities[-1], used_mask)
        used_mask[idx] = True
        items.append(idx)

        r = lm_responses[idx]
        if not _valid_response(r):
            raise ValueError(f"Invalid response for item {idx}.")
        lm_responses_running[idx] = float(r)

        est_kw["lm_responses"] = lm_responses_running.copy()
        est_kw["theta0"] = abilities[-1]
        th = float(estimator(**est_kw))
        abilities.append(th)

    return {"abilities_fb": abilities, "items_fb": items}
