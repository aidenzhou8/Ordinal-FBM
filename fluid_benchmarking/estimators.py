from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np

from fluid_benchmarking import irt_utils

OrdinalIRTModel = Dict[str, Any]


def ability_estimate(
    lm_responses: np.ndarray,
    irt_model: np.ndarray,
    method: Literal["map", "mle", "MAP", "MLE"] = "map",
    D: float = 1.0,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    theta0: Optional[float] = None,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """MAP or MLE ability estimate for 2PL. lm_responses: binary 0/1, NaN=unadministered."""

    method = method.lower()
    if method not in {"map", "mle"}:
        raise ValueError("method must be 'map' or 'mle'.")

    lm_responses = np.asarray(lm_responses, dtype=float)
    irt_model = np.asarray(irt_model, dtype=float)
    
    if irt_model.ndim != 2 or irt_model.shape[1] != 2:
        raise ValueError("irt_model must be an (n_items, 2) array with columns [a, b].")
    n_items = irt_model.shape[0]

    if lm_responses.shape != (n_items,):
        raise ValueError("lm_responses must be shape (n_items,) aligned with irt_model.")

    mask = ~np.isnan(lm_responses)
    a = irt_model[mask, 0]
    b = irt_model[mask, 1]
    lm_responses = lm_responses[mask]

    if method == "map":
        if sigma0 <= 0:
            raise ValueError("sigma0 must be positive for MAP.")
        inv_sigma2 = 1.0 / (sigma0 * sigma0)
    else:
        inv_sigma2 = 0.0

    low, high = float(theta_range[0]), float(theta_range[1])
    if low >= high:
        raise ValueError("theta_range must have low < high.")

    def score(
        theta: float
    ) -> float:
        z = D * a * (theta - b)
        P = irt_utils.sigmoid_stable(z)
        prior_term = (mu0 - theta) * inv_sigma2
        likelihood_term = D * np.sum(a * (lm_responses - P))
        return prior_term + likelihood_term

    def score_prime(
        theta: float
    ) -> float:
        z = D * a * (theta - b)
        P = irt_utils.sigmoid_stable(z)
        PQ = P * (1.0 - P)
        prior_term = -inv_sigma2
        likelihood_term = -(D**2) * np.sum(a * a * PQ)
        return prior_term + likelihood_term

    theta = mu0 if theta0 is None else theta0
    theta = float(np.clip(theta, low, high))

    for _ in range(max_iter):
        T = score(theta)
        if np.abs(T) < tol:
            return theta

        Tp = score_prime(theta)
        if not np.isfinite(Tp) or Tp == 0.0:
            break

        step = -T / Tp
        new_theta = theta + step

        if new_theta < low or new_theta > high or not np.isfinite(new_theta):
            new_theta = float(np.clip(new_theta, low, high))

        T_abs = abs(T)
        for _bt in range(15):
            T_new = score(new_theta)
            if abs(T_new) < T_abs or not np.isfinite(T_new):
                break
            new_theta = 0.5 * (new_theta + theta)

        theta = new_theta

    sL = score(low)
    sH = score(high)

    if sL * sH <= 0:
        lo, hi = low, high
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            sM = score(mid)
            if abs(sM) < tol:
                return mid
            if sL * sM > 0:
                lo, sL = mid, sM
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return high if (sL > 0 and sH > 0) else low


def _grm_prob_and_deriv(
    theta: float, a: np.ndarray, thresholds: np.ndarray, D: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (P, dP) for GRM: P shape (n_items, K), dP shape (n_items, K)."""
    eta = D * a[:, None] * (theta - thresholds)
    P_star = irt_utils.sigmoid_stable(eta)
    P_star = np.concatenate([np.ones((a.shape[0], 1)), P_star, np.zeros((a.shape[0], 1))], axis=1)
    P = P_star[:, :-1] - P_star[:, 1:]
    dP_star = D * a[:, None] * P_star[:, 1:-1] * (1.0 - P_star[:, 1:-1])
    dP_star_full = np.concatenate([np.zeros((a.shape[0], 1)), dP_star, np.zeros((a.shape[0], 1))], axis=1)
    dP = dP_star_full[:, :-1] - dP_star_full[:, 1:]
    return P, dP


def _gpcm_prob_and_deriv(
    theta: float, a: np.ndarray, steps: np.ndarray, D: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (P, dP) for GPCM: P shape (n_items, K), dP shape (n_items, K)."""
    partials = D * a[:, None] * (theta - steps)
    logits = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(partials, axis=1)], axis=1)
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    P = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    K = P.shape[1]
    dlogit_dtheta = np.zeros((a.shape[0], K))
    for j in range(1, K):
        dlogit_dtheta[:, j] = D * a * j
    sum_P_dlogit = np.sum(P * dlogit_dtheta, axis=1, keepdims=True)
    dP = P * (dlogit_dtheta - sum_P_dlogit)
    return P, dP


def ability_estimate_ordinal(
    lm_responses: np.ndarray,
    irt_model: OrdinalIRTModel,
    method: Literal["map", "mle", "MAP", "MLE"] = "map",
    D: float = 1.0,
    mu0: float = 0.0,
    sigma0: float = 1.0,
    theta0: Optional[float] = None,
    theta_range: Tuple[float, float] = (-4.0, 4.0),
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """MAP/MLE for GRM, GPCM, or continuous. lm_responses: categories (ordinal) or raw [0,1] (continuous)."""
    method = method.lower()
    if method not in {"map", "mle"}:
        raise ValueError("method must be 'map' or 'mle'.")

    lm_responses = np.asarray(lm_responses, dtype=float)
    a = np.asarray(irt_model["a"], dtype=float)
    n_items = a.shape[0]
    if lm_responses.shape != (n_items,):
        raise ValueError("lm_responses must align with irt_model.")

    mask = ~np.isnan(lm_responses)
    a_m = a[mask]

    if method == "map" and sigma0 <= 0:
        raise ValueError("sigma0 must be positive for MAP.")
    inv_sigma2 = 1.0 / (sigma0 * sigma0) if method == "map" else 0.0

    low, high = float(theta_range[0]), float(theta_range[1])
    if low >= high:
        raise ValueError("theta_range must have low < high.")

    model_type = irt_model["model_type"].lower()
    if model_type == "continuous":
        b = np.asarray(irt_model["diff"], dtype=float)[mask]
        sigma = float(irt_model["sigma"])
        y = lm_responses[mask]

        def score(theta: float) -> float:
            z = D * a_m * (theta - b)
            P = irt_utils.sigmoid_stable(z)
            dmean = D * a_m * P * (1.0 - P)
            return (mu0 - theta) * inv_sigma2 + np.sum((y - P) / (sigma**2) * dmean)

        def score_prime(theta: float) -> float:
            fi = irt_utils.fisher_information_continuous(theta, a_m, b, sigma, D)
            return -inv_sigma2 - np.sum(fi)
    elif model_type == "continuous_cat":
        b = np.asarray(irt_model["diff"], dtype=float)[mask]
        y = lm_responses[mask]

        def score(theta: float) -> float:
            z = D * (theta - b)
            P = irt_utils.sigmoid_stable(z)
            P = np.clip(P, 1e-15, 1.0 - 1e-15)
            return (mu0 - theta) * inv_sigma2 + np.sum((a_m**2) * (y - P))

        def score_prime(theta: float) -> float:
            fi = irt_utils.fisher_information_continuous_cat(theta, a_m, b, D)
            return -inv_sigma2 - np.sum(fi)
    elif model_type == "grm":
        categories = np.asarray(lm_responses[mask], dtype=np.int64)
        thresholds = np.asarray(irt_model["thresholds"], dtype=float)[mask]
        thresholds = thresholds.reshape(a_m.shape[0], -1)

        def score(theta: float) -> float:
            P, dP = _grm_prob_and_deriv(theta, a_m, thresholds, D)
            eps = 1e-10
            lik = np.sum(dP[np.arange(len(categories)), categories] / np.maximum(P[np.arange(len(categories)), categories], eps))
            return (mu0 - theta) * inv_sigma2 + lik

        def score_prime(theta: float) -> float:
            fi = irt_utils.fisher_information_grm(theta, a_m, thresholds, D)
            return -inv_sigma2 - np.sum(fi)
    else:
        categories = np.asarray(lm_responses[mask], dtype=np.int64)
        steps = np.asarray(irt_model["steps"], dtype=float)[mask]
        steps = steps.reshape(a_m.shape[0], -1)

        def score(theta: float) -> float:
            P, dP = _gpcm_prob_and_deriv(theta, a_m, steps, D)
            eps = 1e-10
            lik = np.sum(dP[np.arange(len(categories)), categories] / np.maximum(P[np.arange(len(categories)), categories], eps))
            return (mu0 - theta) * inv_sigma2 + lik

        def score_prime(theta: float) -> float:
            fi = irt_utils.fisher_information_gpcm(theta, a_m, steps, D)
            return -inv_sigma2 - np.sum(fi)

    theta = mu0 if theta0 is None else theta0
    theta = float(np.clip(theta, low, high))

    for _ in range(max_iter):
        T = score(theta)
        if np.abs(T) < tol:
            return theta
        Tp = score_prime(theta)
        if not np.isfinite(Tp) or Tp == 0.0:
            break
        step = -T / Tp
        new_theta = theta + step
        if new_theta < low or new_theta > high or not np.isfinite(new_theta):
            new_theta = float(np.clip(new_theta, low, high))
        T_abs = abs(T)
        for _bt in range(15):
            T_new = score(new_theta)
            if abs(T_new) < T_abs or not np.isfinite(T_new):
                break
            new_theta = 0.5 * (new_theta + theta)
        theta = new_theta

    sL, sH = score(low), score(high)
    if sL * sH <= 0:
        lo, hi = low, high
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            sM = score(mid)
            if abs(sM) < tol:
                return mid
            if sL * sM > 0:
                lo, sL = mid, sM
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return high if (sL > 0 and sH > 0) else low
