import numpy as np


def sigmoid_stable(
    z: np.ndarray,
) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def fisher_information(theta: float, a: np.ndarray, b: np.ndarray, D: float = 1.0) -> np.ndarray:
    z = D * a * (theta - b)
    P = sigmoid_stable(z)
    return (D**2) * (a**2) * (P * (1.0 - P))


def fisher_information_grm(theta: float, a: np.ndarray, thresholds: np.ndarray, D: float = 1.0) -> np.ndarray:
    """Fisher information for GRM at theta."""
    a = np.asarray(a, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    n_items = a.shape[0]
    eta = D * a[:, None] * (float(theta) - thresholds)
    P_star = np.concatenate([np.ones((n_items, 1)), sigmoid_stable(eta), np.zeros((n_items, 1))], axis=1)
    P = P_star[:, :-1] - P_star[:, 1:]
    dP_star = D * a[:, None] * P_star[:, 1:-1] * (1.0 - P_star[:, 1:-1])
    dP_star_full = np.concatenate([np.zeros((n_items, 1)), dP_star, np.zeros((n_items, 1))], axis=1)
    dP = dP_star_full[:, :-1] - dP_star_full[:, 1:]
    eps = 1e-10
    info = np.sum(dP**2 / np.maximum(P, eps), axis=1)
    return info


def fisher_information_gpcm(theta: float, a: np.ndarray, steps: np.ndarray, D: float = 1.0) -> np.ndarray:
    """Fisher information for GPCM at theta."""
    a = np.asarray(a, dtype=float)
    steps = np.asarray(steps, dtype=float)
    n_items, K = a.shape[0], steps.shape[1] + 1
    partials = D * a[:, None] * (float(theta) - steps)
    logits = np.concatenate([np.zeros((n_items, 1)), np.cumsum(partials, axis=1)], axis=1)
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    P = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    dlogit_dtheta = np.zeros((n_items, K))
    for j in range(1, K):
        dlogit_dtheta[:, j] = D * a * j
    sum_P_dlogit = np.sum(P * dlogit_dtheta, axis=1, keepdims=True)
    dP = P * (dlogit_dtheta - sum_P_dlogit)
    eps = 1e-10
    info = np.sum(dP**2 / np.maximum(P, eps), axis=1)
    return info


def fisher_information_continuous(theta: float, a: np.ndarray, b: np.ndarray, sigma: float, D: float = 1.0) -> np.ndarray:
    """Fisher information for continuous (Gaussian-on-sigmoid) model."""
    z = D * a * (theta - b)
    P = sigmoid_stable(z)
    return (D * a * P * (1.0 - P)) ** 2 / (sigma**2)


def fisher_information_continuous_cat(theta: float, a: np.ndarray, b: np.ndarray, D: float = 1.0, epsilon: float = 1e-15) -> np.ndarray:
    """Fisher information for continuous-cat (heteroskedastic) model.

    From https://github.com/trismik/continuous-cat: FI = a² × μ(1-μ)
    where μ = sigmoid(θ - b). Variance σ² = μ(1-μ)/a² matches Bernoulli structure.
    """
    z = D * (float(theta) - np.asarray(b, dtype=float))
    mu = sigmoid_stable(z)
    mu = np.clip(mu, epsilon, 1.0 - epsilon)
    a = np.asarray(a, dtype=float)
    return np.maximum((a**2) * mu * (1.0 - mu), epsilon)
