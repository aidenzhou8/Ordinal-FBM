"""Ordinal IRT models: GRM, GPCM, continuous."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam


EPS = 1e-6


@dataclass
class IrtFitResult:
    model_type: str
    score_values: List[float]
    ability: np.ndarray
    item_ids: List[str]
    subject_ids: List[str]
    disc: Optional[np.ndarray] = None
    diff: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None
    steps: Optional[np.ndarray] = None
    sigma: Optional[float] = None
    expected_score: Optional[np.ndarray] = None

    def item_frame(self):
        import pandas as pd

        data = {"item_id": self.item_ids}
        if self.disc is not None:
            data["a"] = self.disc
        if self.diff is not None:
            data["b"] = self.diff
        if self.thresholds is not None:
            for j in range(self.thresholds.shape[1]):
                data[f"b{j+1}"] = self.thresholds[:, j]
        if self.steps is not None:
            for j in range(self.steps.shape[1]):
                data[f"step{j+1}"] = self.steps[:, j]
        if self.expected_score is not None:
            data["expected_score_at_theta0"] = self.expected_score
        if self.sigma is not None:
            data["sigma"] = self.sigma
        return pd.DataFrame(data)

    def subject_frame(self):
        import pandas as pd

        return pd.DataFrame({"subject_id": self.subject_ids, "theta": self.ability})


def _ordered_thresholds(base: torch.Tensor, gaps_raw: torch.Tensor) -> torch.Tensor:
    gaps = F.softplus(gaps_raw) + 1e-3
    return base.unsqueeze(-1) + torch.cumsum(gaps, dim=-1)


class BaseSafetyIrt:
    def __init__(
        self,
        *,
        num_items: int,
        num_subjects: int,
        score_values: Iterable[float],
        device: str = "cpu",
    ):
        self.num_items = int(num_items)
        self.num_subjects = int(num_subjects)
        self.score_values = torch.tensor(list(score_values), dtype=torch.float32, device=device)
        self.num_categories = len(self.score_values)
        self.device = device
        self.guide = None

    def fit(
        self,
        *,
        subjects: np.ndarray,
        items: np.ndarray,
        obs: np.ndarray,
        lr: float = 0.03,
        epochs: int = 3000,
        seed: int = 0,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)

        subjects_t = torch.as_tensor(subjects, dtype=torch.long, device=self.device)
        items_t = torch.as_tensor(items, dtype=torch.long, device=self.device)
        obs_t = self._obs_tensor(obs)

        self.guide = AutoNormal(self.model)
        svi = SVI(self.model, self.guide, Adam({"lr": lr}), loss=Trace_ELBO())

        for epoch in range(epochs):
            loss = svi.step(subjects_t, items_t, obs_t)
            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                print(f"[{self.__class__.__name__}] epoch={epoch:5d} loss={loss:.4f}")

        return self.export(subjects_t, items_t)

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def model(self, subjects: torch.Tensor, items: torch.Tensor, obs: torch.Tensor):
        raise NotImplementedError

    def export(self, subjects: torch.Tensor, items: torch.Tensor) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def predict_expected_score(
        self,
        abilities: np.ndarray,
        item_params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError


class GradedResponseModel(BaseSafetyIrt):
    """Graded response model for ordered score buckets."""

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.long, device=self.device)

    def model(self, subjects: torch.Tensor, items: torch.Tensor, obs: torch.Tensor):
        theta = pyro.sample(
            "theta",
            dist.Normal(torch.zeros(self.num_subjects, device=self.device), torch.ones(self.num_subjects, device=self.device)).to_event(1),
        )
        log_a = pyro.sample(
            "log_a",
            dist.Normal(torch.zeros(self.num_items, device=self.device), 0.5 * torch.ones(self.num_items, device=self.device)).to_event(1),
        )
        base = pyro.sample(
            "base",
            dist.Normal(torch.zeros(self.num_items, device=self.device), torch.ones(self.num_items, device=self.device)).to_event(1),
        )
        gap_raw = pyro.sample(
            "gap_raw",
            dist.Normal(torch.zeros((self.num_items, self.num_categories - 1), device=self.device), torch.ones((self.num_items, self.num_categories - 1), device=self.device)).to_event(2),
        )

        a = torch.exp(log_a)
        thresholds = _ordered_thresholds(base, gap_raw)

        eta = a[items].unsqueeze(-1) * (theta[subjects].unsqueeze(-1) - thresholds[items])
        s = torch.sigmoid(eta)
        probs = torch.cat(
            [
                (1.0 - s[:, :1]),
                s[:, :-1] - s[:, 1:],
                s[:, -1:],
            ],
            dim=1,
        )
        probs = probs.clamp_min(EPS)
        probs = probs / probs.sum(dim=1, keepdim=True)

        with pyro.plate("responses", obs.shape[0]):
            pyro.sample("obs", dist.Categorical(probs=probs), obs=obs)

    def export(self, subjects: torch.Tensor, items: torch.Tensor) -> Dict[str, np.ndarray]:
        q = self.guide.quantiles([0.5])
        theta = q["theta"][0].detach().cpu().numpy()
        a = torch.exp(q["log_a"][0]).detach().cpu().numpy()
        base = q["base"][0]
        gaps = q["gap_raw"][0]
        thresholds = _ordered_thresholds(base, gaps).detach().cpu().numpy()

        expected = self.predict_expected_score(theta, {"disc": a, "thresholds": thresholds})
        return {
            "ability": theta,
            "disc": a,
            "thresholds": thresholds,
            "expected_score": expected,
        }

    def predict_expected_score(self, abilities: np.ndarray, item_params: Dict[str, np.ndarray]) -> np.ndarray:
        theta = torch.as_tensor(abilities, dtype=torch.float32)
        a = torch.as_tensor(item_params["disc"], dtype=torch.float32)
        thresholds = torch.as_tensor(item_params["thresholds"], dtype=torch.float32)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        eta = a.unsqueeze(0).unsqueeze(-1) * (theta - thresholds.unsqueeze(0))
        s = torch.sigmoid(eta)
        probs = torch.cat([1.0 - s[..., :1], s[..., :-1] - s[..., 1:], s[..., -1:]], dim=-1)
        probs = probs.clamp_min(EPS)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        expected = (probs * self.score_values.cpu()).sum(dim=-1)
        return expected.mean(dim=0).numpy()


class GeneralizedPartialCreditModel(BaseSafetyIrt):
    """1D GPCM with per-item discrimination and ordered step parameters."""

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.long, device=self.device)

    def model(self, subjects: torch.Tensor, items: torch.Tensor, obs: torch.Tensor):
        theta = pyro.sample(
            "theta",
            dist.Normal(torch.zeros(self.num_subjects, device=self.device), torch.ones(self.num_subjects, device=self.device)).to_event(1),
        )
        log_a = pyro.sample(
            "log_a",
            dist.Normal(torch.zeros(self.num_items, device=self.device), 0.5 * torch.ones(self.num_items, device=self.device)).to_event(1),
        )
        base = pyro.sample(
            "base",
            dist.Normal(torch.zeros(self.num_items, device=self.device), torch.ones(self.num_items, device=self.device)).to_event(1),
        )
        gap_raw = pyro.sample(
            "gap_raw",
            dist.Normal(torch.zeros((self.num_items, self.num_categories - 1), device=self.device), torch.ones((self.num_items, self.num_categories - 1), device=self.device)).to_event(2),
        )

        a = torch.exp(log_a)
        steps = _ordered_thresholds(base, gap_raw)

        partials = a[items].unsqueeze(-1) * (theta[subjects].unsqueeze(-1) - steps[items])
        logits = torch.cat(
            [torch.zeros((obs.shape[0], 1), device=self.device), torch.cumsum(partials, dim=-1)],
            dim=1,
        )
        probs = torch.softmax(logits, dim=-1).clamp_min(EPS)
        probs = probs / probs.sum(dim=1, keepdim=True)

        with pyro.plate("responses", obs.shape[0]):
            pyro.sample("obs", dist.Categorical(probs=probs), obs=obs)

    def export(self, subjects: torch.Tensor, items: torch.Tensor) -> Dict[str, np.ndarray]:
        q = self.guide.quantiles([0.5])
        theta = q["theta"][0].detach().cpu().numpy()
        a = torch.exp(q["log_a"][0]).detach().cpu().numpy()
        base = q["base"][0]
        gaps = q["gap_raw"][0]
        steps = _ordered_thresholds(base, gaps).detach().cpu().numpy()

        expected = self.predict_expected_score(theta, {"disc": a, "steps": steps})
        return {
            "ability": theta,
            "disc": a,
            "steps": steps,
            "expected_score": expected,
        }

    def predict_expected_score(self, abilities: np.ndarray, item_params: Dict[str, np.ndarray]) -> np.ndarray:
        theta = torch.as_tensor(abilities, dtype=torch.float32)
        a = torch.as_tensor(item_params["disc"], dtype=torch.float32)
        steps = torch.as_tensor(item_params["steps"], dtype=torch.float32)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        partials = a.unsqueeze(0).unsqueeze(-1) * (theta - steps.unsqueeze(0))
        logits = torch.cat([torch.zeros((*partials.shape[:-1], 1), device=partials.device), torch.cumsum(partials, dim=-1)], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * self.score_values.cpu()).sum(dim=-1)
        return expected.mean(dim=0).numpy()


class ContinuousSafetyModel(BaseSafetyIrt):
    """Naive continuous model: Gaussian likelihood around sigmoid(a*(theta-b))."""

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def model(self, subjects: torch.Tensor, items: torch.Tensor, obs: torch.Tensor):
        theta = pyro.sample(
            "theta",
            dist.Normal(torch.zeros(self.num_subjects, device=self.device), torch.ones(self.num_subjects, device=self.device)).to_event(1),
        )
        log_a = pyro.sample(
            "log_a",
            dist.Normal(torch.zeros(self.num_items, device=self.device), 0.5 * torch.ones(self.num_items, device=self.device)).to_event(1),
        )
        b = pyro.sample(
            "b",
            dist.Normal(torch.zeros(self.num_items, device=self.device), torch.ones(self.num_items, device=self.device)).to_event(1),
        )
        log_sigma = pyro.sample("log_sigma", dist.Normal(torch.tensor(-1.5, device=self.device), torch.tensor(0.5, device=self.device)))

        a = torch.exp(log_a)
        sigma = F.softplus(log_sigma) + 1e-3
        mean = torch.sigmoid(a[items] * (theta[subjects] - b[items]))

        with pyro.plate("responses", obs.shape[0]):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=obs)

    def export(self, subjects: torch.Tensor, items: torch.Tensor) -> Dict[str, np.ndarray]:
        q = self.guide.quantiles([0.5])
        theta = q["theta"][0].detach().cpu().numpy()
        a = torch.exp(q["log_a"][0]).detach().cpu().numpy()
        b = q["b"][0].detach().cpu().numpy()
        sigma = float((F.softplus(q["log_sigma"][0]) + 1e-3).detach().cpu().item())
        expected = self.predict_expected_score(theta, {"disc": a, "diff": b})
        return {
            "ability": theta,
            "disc": a,
            "diff": b,
            "sigma": sigma,
            "expected_score": expected,
        }

    def predict_expected_score(self, abilities: np.ndarray, item_params: Dict[str, np.ndarray]) -> np.ndarray:
        theta = np.asarray(abilities, dtype=np.float32)[:, None]
        a = np.asarray(item_params["disc"], dtype=np.float32)[None, :]
        b = np.asarray(item_params["diff"], dtype=np.float32)[None, :]
        return (1.0 / (1.0 + np.exp(-a * (theta - b)))).mean(axis=0)


class ContinuousCategoricalModel(BaseSafetyIrt):
    """Heteroskedastic Gaussian with σ² = μ(1-μ)/a².
    See Balkir et al., arXiv:2601.13885 (https://arxiv.org/html/2601.13885v1).
    """

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def model(self, subjects: torch.Tensor, items: torch.Tensor, obs: torch.Tensor):
        theta = pyro.sample(
            "theta",
            dist.Normal(torch.zeros(self.num_subjects, device=self.device), torch.ones(self.num_subjects, device=self.device)).to_event(1),
        )
        log_a = pyro.sample(
            "log_a",
            dist.Normal(torch.zeros(self.num_items, device=self.device), 0.5 * torch.ones(self.num_items, device=self.device)).to_event(1),
        )
        b = pyro.sample(
            "b",
            dist.Normal(torch.zeros(self.num_items, device=self.device), torch.ones(self.num_items, device=self.device)).to_event(1),
        )

        a = torch.exp(log_a)
        # 1PL mean: μ = sigmoid(θ - b)
        mean = torch.sigmoid(theta[subjects] - b[items])
        mean = mean.clamp(EPS, 1.0 - EPS)
        # Heteroskedastic variance: σ² = μ(1-μ)/a²
        var = mean * (1.0 - mean) / (a[items] ** 2)
        std = torch.sqrt(var.clamp_min(EPS))

        with pyro.plate("responses", obs.shape[0]):
            pyro.sample("obs", dist.Normal(mean, std), obs=obs)

    def export(self, subjects: torch.Tensor, items: torch.Tensor) -> Dict[str, np.ndarray]:
        q = self.guide.quantiles([0.5])
        theta = q["theta"][0].detach().cpu().numpy()
        a = torch.exp(q["log_a"][0]).detach().cpu().numpy()
        b = q["b"][0].detach().cpu().numpy()
        expected = self.predict_expected_score(theta, {"disc": a, "diff": b})
        return {
            "ability": theta,
            "disc": a,
            "diff": b,
            "expected_score": expected,
        }

    def predict_expected_score(self, abilities: np.ndarray, item_params: Dict[str, np.ndarray]) -> np.ndarray:
        theta = np.asarray(abilities, dtype=np.float32)[:, None]
        b = np.asarray(item_params["diff"], dtype=np.float32)[None, :]
        # 1PL mean
        return (1.0 / (1.0 + np.exp(-(theta - b)))).mean(axis=0)


MODEL_REGISTRY = {
    "grm": GradedResponseModel,
    "gpcm": GeneralizedPartialCreditModel,
    "continuous": ContinuousSafetyModel,
    "continuous_cat": ContinuousCategoricalModel,
}


def create_model(model_type: str, *, num_items: int, num_subjects: int, score_values: Iterable[float], device: str = "cpu"):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type={model_type!r}. Choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_type](
        num_items=num_items,
        num_subjects=num_subjects,
        score_values=score_values,
        device=device,
    )
