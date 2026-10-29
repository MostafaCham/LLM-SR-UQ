#!/usr/bin/env python3
"""Bayesian model averaging and conformal prediction."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from posterior_buffer import PosteriorBuffer

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None


class BMAPredictor:
    """Posterior-weighted average over the buffer, with parameter uncertainty."""

    WEIGHTINGS = ("tempered", "hybrid")

    def __init__(
        self,
        buffer: PosteriorBuffer,
        n_param_samples: int = 50,
        seed: int = 0,
        draw_tolerance: float = 50.0,
        weighting: str = "tempered",
    ):
        """``weighting`` chooses which weights make the mean."""
        if weighting not in self.WEIGHTINGS:
            raise ValueError(f"unknown weighting {weighting!r}; expected one of {self.WEIGHTINGS}")
        self.buffer = buffer
        self.n_param_samples = n_param_samples
        self.seed = seed
        self.draw_tolerance = draw_tolerance
        self.weighting = weighting

    @staticmethod
    def _safe_cholesky(A: np.ndarray, k: int) -> Optional[np.ndarray]:
        A = np.asarray(A, dtype=float)
        if A.shape != (k, k) or not np.all(np.isfinite(A)):
            return None
        A = 0.5 * (A + A.T)
        jitter = 1e-9
        for _ in range(10):
            try:
                return np.linalg.cholesky(A + jitter * np.eye(k))
            except np.linalg.LinAlgError:
                jitter *= 10.0
        try:
            w, V = np.linalg.eigh(A)
            return np.linalg.cholesky((V * np.maximum(w, 1e-12)) @ V.T + 1e-9 * np.eye(k))
        except Exception:
            return None

    def _model_mean_var(self, prog, fit, X: np.ndarray, rng):
        """MAP prediction for this model, with its variance split in two."""
        try:
            mu = np.asarray(prog.f_np(X, fit.theta_map), dtype=float)
        except Exception:
            return None
        if not np.all(np.isfinite(mu)):
            return None

        noise = max(fit.sigma2_hat, 0.0)
        k = prog.n_params()
        zeros = np.zeros(len(mu))
        if k == 0:
            return mu, zeros, np.full(len(mu), noise)

        L = self._safe_cholesky(fit.cov_theta, k)
        if L is None:
            return mu, zeros, np.full(len(mu), noise)

        band = self.draw_tolerance * max(float(np.std(mu)), np.sqrt(noise), 1e-12)
        theta0 = np.array([fit.theta_map[p] for p in prog.param_names], dtype=float)

        acc = np.zeros(len(mu))
        kept = 0
        for _ in range(self.n_param_samples):
            theta = dict(zip(prog.param_names, theta0 + L @ rng.normal(size=k)))
            try:
                yhat = np.asarray(prog.f_np(X, theta), dtype=float)
            except Exception:
                continue
            dev = yhat - mu
            if not np.all(np.isfinite(dev)) or np.abs(dev).max() > band:
                continue
            acc += dev ** 2
            kept += 1

        param_var = acc / kept if kept else np.zeros(len(mu))
        return mu, param_var, np.full(len(mu), noise)

    def predict_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """BMA mean with its predictive variance split three ways."""
        n = X.shape[0]
        flat = {
            "mean": np.zeros(n),
            "var": np.ones(n),
            "var_noise": np.ones(n),
            "var_param": np.zeros(n),
            "var_struct": np.zeros(n),
            "n_models": 0.0,
        }
        if not self.buffer.items:
            return flat

        rng = np.random.default_rng(self.seed)

        contributions = []
        log_weights = []
        total_w = 0.0
        for item in self.buffer.items:
            prog, fit, w = item.program, item.fit, item.weight
            if w <= 0 or not np.isfinite(fit.sigma2_hat) or fit.sigma2_hat > 1e20:
                continue
            got = self._model_mean_var(prog, fit, X, rng)
            if got is None:
                continue
            contributions.append((w, *got))
            log_weights.append(item.log_weight)
            total_w += w

        if not contributions:
            return flat

        if self.weighting == "hybrid":
            lw = np.array(log_weights, dtype=float)
            lw = np.where(np.isfinite(lw), lw, -np.inf)
            lw = lw - lw.max()
            mean_shares = np.exp(lw) / np.exp(lw).sum()
        else:
            mean_shares = np.array([w / total_w for w, _, _, _ in contributions])

        means = np.zeros(n)
        for s, (_, mu, _, _) in zip(mean_shares, contributions):
            means += s * mu

        var_param = np.zeros(n)
        var_noise = np.zeros(n)
        var_struct = np.zeros(n)
        for w, mu, pv, nv in contributions:
            share = w / total_w
            var_param += share * pv
            var_noise += share * nv
            var_struct += share * (mu - means) ** 2

        def clean(a, fill):
            return np.nan_to_num(np.maximum(a, 0.0), nan=fill, posinf=1e12, neginf=1e12)

        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        var_param, var_noise, var_struct = (clean(var_param, 0.0),
                                            clean(var_noise, 1.0),
                                            clean(var_struct, 0.0))
        return {
            "mean": means,
            "var": clean(var_param + var_noise + var_struct, 1.0),
            "var_noise": var_noise,
            "var_param": var_param,
            "var_struct": var_struct,
            "n_models": float(len(contributions)),
        }

    def predict_mean_var(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out = self.predict_components(X)
        return out["mean"], out["var"]


class SplitConformal:
    """Distribution-free intervals from calibration residuals."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q: Optional[float] = None

    def fit(self, y_cal: np.ndarray, yhat_cal: np.ndarray) -> None:
        resid = np.abs(np.asarray(y_cal, float) - np.asarray(yhat_cal, float))
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            self.q = float("inf")
            return
        n = len(resid)
        k = int(math.ceil((n + 1) * (1 - self.alpha)))
        k = max(1, min(k, n))
        self.q = float(np.partition(resid, k - 1)[k - 1])

    def predict_interval(self, yhat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.q is None:
            raise RuntimeError("call fit() before predict_interval()")
        return yhat - self.q, yhat + self.q


class WeightedConformal(SplitConformal):
    """Covariate-shift-aware intervals via density-ratio weighting."""

    def fit_weighted(self, y_cal: np.ndarray, yhat_cal: np.ndarray, w: np.ndarray) -> None:
        from utils import weighted_quantile

        resid = np.abs(np.asarray(y_cal, float) - np.asarray(yhat_cal, float))
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(resid) & np.isfinite(w) & (w > 0)
        if not mask.any():
            self.q = float("inf")
            return
        self.q = weighted_quantile(resid[mask], w[mask], 1 - self.alpha)


class NormalizedConformal(SplitConformal):
    """Split conformal on residuals scaled by the posterior's own spread."""

    COMPONENTS = {"var_struct": 1.0, "var_noise": 1.0}

    @classmethod
    def sigma(cls, components: Dict[str, np.ndarray]) -> np.ndarray:
        total = sum(w * np.asarray(components[c], dtype=float) for c, w in cls.COMPONENTS.items())
        return np.sqrt(np.maximum(total, 1e-300))

    def fit_normalized(self, y_cal: np.ndarray, yhat_cal: np.ndarray, sigma_cal: np.ndarray) -> None:
        scores = np.abs(np.asarray(y_cal, float) - np.asarray(yhat_cal, float)) / np.asarray(sigma_cal, float)
        self.fit(scores, np.zeros_like(scores))

    def predict_interval_scaled(self, yhat: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.q is None:
            raise RuntimeError("call fit_normalized() before predict_interval_scaled()")
        return yhat - self.q * sigma, yhat + self.q * sigma


class StructuralConformal(NormalizedConformal):
    """Normalized conformal with sigma from disagreement between structures alone."""

    COMPONENTS = {"var_struct": 1.0, "var_noise": 1e-6}


def estimate_density_ratio_logreg(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """Ratio p_target(x)/p_source(x) via a probabilistic classifier."""
    if LogisticRegression is None:
        raise RuntimeError("scikit-learn is required for density-ratio weighting")

    X = np.vstack([X_source, X_target])
    y = np.hstack([np.zeros(len(X_source)), np.ones(len(X_target))])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)

    p1 = clf.predict_proba(X_source)[:, 1]
    p0 = 1.0 - p1
    pi0 = len(X_source) / (len(X_source) + len(X_target))
    pi1 = 1.0 - pi0
    ratio = (p1 / np.maximum(p0, 1e-8)) * (pi0 / np.maximum(pi1, 1e-8))
    return np.clip(ratio, 1e-3, 1e3)
