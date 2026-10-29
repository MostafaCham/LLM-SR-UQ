#!/usr/bin/env python3
"""MAP parameter fitting and evidence estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import time

import numpy as np
from scipy.optimize import least_squares

from equation_program import EquationProgram

_HUGE = 1e6


@dataclass
class FitResult:
    """MAP fit, evidence estimates, and the diagnostics the critic consumes."""

    theta_map: Dict[str, float]
    sigma2_hat: float
    cov_theta: np.ndarray
    log_evidence_bic: float
    log_evidence_laplace: Optional[float]
    train_mse: float

    param_sd: Dict[str, float] = field(default_factory=dict)
    unidentified_parameters: List[str] = field(default_factory=list)
    insignificant_parameters: List[str] = field(default_factory=list)
    protection: Dict[str, float] = field(default_factory=dict)
    n_restarts: int = 0
    n_restarts_succeeded: int = 0
    converged: bool = True

    @property
    def n_params(self) -> int:
        return len(self.theta_map)

    @property
    def weak_parameters(self) -> List[str]:
        """Parameters whose attached term is doing no work."""
        return sorted(set(self.unidentified_parameters) | set(self.insignificant_parameters))


SIGMA2_FLOOR_REL = 1e-12


def _sigma2(rss: float, y: np.ndarray) -> float:
    scale = float(np.var(y)) if len(y) else 1.0
    return max(rss / max(len(y), 1), SIGMA2_FLOOR_REL * scale, 1e-300)


def _residuals(prog: EquationProgram, X: np.ndarray, y: np.ndarray, names: Sequence[str]):
    def f(vec: np.ndarray) -> np.ndarray:
        theta = {p: float(v) for p, v in zip(names, vec)}
        try:
            pred = prog.f_np(X, theta)
        except Exception:
            return np.full_like(y, _HUGE)
        r = y - np.asarray(pred, dtype=float)
        return np.clip(np.where(np.isfinite(r), r, _HUGE), -_HUGE, _HUGE)

    return f


def fit_program(
    prog: EquationProgram,
    X: np.ndarray,
    y: np.ndarray,
    n_restarts: int = 8,
    max_nfev: int = 400,
    seed: int = 0,
    unidentified_prior_frac: float = 0.5,
    time_budget: Optional[float] = None,
) -> FitResult:
    """Fit ``prog`` to ``(X, y)`` by multi-start least squares."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    names = list(prog.param_names)
    k = len(names)

    if k == 0:
        pred = np.asarray(prog.f_np(X, {}), dtype=float)
        resid = y - pred
        rss = float(np.sum(np.where(np.isfinite(resid), resid, _HUGE) ** 2))
        return _assemble(prog, X, y, {}, rss, n, 0, np.zeros((0, 0)), {}, 1, 1, True)

    residual_fn = _residuals(prog, X, y, names)
    rng = np.random.default_rng(seed)

    best_vec: Optional[np.ndarray] = None
    best_rss = np.inf
    succeeded = 0

    prog.protection_np.enabled = False
    try:
        best_vec, best_rss, succeeded, attempted = _multistart(
            residual_fn, k, n_restarts, max_nfev, rng, time_budget
        )
    finally:
        prog.protection_np.enabled = True

    if best_vec is None:
        raise RuntimeError(f"all {n_restarts} restarts failed for {prog.expr!r}")

    theta = {p: float(v) for p, v in zip(names, best_vec)}
    cov, sds, scales = _laplace_covariance(prog, X, theta, names, _sigma2(best_rss, y))

    unidentified = [p for i, p in enumerate(names)
                    if sds.get(p, np.inf) >= unidentified_prior_frac * scales[i]]
    insignificant = [
        p for p in names
        if p not in unidentified and abs(theta[p]) < sds.get(p, 0.0)
    ]

    return _assemble(
        prog, X, y, theta, best_rss, n, k, cov,
        {p: sds.get(p, float("nan")) for p in names},
        attempted, succeeded, True, unidentified, insignificant,
    )


def _multistart(residual_fn, k, n_restarts, max_nfev, rng, time_budget=None):
    """Best of up to ``n_restarts`` local solves; also returns how many were attempted."""
    best_vec, best_rss, succeeded, attempted = None, np.inf, 0, 0
    start = time.perf_counter()

    for attempt in range(n_restarts):
        if attempt > 0 and time_budget is not None and time.perf_counter() - start > time_budget:
            break
        attempted += 1
        if attempt == 0:
            x0 = np.zeros(k)
        else:
            scale = 10.0 ** rng.uniform(-1.0, 1.0)
            x0 = rng.normal(scale=scale, size=k)

        try:
            sol = least_squares(
                residual_fn, x0, method="trf", max_nfev=max_nfev,
                xtol=1e-10, ftol=1e-10, gtol=1e-10,
            )
        except Exception:
            continue

        rss = float(np.sum(sol.fun ** 2))
        if np.isfinite(rss):
            succeeded += 1
            if rss < best_rss:
                best_rss, best_vec = rss, sol.x

    return best_vec, best_rss, succeeded, attempted


PRIOR_REL = 2.0


def _prior_scales(theta: Dict[str, float], names: Sequence[str], prior_sigma: float) -> np.ndarray:
    return np.array([max(prior_sigma, PRIOR_REL * abs(theta[p])) for p in names], dtype=float)


def _fisher(prog, X, theta, names, sigma2):
    """Data information J^T J / sigma^2, from a finite-difference Jacobian."""
    k = len(names)
    base = np.asarray(prog.f_np(X, theta), dtype=float)
    J = np.zeros((len(base), k), dtype=float)

    for j, p in enumerate(names):
        step = 1e-6 * max(abs(theta[p]), 1.0)
        bumped = dict(theta)
        bumped[p] = theta[p] + step
        try:
            J[:, j] = (np.asarray(prog.f_np(X, bumped), dtype=float) - base) / step
        except Exception:
            J[:, j] = 0.0

    J = np.where(np.isfinite(J), J, 0.0)
    return (J.T @ J) / sigma2


def _laplace_covariance(prog, X, theta, names, sigma2):
    """Fisher information plus a per-coefficient prior -> covariance."""
    k = len(names)
    scales = _prior_scales(theta, names, prog.prior_sigma)
    H = _fisher(prog, X, theta, names, sigma2) + np.diag(1.0 / scales ** 2)

    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H + 1e-6 * np.eye(k))

    sds = {p: float(np.sqrt(max(cov[i, i], 0.0))) for i, p in enumerate(names)}
    return cov, sds, scales


def _assemble(prog, X, y, theta, rss, n, k, cov, sds, n_restarts, succeeded,
              converged, unidentified=None, insignificant=None):
    mse = rss / max(n, 1)
    sigma2 = _sigma2(rss, y)

    bic = n * np.log(sigma2) + k * np.log(max(n, 2))
    log_ev_bic = -0.5 * bic

    log_ev_laplace = None
    if k > 0 and np.all(np.isfinite(cov)):
        sign, logdet = np.linalg.slogdet(np.linalg.pinv(cov))
        if sign > 0:
            loglik = -0.5 * n * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
            log_ev_laplace = float(loglik + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet)

    try:
        prog.f_np(X, theta)
        protection = prog.protection_summary()
    except Exception:
        protection = {}

    return FitResult(
        theta_map=theta,
        sigma2_hat=float(sigma2),
        cov_theta=cov,
        log_evidence_bic=float(log_ev_bic),
        log_evidence_laplace=log_ev_laplace,
        train_mse=float(mse),
        param_sd=sds,
        unidentified_parameters=list(unidentified or []),
        insignificant_parameters=list(insignificant or []),
        protection=protection,
        n_restarts=n_restarts,
        n_restarts_succeeded=succeeded,
        converged=converged,
    )
