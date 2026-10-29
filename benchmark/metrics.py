#!/usr/bin/env python3
"""Evaluation metrics for LLM-SRBench, plus uncertainty metrics."""

from __future__ import annotations

import ast
from typing import Dict, Optional, Sequence

import math

import numpy as np
import sympy as sp

from grammar import FUNCTION_NAMES

from .complexity import complexity_report
from .expressions import abstract_constants, lambdify_expression, parse_expression


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(np.asarray(a, dtype=float))
    return mask


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error normalised by the variance of the target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = _finite_mask(y_true, y_pred)
    if not mask.any():
        return float("inf")
    denom = np.var(y_true[mask])
    if denom < 1e-30:
        denom = 1e-30
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2) / denom)


def max_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Worst-case relative error over the split; the quantity Acc_tau thresholds."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("inf")
    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
        return float("inf")
    denom = np.maximum(np.abs(y_true), 1e-12)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        rel = np.abs(y_pred - y_true) / denom
    if not np.all(np.isfinite(rel)):
        return float("inf")
    return float(np.max(rel))


def accuracy_to_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.1) -> float:
    """Acc_tau exactly as LLM-SRBench defines it."""
    return 1.0 if max_relative_error(y_true, y_pred) <= tol else 0.0


def pointwise_within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.1) -> float:
    """Fraction of points inside ``tol``; a diagnostic, not Acc_tau."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return 0.0
    denom = np.maximum(np.abs(y_true), 1e-12)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        rel = np.abs(y_pred - y_true) / denom
    return float(np.mean(np.where(np.isfinite(rel), rel <= tol, False)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - nmse(y_true, y_pred)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unnormalised MSE, on the target's own units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = _finite_mask(y_true, y_pred)
    if not mask.any():
        return float("inf")
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = _finite_mask(y_true, y_pred)
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, as a fraction (0.05 = 5%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = _finite_mask(y_true, y_pred)
    if not mask.any():
        return float("inf")
    denom = np.maximum(np.abs(y_true[mask]), 1e-12)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        rel = np.abs(y_pred[mask] - y_true[mask]) / denom
    rel = rel[np.isfinite(rel)]
    return float(np.mean(rel)) if rel.size else float("inf")


def symbolic_equivalence(
    pred: sp.Expr,
    gt: sp.Expr,
    variables: Sequence[str],
    n_probe: int = 64,
    rtol: float = 1e-6,
    seed: int = 0,
) -> bool:
    """Exact symbolic equality, with a numeric probe as fallback."""
    try:
        if sp.simplify(pred - gt) == 0:
            return True
    except Exception:
        pass

    try:
        f_pred = lambdify_expression(pred, variables)
        f_gt = lambdify_expression(gt, variables)
    except Exception:
        return False

    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.5, 2.0, size=(n_probe, len(variables)))
    try:
        a = np.asarray(f_pred(*[pts[:, i] for i in range(len(variables))]), dtype=float)
        b = np.asarray(f_gt(*[pts[:, i] for i in range(len(variables))]), dtype=float)
    except Exception:
        return False

    a = np.broadcast_to(a, (n_probe,))
    b = np.broadcast_to(b, (n_probe,))
    mask = _finite_mask(a, b)
    if mask.sum() < n_probe // 2:
        return False
    return bool(np.allclose(a[mask], b[mask], rtol=rtol, atol=1e-8))


def structural_match(pred: sp.Expr, gt: sp.Expr, variables: Sequence[str]) -> bool:
    """Equality modulo fitted constants."""
    try:
        a = sp.srepr(sp.simplify(abstract_constants(pred, variables)))
        b = sp.srepr(sp.simplify(abstract_constants(gt, variables)))
        return a == b
    except Exception:
        return False


def symbolic_accuracy(
    pred_expression: str,
    gt_expression: str,
    variables: Sequence[str],
    gt_is_numeric: bool = True,
) -> float:
    """1.0 when the prediction recovers the ground-truth law, else 0.0."""
    try:
        pred = parse_expression(pred_expression, variables)
        gt = parse_expression(gt_expression, variables)
    except Exception:
        return 0.0

    if gt_is_numeric and symbolic_equivalence(pred, gt, variables):
        return 1.0
    return 1.0 if structural_match(pred, gt, variables) else 0.0


def symbolic_accuracy_at_k(
    structures: Sequence[Dict],
    gt_expression: str,
    variables: Sequence[str],
    gt_is_numeric: bool = True,
    ks: Sequence[int] = (1, 3, 5, 10),
    alpha: float = 0.1,
) -> Dict[str, float]:
    """Symbolic accuracy over the posterior buffer, not over its argmax."""
    ks = tuple(sorted(set(int(k) for k in ks)))
    n_families = len(structures)
    out: Dict[str, float] = {f"sa_at_{k}": 0.0 for k in ks}
    out.update({
        "sa_at_all": 0.0,
        "sa_rank": 0.0,
        "sa_mass": 0.0,
        "n_families": float(n_families),
        "credible_set_size": 0.0,
        "sa_in_credible_set": 0.0,
    })
    if n_families == 0:
        return out

    target = 1.0 - float(alpha)
    cumulative = 0.0
    credible_size = n_families
    for i, entry in enumerate(structures, 1):
        cumulative += float(entry.get("mass", 0.0))
        if cumulative >= target:
            credible_size = i
            break
    out["credible_set_size"] = float(credible_size)

    for entry in structures:
        hit = any(
            symbolic_accuracy(expr, gt_expression, variables, gt_is_numeric) == 1.0
            for expr in entry.get("expressions", [])
        )
        if not hit:
            continue
        rank = int(entry["rank"])
        out["sa_rank"] = float(rank)
        out["sa_mass"] = float(entry.get("mass", 0.0))
        out["sa_at_all"] = 1.0
        out["sa_in_credible_set"] = 1.0 if rank <= credible_size else 0.0
        for k in ks:
            out[f"sa_at_{k}"] = 1.0 if rank <= k else 0.0
        break

    return out


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of targets falling inside the predicted interval."""
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    mask = _finite_mask(y_true, lo, hi)
    if not mask.any():
        return 0.0
    inside = (y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])
    return float(np.mean(inside))


def mean_interval_width(lo: np.ndarray, hi: np.ndarray, normalise_by: Optional[np.ndarray] = None) -> float:
    """Average interval width, optionally in units of the target's spread."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    mask = _finite_mask(lo, hi)
    if not mask.any():
        return float("inf")
    width = float(np.mean(hi[mask] - lo[mask]))
    if normalise_by is not None:
        scale = float(np.std(np.asarray(normalise_by, dtype=float)))
        width /= max(scale, 1e-12)
    return width


def sharpness_at_coverage(
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    target_coverage: float = 0.9,
    normalise_by: Optional[np.ndarray] = None,
) -> float:
    """Interval width, but only for intervals that meet the coverage target."""
    if coverage(y_true, lo, hi) < target_coverage:
        return float("inf")
    return mean_interval_width(lo, hi, normalise_by=normalise_by)


def calibration_report(
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float = 0.1,
    normalise_by: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    target = 1.0 - alpha
    cov = coverage(y_true, lo, hi)
    return {
        "coverage": cov,
        "target_coverage": target,
        "coverage_gap": cov - target,
        "mean_width": mean_interval_width(lo, hi, normalise_by=normalise_by),
        "sharpness_at_coverage": sharpness_at_coverage(
            y_true, lo, hi, target_coverage=target, normalise_by=normalise_by
        ),
    }


def evaluate_point(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerances: Sequence[float] = (0.001, 0.01, 0.1),
) -> Dict[str, float]:
    """Point metrics for one problem on one split."""
    out: Dict[str, float] = {
        "nmse": nmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "max_rel_error": max_relative_error(y_true, y_pred),
    }
    for t in tolerances:
        out[f"acc_tau_{t}"] = accuracy_to_tolerance(y_true, y_pred, tol=t)
        out[f"frac_within_{t}"] = pointwise_within_tolerance(y_true, y_pred, tol=t)
    return out


def inclusion_probabilities(
    structures: Sequence[Dict],
    variables: Sequence[str],
    functions: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Posterior mass on structures that contain each variable and operator."""
    functions = list(functions) if functions is not None else sorted(FUNCTION_NAMES)
    total = sum(float(s.get("mass", 0.0)) for s in structures)
    out: Dict[str, float] = {}
    if total <= 0:
        for v in variables:
            out[f"incl_var_{v}"] = 0.0
        for f in functions:
            out[f"incl_fn_{f}"] = 0.0
        return out

    tokens = []
    for s in structures:
        names = set()
        for expr in s.get("expressions") or [s.get("best_expression", "")]:
            names |= set(_identifiers(str(expr)))
        tokens.append((float(s.get("mass", 0.0)), names))

    for v in variables:
        out[f"incl_var_{v}"] = sum(m for m, n in tokens if v in n) / total
    for f in functions:
        mass = sum(m for m, n in tokens if f in n)
        if mass > 0:
            out[f"incl_fn_{f}"] = mass / total
    return out


def _identifiers(expr: str) -> Sequence[str]:
    """Every bare name in an expression, function names included."""
    try:
        tree = ast.parse(expr.strip(), mode="eval").body
    except SyntaxError:
        return []
    return [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]


def posterior_structure_report(
    structures: Sequence[Dict],
    variables: Sequence[str],
    n_data: Optional[int] = None,
    complexity_penalty: float = 0.05,
) -> Dict[str, float]:
    """Simplicity and inclusion metrics summarised over the whole posterior."""
    if not structures:
        return {}

    top = structures[0]
    top_expr = top.get("best_expression", "")
    out: Dict[str, float] = {
        f"map_{k}": v for k, v in
        complexity_report(top_expr, variables, n_data, complexity_penalty).items()
    }

    total = sum(float(s.get("mass", 0.0)) for s in structures)
    if total > 0:
        for key in ("n_nodes", "n_operators", "n_params", "depth", "mdl_total_bits"):
            acc = 0.0
            for s in structures:
                rep = complexity_report(s.get("best_expression", ""), variables,
                                        n_data, complexity_penalty)
                val = rep[key]
                if not np.isfinite(val):
                    continue
                acc += float(s.get("mass", 0.0)) * val
            out[f"weighted_{key}"] = acc / total

    out.update(inclusion_probabilities(structures, variables))
    return out


ATTAIN_EPS = 0.02


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval for k successes in n trials (95% by default)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def attains(coverage: float, alpha: float = 0.1, eps: float = ATTAIN_EPS) -> bool:
    return bool(np.isfinite(coverage) and coverage >= 1.0 - alpha - eps)


def normalized_width(lo: np.ndarray, hi: np.ndarray, y_train: np.ndarray) -> float:
    """Mean interval width in units of the training target's standard deviation."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    mask = _finite_mask(lo, hi)
    if not mask.any():
        return float("inf")
    return float(np.mean(hi[mask] - lo[mask])) / max(float(np.std(np.asarray(y_train, dtype=float))), 1e-300)


def interval_score(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.1,
                   y_train: Optional[np.ndarray] = None) -> float:
    """Mean interval score (Gneiting & Raftery 2007), normalised by sd(y_train) when given."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y_true, lo, hi))
    mask = _finite_mask(y, lo, hi)
    if not mask.any():
        return float("inf")
    y, lo, hi = y[mask], lo[mask], hi[mask]
    score = (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)
    scale = max(float(np.std(np.asarray(y_train, dtype=float))), 1e-300) if y_train is not None else 1.0
    return float(np.mean(score)) / scale


def uncertainty_pair(ood_coverages, ood_widths_norm, alpha: float = 0.1, eps: float = ATTAIN_EPS) -> Dict[str, float]:
    """A_alpha with its count and Wilson interval, and S_alpha over the attaining problems."""
    cov = np.asarray(ood_coverages, dtype=float)
    width = np.asarray(ood_widths_norm, dtype=float)
    counted = np.isfinite(cov)
    attained = counted & (cov >= 1.0 - alpha - eps)
    n, k = int(counted.sum()), int(attained.sum())
    w_lo, w_hi = wilson_interval(k, n)
    widths = width[attained & np.isfinite(width)]
    return {
        "attainment_rate": k / n if n else float("nan"),
        "attained": k,
        "n": n,
        "attainment_wilson_lo": w_lo,
        "attainment_wilson_hi": w_hi,
        "sharpness_at_coverage": float(np.median(widths)) if widths.size else float("nan"),
        "sharpness_n": int(widths.size),
    }

