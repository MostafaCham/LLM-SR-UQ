#!/usr/bin/env python3
"""Diagnostics engine and mechanical feedback for rung A1."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.metrics import inclusion_probabilities
from canonical import family_signature

TOP_K = 3
STAGNATION_K = 10
REPEAT_MIN = 3
MAX_REPEATED = 5
PROTECTION_MIN = 0.05
RESIDUAL_RHO_MIN = 0.2
RESIDUAL_RANGE_MIN = 0.5
RESIDUAL_BINS = 5
RESIDUAL_MAX_ROWS = 2000
MAX_CHARS = 5000


def _tree(expr: str) -> Optional[ast.AST]:
    try:
        return ast.parse(str(expr).strip(), mode="eval").body
    except SyntaxError:
        return None


def _names(node: ast.AST) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _additive_terms(expr: str) -> List[ast.AST]:
    """Top-level summands of an expression, so a finding can name a term."""
    tree = _tree(expr)
    if tree is None:
        return []
    terms: List[ast.AST] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            walk(node.left)
            walk(node.right)
        else:
            terms.append(node)

    walk(tree)
    return terms


def _rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)


def weak_coefficient_findings(structures: Sequence[Dict], top_k: int = TOP_K) -> List[Dict]:
    """Coefficients the data does not determine, and the term each sits in."""
    out: List[Dict] = []
    for rank, s in enumerate(structures[:top_k], 1):
        weak = list(s.get("weak_parameters") or [])
        if not weak:
            continue
        unidentified = set(s.get("unidentified_parameters") or [])
        terms = _additive_terms(s.get("best_expression", ""))
        for p in weak:
            out.append({
                "rank": rank,
                "expression": s.get("best_expression", ""),
                "coefficient": p,
                "kind": "unidentified" if p in unidentified else "insignificant",
                "terms": [ast.unparse(t) for t in terms if p in _names(t)],
            })
    return out


def variable_findings(structures: Sequence[Dict], var_names: Sequence[str],
                      top_k: int = TOP_K) -> Dict:
    """Declared inputs the best structures ignore, and posterior inclusion."""
    inc = inclusion_probabilities(structures, var_names)
    used_top: set = set()
    for s in structures[:top_k]:
        tree = _tree(s.get("best_expression", ""))
        if tree is not None:
            used_top |= _names(tree) & set(var_names)
    return {
        "inclusion": {v: float(inc.get(f"incl_var_{v}", 0.0)) for v in var_names},
        "unused_in_top": [v for v in var_names if v not in used_top],
    }


def residual_findings(pipe, var_names: Sequence[str]) -> Dict:
    """Systematic structure left in the best model's residual, per input."""
    best = pipe.buffer.best()
    X = getattr(pipe, "_Xtr", None)
    y = getattr(pipe, "_ytr", None)
    if best is None or X is None or y is None:
        return {}

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) > RESIDUAL_MAX_ROWS:
        step = int(np.ceil(len(y) / RESIDUAL_MAX_ROWS))
        X, y = X[::step], y[::step]

    try:
        with np.errstate(all="ignore"):
            r = y - np.asarray(best.program.predict(X, best.fit.theta_map), dtype=float)
    except Exception:
        return {}

    mask = np.isfinite(r) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 20:
        return {}
    X, r, y = X[mask], r[mask], y[mask]

    sd = float(np.std(r))
    out: Dict = {
        "expression": best.program.expr,
        "residual_rel_sd": sd / max(float(np.std(y)), 1e-12),
        "patterns": [],
    }
    if sd < 1e-12:
        return out

    rr = _rank(r)
    for j, v in enumerate(var_names):
        if j >= X.shape[1]:
            break
        x = X[:, j]
        if np.std(x) < 1e-12:
            continue
        rho = float(np.corrcoef(_rank(x), rr)[0, 1])
        if not np.isfinite(rho):
            rho = 0.0

        edges = np.quantile(x, np.linspace(0.0, 1.0, RESIDUAL_BINS + 1))
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, RESIDUAL_BINS - 1)
        means = [float(r[idx == b].mean()) for b in range(RESIDUAL_BINS) if np.any(idx == b)]
        if len(means) < 3:
            continue
        span = (max(means) - min(means)) / sd
        steps = np.diff(means)
        monotone = bool(np.all(steps > 0) or np.all(steps < 0))

        if abs(rho) >= RESIDUAL_RHO_MIN:
            shape = "rises" if rho > 0 else "falls"
        elif span >= RESIDUAL_RANGE_MIN and not monotone:
            shape = "non-monotone"
        else:
            continue
        out["patterns"].append({
            "variable": v, "shape": shape, "rank_corr": rho, "bin_range_sd": span,
        })
    return out


def protection_findings(structures: Sequence[Dict], top_k: int = TOP_K) -> List[Dict]:
    """Best structures whose protected operators clamp on a real share of rows."""
    out: List[Dict] = []
    for rank, s in enumerate(structures[:top_k], 1):
        for op, rate in sorted((s.get("protection") or {}).items()):
            if rate >= PROTECTION_MIN:
                out.append({"rank": rank, "expression": s.get("best_expression", ""),
                            "operator": op, "rate": float(rate)})
    return out


def stagnation(history: Sequence, k: int = STAGNATION_K) -> Dict:
    """Rounds since best log-evidence last strictly improved."""
    best, last = -np.inf, None
    for i, rec in enumerate(history):
        v = rec.best_log_evidence
        if v is not None and v > best + 1e-9:
            best, last = v, i
    since = len(history) - 1 - last if last is not None else len(history)
    return {"iterations_since_improvement": int(since), "stagnant": bool(since >= k)}


def compute_diagnostics(pipe, history: Sequence, var_names: Sequence[str],
                        family_counts: Dict[str, int], exemplars: Dict[str, str],
                        top_k: int = TOP_K, stagnation_k: int = STAGNATION_K) -> Dict:
    """Every finding for the next round, as one JSON-serialisable dict."""
    structures = pipe.posterior_structures()
    top_families = {s["family"] for s in structures[:top_k]}
    ytr = getattr(pipe, "_ytr", None)
    y_var = float(np.var(ytr)) if ytr is not None and len(ytr) else 1.0
    last = history[-1]

    repeated = sorted(
        ((f, c) for f, c in family_counts.items()
         if c >= REPEAT_MIN and f not in top_families),
        key=lambda fc: (-fc[1], exemplars.get(fc[0], "")),
    )

    return {
        "iteration": len(history),
        "candidates_fitted": int(sum(len(r.accepted) for r in history)),
        "last_batch": {
            "proposed": len(last.candidates),
            "fitted": len(last.accepted),
            "entered_buffer": sum(1 for c in last.accepted if c.entered_buffer),
            "rejections": [{"expression": c.expression, "reason": c.reason[:160]}
                           for c in last.rejected[:2]],
        },
        "top_structures": [
            {"rank": i, "expression": s["best_expression"],
             "train_mse": float(s["train_mse"]),
             "train_nmse": float(s["train_mse"]) / max(y_var, 1e-300),
             "n_params": int(s["n_params"]),
             "mass": float(s["mass"])}
            for i, s in enumerate(structures[:top_k], 1)
        ],
        "weak_coefficients": weak_coefficient_findings(structures, top_k),
        "variables": variable_findings(structures, var_names, top_k),
        "residual": residual_findings(pipe, var_names),
        "protection": protection_findings(structures, top_k),
        "repeated_families": [{"expression": exemplars.get(f, f), "times": int(c)}
                              for f, c in repeated[:MAX_REPEATED]],
        "stagnation": stagnation(history, stagnation_k),
    }


def render_a1(diag: Dict, max_chars: int = MAX_CHARS) -> str:
    """The A1 prompt block: fixed wording, deterministic, hard-capped."""
    header = (f"Feedback on the candidates fitted so far "
              f"({diag['candidates_fitted']} fitted over {diag['iteration']} rounds):")
    sections: List[Tuple[int, int, str]] = []

    st = diag["stagnation"]
    if st["stagnant"]:
        sections.append((0, 90,
            f"No improvement in the best structure for {st['iterations_since_improvement']} "
            "rounds. Propose structurally different forms, not small variations of the "
            "structures above."))

    if diag["top_structures"]:
        lines = ["Best structures so far (NMSE on the training data; lower is better):"]
        for s in diag["top_structures"]:
            lines.append(f"  {s['rank']}. {s['expression']}   "
                         f"NMSE={s['train_nmse']:.3g}  coefficients={s['n_params']}")
        sections.append((1, 10, "\n".join(lines)))

    if diag["weak_coefficients"]:
        lines = ["Coefficients the data does not determine; the term they sit in is doing "
                 "no work, so remove or rewrite it:"]
        for w in diag["weak_coefficients"]:
            term = w["terms"][0] if w["terms"] else w["coefficient"]
            why = ("not constrained by the data" if w["kind"] == "unidentified"
                   else "indistinguishable from zero")
            lines.append(f"  - structure {w['rank']}: {w['coefficient']} in `{term}` ({why})")
        sections.append((2, 20, "\n".join(lines)))

    var = diag["variables"]
    if var.get("unused_in_top"):
        share = ", ".join(f"{v} {p:.2f}" for v, p in var["inclusion"].items())
        sections.append((3, 30,
            f"Declared inputs not used by any of the best structures: "
            f"{', '.join(var['unused_in_top'])}.\n"
            f"  Share of posterior mass on structures that use each input: {share}."))

    res = diag["residual"]
    if res.get("patterns"):
        lines = [f"Residual of structure 1 (residual scale {res['residual_rel_sd']:.2g} of "
                 "the target's):"]
        for p in res["patterns"]:
            v = p["variable"]
            if p["shape"] == "non-monotone":
                lines.append(f"  - varies non-monotonically with {v}: curvature in {v} is missing")
            else:
                lines.append(f"  - {p['shape']} with {v} (rank correlation {p['rank_corr']:+.2f}): "
                             f"the dependence on {v} is missing or misshaped")
        sections.append((4, 40, "\n".join(lines)))

    if diag["protection"]:
        lines = ["Forms that leave their domain (a protected operator had to clamp):"]
        for pr in diag["protection"]:
            lines.append(f"  - structure {pr['rank']}: {pr['operator']} clamps on "
                         f"{pr['rate']:.0%} of rows")
        sections.append((5, 50, "\n".join(lines)))

    if diag["repeated_families"]:
        lines = ["Tried repeatedly without reaching the best structures; do not propose "
                 "these families again:"]
        for f in diag["repeated_families"]:
            lines.append(f"  - {f['expression']}   ({f['times']} times)")
        sections.append((6, 60, "\n".join(lines)))

    rej = diag["last_batch"]["rejections"]
    if rej:
        lines = ["Rejected last round:"]
        for r in rej:
            lines.append(f"  - {r['expression']}: {r['reason']}")
        sections.append((7, 70, "\n".join(lines)))

    chosen: List[Tuple[int, str]] = []
    used = len(header)
    for _, order, text in sorted(sections):
        cost = len(text) + 2
        if used + cost <= max_chars:
            chosen.append((order, text))
            used += cost
            continue
        lines = text.split("\n")
        kept, kept_cost = [lines[0]], len(lines[0]) + 2
        for line in lines[1:]:
            if used + kept_cost + len(line) + 1 > max_chars:
                break
            kept.append(line)
            kept_cost += len(line) + 1
        if len(kept) > 1:
            chosen.append((order, "\n".join(kept)))
            used += kept_cost

    if not chosen:
        return ""
    text = "\n\n".join([header] + [t for _, t in sorted(chosen)])
    return text[:max_chars]


class A1Feedback:
    """The ``feedback_fn`` for rung A1."""

    def __init__(self, var_names: Sequence[str], top_k: int = TOP_K,
                 stagnation_k: int = STAGNATION_K, max_chars: int = MAX_CHARS):
        self.var_names = list(var_names)
        self.top_k = top_k
        self.stagnation_k = stagnation_k
        self.max_chars = max_chars
        self._seen = 0
        self._family_counts: Dict[str, int] = {}
        self._exemplars: Dict[str, str] = {}

    def _update_families(self, history: Sequence) -> None:
        for rec in history[self._seen:]:
            for c in rec.candidates:
                if not c.accepted:
                    continue
                try:
                    fam = family_signature(c.expression, self.var_names)
                except Exception:
                    continue
                self._family_counts[fam] = self._family_counts.get(fam, 0) + 1
                self._exemplars.setdefault(fam, c.expression)
        self._seen = len(history)

    def __call__(self, pipe, history: Sequence) -> Tuple[str, Optional[Dict]]:
        if not history:
            return "", None
        self._update_families(history)
        diag = compute_diagnostics(pipe, history, self.var_names,
                                   self._family_counts, self._exemplars,
                                   self.top_k, self.stagnation_k)
        return render_a1(diag, self.max_chars), diag
