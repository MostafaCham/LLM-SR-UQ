#!/usr/bin/env python3
"""Loader for LLM-SRBench (Shojaee et al., ICML 2025; arXiv:2504.10415)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .expressions import free_constants, is_numeric, parse_expression

MIRROR_REPO = "pkuHaowei/llm-srbench"

SUBSETS = (
    "lsr_transform",
    "lsr_synth_bio_pop_growth",
    "lsr_synth_chem_react",
    "lsr_synth_matsci",
    "lsr_synth_phys_osc",
)

LSR_SYNTH_SUBSETS = tuple(s for s in SUBSETS if s.startswith("lsr_synth"))

STRUCTURE_ONLY_SUBSETS = ("lsr_synth_chem_react", "lsr_synth_phys_osc")

KNOWN_BAD_GROUND_TRUTH = ("lsr_transform_ii.24.17_2_1",)


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Problem:
    """One benchmark task: data, scientific context, and ground truth."""

    instance_id: str
    subset: str
    description: str
    input_vars: List[str]
    output_var: str
    gt_expression: str

    train: Split
    cal: Split
    test: Split
    ood: Optional[Split] = None

    full_train: Optional[Split] = None

    gt_free_constants: List[str] = field(default_factory=list)
    gt_is_numeric: bool = False

    @property
    def n_inputs(self) -> int:
        return len(self.input_vars)

    @property
    def has_ood(self) -> bool:
        return self.ood is not None

    @property
    def structure_only(self) -> bool:
        """True when the ground truth cannot be evaluated numerically."""
        return not self.gt_is_numeric

    def __repr__(self) -> str:
        ood = f", ood={len(self.ood)}" if self.ood is not None else ""
        return (
            f"Problem({self.instance_id}, vars={self.input_vars}->{self.output_var}, "
            f"train={len(self.train)}, cal={len(self.cal)}, test={len(self.test)}{ood})"
        )


def _as_2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    return arr if arr.ndim == 2 else arr.reshape(len(arr), -1)


def _as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def _carve_calibration(
    X: np.ndarray, y: np.ndarray, cal_frac: float, rng: np.random.Generator
) -> Tuple[Split, Split]:
    n = len(y)
    n_cal = max(1, int(round(n * cal_frac)))
    perm = rng.permutation(n)
    cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
    return Split(X[train_idx], y[train_idx]), Split(X[cal_idx], y[cal_idx])


def _subsample(split: Split, max_rows: Optional[int], rng: np.random.Generator) -> Split:
    """Cap the rows used for fitting."""
    if max_rows is None or len(split) <= max_rows:
        return split
    idx = rng.choice(len(split), size=max_rows, replace=False)
    return Split(split.X[idx], split.y[idx])


def load_subset(
    subset: str,
    cal_frac: float = 0.2,
    max_fit_rows: Optional[int] = 4000,
    seed: int = 0,
    repo: str = MIRROR_REPO,
    limit: Optional[int] = None,
) -> List[Problem]:
    """Load every problem in one benchmark subset."""
    import datasets

    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSETS}")

    records = datasets.load_dataset(repo, subset)["train"]
    problems: List[Problem] = []

    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break

        rng = np.random.default_rng(seed + i)
        input_vars = [str(v) for v in rec["input_vars"]]
        output_var = str(rec["output_vars"][0])

        train_full = Split(_as_2d(rec["train_input"]), _as_1d(rec["train_output"]))
        fit_split = _subsample(train_full, max_fit_rows, rng)
        train, cal = _carve_calibration(fit_split.X, fit_split.y, cal_frac, rng)

        test = Split(_as_2d(rec["test_input"]), _as_1d(rec["test_output"]))
        ood = None
        if rec.get("ood_input") is not None:
            ood = Split(_as_2d(rec["ood_input"]), _as_1d(rec["ood_output"]))

        gt_text = rec["gt_expression"]
        try:
            gt_expr = parse_expression(gt_text, input_vars)
            consts = free_constants(gt_expr, input_vars)
            numeric = is_numeric(gt_expr, input_vars)
        except Exception:
            consts, numeric = [], False

        problems.append(
            Problem(
                instance_id=rec["instance_id"],
                subset=subset,
                description=rec["description"],
                input_vars=input_vars,
                output_var=output_var,
                gt_expression=gt_text,
                train=train,
                cal=cal,
                test=test,
                ood=ood,
                full_train=train_full,
                gt_free_constants=consts,
                gt_is_numeric=numeric,
            )
        )

    return problems


def load_problems(
    subsets: Optional[Sequence[str]] = None,
    **kwargs,
) -> List[Problem]:
    """Load several subsets at once. Defaults to all five."""
    subsets = list(subsets) if subsets is not None else list(SUBSETS)
    out: List[Problem] = []
    for s in subsets:
        out.extend(load_subset(s, **kwargs))
    return out


def dev_split(problems: Sequence[Problem], per_domain: int = 5, seed: int = 0) -> Tuple[List[Problem], List[Problem]]:
    """Partition into a small dev set and a held-out set."""
    rng = np.random.default_rng(seed)
    by_subset: Dict[str, List[Problem]] = {}
    for p in problems:
        by_subset.setdefault(p.subset, []).append(p)

    dev, held = [], []
    for subset in sorted(by_subset):
        group = sorted(by_subset[subset], key=lambda p: p.instance_id)
        idx = rng.permutation(len(group))
        chosen = set(idx[:per_domain].tolist())
        for j, p in enumerate(group):
            (dev if j in chosen else held).append(p)
    return dev, held


def crosscheck_hdf5(problem: Problem, hdf5_path: str) -> Optional[float]:
    """Max absolute disagreement between mirror arrays and the local HDF5."""
    import h5py

    if not os.path.exists(hdf5_path):
        return None

    if problem.subset == "lsr_transform":
        group = "lsr_transform"
        wanted = problem.instance_id[len("lsr_transform_"):]
    else:
        group = f"lsr_synth/{problem.subset[len('lsr_synth_'):]}"
        wanted = problem.instance_id.split("_")[-1]

    train = problem.full_train or problem.train
    mirror = np.hstack([train.y.reshape(-1, 1), train.X])

    with h5py.File(hdf5_path, "r") as fh:
        if group not in fh:
            return None
        match = next((k for k in fh[group] if k.lower() == wanted.lower()), None)
        if match is None:
            return None
        node = fh[group][match]
        key = next((k for k in ("train", "train_data") if k in node), None)
        if key is None:
            return None
        arr = np.asarray(node[key], dtype=np.float64)
        if arr.shape != mirror.shape:
            return float("nan")
        return float(np.abs(arr - mirror).max())
