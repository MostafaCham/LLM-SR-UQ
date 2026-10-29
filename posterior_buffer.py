#!/usr/bin/env python3
"""Evidence-weighted buffer of candidate models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from equation_program import EquationProgram
from parameter_inference import FitResult


@dataclass
class BufferItem:
    program: EquationProgram
    fit: FitResult
    weight: float
    log_weight: float
    signature: str
    family: str
    source_note: str = ""


class PosteriorBuffer:
    def __init__(
        self,
        max_items: int = 24,
        evidence_mode: str = "bic",
        temperature: Optional[float] = None,
        complexity_penalty: float = 0.05,
    ):
        """``temperature`` divides log-weights before the softmax."""
        self.items: List[BufferItem] = []
        self.max_items = max_items
        self.evidence_mode = evidence_mode
        self.temperature = float(temperature) if temperature is not None else 1.0
        self.complexity_penalty = complexity_penalty

        self.seen_signatures: Dict[str, int] = {}
        self.seen_families: Dict[str, int] = {}

    def choose_log_evidence(self, fit: FitResult) -> float:
        if self.evidence_mode == "laplace" and fit.log_evidence_laplace is not None:
            return fit.log_evidence_laplace
        return fit.log_evidence_bic

    def add(
        self,
        program: EquationProgram,
        fit: FitResult,
        pM: float = 1.0,
        note: str = "",
    ) -> bool:
        """Insert a fitted candidate. Returns True if it entered the buffer."""
        signature = program.structure_signature()
        family = program.family_signature()
        self.seen_signatures[signature] = self.seen_signatures.get(signature, 0) + 1
        self.seen_families[family] = self.seen_families.get(family, 0) + 1

        log_ev = self.choose_log_evidence(fit)
        log_prior = -self.complexity_penalty * program.n_params() + np.log(max(pM, 1e-12))
        log_w = float(log_ev + log_prior)

        for item in self.items:
            if item.signature == signature:
                if log_w > item.log_weight:
                    item.program, item.fit = program, fit
                    item.log_weight, item.source_note = log_w, note
                    self._prune_and_normalise()
                    return True
                return False

        self.items.append(
            BufferItem(program, fit, 0.0, log_w, signature, family, note)
        )
        self._prune_and_normalise()
        return any(it.signature == signature for it in self.items)

    def _prune_and_normalise(self) -> None:
        self.items.sort(key=lambda it: it.log_weight, reverse=True)
        if self.max_items is not None:
            self.items = self.items[: self.max_items]

        if not self.items:
            return

        T = max(self.temperature, 1e-12)
        logs = np.array([it.log_weight for it in self.items], dtype=float) / T
        logs -= logs.max()
        probs = np.exp(logs)
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(len(probs), 1.0 / len(probs))
        for item, p in zip(self.items, probs):
            item.weight = float(p)

    @property
    def weights(self) -> np.ndarray:
        return np.array([it.weight for it in self.items], dtype=float)

    def effective_model_count(self) -> float:
        """Kish effective sample size of the weight distribution."""
        w = self.weights
        if w.size == 0:
            return 0.0
        denom = float(np.sum(w ** 2))
        return float(1.0 / denom) if denom > 0 else 0.0

    def _effective_at(self, temperature: float) -> float:
        logs = np.array([it.log_weight for it in self.items], dtype=float)
        logs = logs / max(temperature, 1e-12)
        logs -= logs.max()
        p = np.exp(logs)
        total = p.sum()
        if total <= 0:
            return float(len(logs))
        p /= total
        return float(1.0 / np.sum(p ** 2))

    def calibrate_temperature(
        self,
        target_fraction: float = 0.35,
        lo: float = 1e-3,
        hi: float = 1e9,
        max_iter: int = 60,
    ) -> float:
        """Choose the tempering constant that hits a target effective model count."""
        n = len(self.items)
        if n <= 1:
            return self.temperature

        target = float(np.clip(target_fraction * n, 1.0, n))
        if self._effective_at(hi) <= target:
            best = hi
        elif self._effective_at(lo) >= target:
            best = lo
        else:
            for _ in range(max_iter):
                mid = float(np.sqrt(lo * hi))
                if self._effective_at(mid) < target:
                    lo = mid
                else:
                    hi = mid
            best = float(np.sqrt(lo * hi))

        self.temperature = best
        self._prune_and_normalise()
        return best

    def diagnostics(self) -> Dict[str, float]:
        w = self.weights
        return {
            "n_items": len(self.items),
            "n_distinct_structures": len({it.signature for it in self.items}),
            "n_distinct_families": len({it.family for it in self.items}),
            "effective_models": self.effective_model_count(),
            "max_weight": float(w.max()) if w.size else 0.0,
            "weight_ratio": float(w.max() / max(w.min(), 1e-30)) if w.size else 0.0,
            "temperature": self.temperature,
        }

    def repeated_families(self, min_count: int = 2) -> Dict[str, int]:
        """Families proposed at least ``min_count`` times over the whole run."""
        return {f: c for f, c in self.seen_families.items() if c >= min_count}

    def sample_for_prompt(
        self,
        k_top: int = 4,
        k_diverse: int = 4,
        rng: Optional[np.random.Generator] = None,
    ) -> List[BufferItem]:
        """Top-weighted items plus a diverse sample drawn from distinct families."""
        if not self.items:
            return []
        rng = rng or np.random.default_rng()

        ordered = sorted(self.items, key=lambda it: it.weight, reverse=True)
        top = ordered[:k_top]

        chosen_families = {it.family for it in top}
        pool = [it for it in ordered[k_top:] if it.family not in chosen_families]
        if not pool:
            pool = ordered[k_top:]
        if not pool or k_diverse <= 0:
            return top

        size = min(k_diverse, len(pool))
        w = np.array([it.weight for it in pool], dtype=float)
        if w.sum() <= 0:
            idx = rng.choice(len(pool), size=size, replace=False)
        else:
            idx = rng.choice(len(pool), size=size, replace=False, p=w / w.sum())
        return top + [pool[i] for i in idx]

    def best(self) -> Optional[BufferItem]:
        return self.items[0] if self.items else None

    def __len__(self) -> int:
        return len(self.items)
