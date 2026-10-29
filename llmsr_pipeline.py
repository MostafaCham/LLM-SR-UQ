#!/usr/bin/env python3
"""Search loop: propose, fit, retain, then calibrate intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from bma_conformal import (
    BMAPredictor,
    NormalizedConformal,
    SplitConformal,
    StructuralConformal,
    WeightedConformal,
    estimate_density_ratio_logreg,
)
from equation_program import EquationProgram
from hypothesis_generators import HypothesisGenerator, _validate
from parameter_inference import FitResult, fit_program
from posterior_buffer import PosteriorBuffer


@dataclass
class PipelineConfig:
    iterations: int = 16
    batch_hypotheses: int = 8

    n_restarts: int = 6
    max_nfev: int = 2000
    fit_time_budget: Optional[float] = 120.0

    buffer_size: int = 24
    evidence_mode: str = "bic"
    complexity_penalty: float = 0.05
    auto_temperature: bool = True
    target_effective_fraction: float = 0.35
    evidence_temperature: float = 1.0

    alpha: float = 0.1
    n_param_samples: int = 30
    weighted_conformal: bool = False
    conformal: str = "split"
    weighting: str = "tempered"

    standardize: bool = False
    max_params: int = 12

    fallback: bool = False

    ablation: str = "A0"
    rung: str = "A0"
    critic_model: Optional[str] = None
    critic_base_url: Optional[str] = None
    critic_every: int = 5
    critic_extra: bool = False
    critic_max_tokens: int = 700
    llm_calls_budget: Optional[int] = None
    seed: int = 0
    base_seed: int = 0
    verbose: bool = True


@dataclass
class Candidate:
    expression: str
    accepted: bool
    reason: str = ""
    train_mse: Optional[float] = None
    log_evidence: Optional[float] = None
    n_params: Optional[int] = None
    unused_variables: List[str] = field(default_factory=list)
    weak_parameters: List[str] = field(default_factory=list)
    protection: Dict[str, float] = field(default_factory=dict)
    entered_buffer: bool = False
    source: str = "llm"

    def as_dict(self) -> Dict:
        return {
            "expression": self.expression,
            "accepted": self.accepted,
            "source": self.source,
            "reason": self.reason,
            "train_mse": self.train_mse,
            "log_evidence": self.log_evidence,
            "n_params": self.n_params,
            "unused_variables": list(self.unused_variables),
            "weak_parameters": list(self.weak_parameters),
            "protection": dict(self.protection),
            "entered_buffer": self.entered_buffer,
        }


@dataclass
class IterationRecord:
    iteration: int
    prompt: str
    candidates: List[Candidate] = field(default_factory=list)
    buffer_diagnostics: Dict[str, float] = field(default_factory=dict)
    best_expression: Optional[str] = None
    best_log_evidence: Optional[float] = None
    raw_response: Optional[str] = None
    system_message: Optional[str] = None
    feedback: str = ""
    diagnostics: Optional[Dict] = None

    @property
    def accepted(self) -> List[Candidate]:
        return [c for c in self.candidates if c.accepted]

    @property
    def rejected(self) -> List[Candidate]:
        return [c for c in self.candidates if not c.accepted]

    @property
    def rejection_rate(self) -> float:
        return len(self.rejected) / max(len(self.candidates), 1)

    @property
    def n_fallback(self) -> int:
        """Proposals this iteration that cost no LLM call."""
        return sum(1 for c in self.candidates if c.source == "fallback")

    def as_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "system_message": self.system_message,
            "feedback": self.feedback,
            "diagnostics": self.diagnostics,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "best_expression": self.best_expression,
            "best_log_evidence": self.best_log_evidence,
            "buffer_diagnostics": dict(self.buffer_diagnostics),
            "n_accepted": len(self.accepted),
            "n_rejected": len(self.rejected),
            "n_fallback": self.n_fallback,
            "proposed_expressions": [c.expression for c in self.candidates],
            "candidates": [c.as_dict() for c in self.candidates],
        }


class LLMSRUQPipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        generator: HypothesisGenerator,
        feedback_fn: Optional[Callable[["LLMSRUQPipeline", List[IterationRecord]],
                                       Union[str, Tuple[str, Optional[Dict]]]]] = None,
    ):
        self.cfg = cfg
        self.generator = generator
        self.feedback_fn = feedback_fn

        self.buffer = PosteriorBuffer(
            max_items=cfg.buffer_size,
            evidence_mode=cfg.evidence_mode,
            temperature=cfg.evidence_temperature,
            complexity_penalty=cfg.complexity_penalty,
        )
        self.bma = BMAPredictor(self.buffer, n_param_samples=cfg.n_param_samples, seed=cfg.seed,
                                weighting=cfg.weighting)
        mode = "weighted" if cfg.weighted_conformal else cfg.conformal
        classes = {"split": SplitConformal, "weighted": WeightedConformal,
                   "normalized": NormalizedConformal, "structural": StructuralConformal}
        if mode not in classes:
            raise ValueError(f"unknown conformal mode {mode!r}; expected one of {sorted(classes)}")
        self.conformal = classes[mode](alpha=cfg.alpha)

        self.history: List[IterationRecord] = []
        self._Xtr: Optional[np.ndarray] = None
        self._ytr: Optional[np.ndarray] = None
        self._Xca: Optional[np.ndarray] = None
        self._yca: Optional[np.ndarray] = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def _fit_scalers(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self.cfg.standardize:
            return
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std < 1e-12] = 1.0
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0

    def _tx(self, X: np.ndarray) -> np.ndarray:
        return X if self._x_mean is None else (X - self._x_mean) / self._x_std

    def _tx_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self._y_mean) / self._y_std

    def _inv_y(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self._y_std + self._y_mean

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        var_names: Sequence[str],
        base_prompt: str,
        X_target_for_cal: Optional[np.ndarray] = None,
    ) -> None:
        self._fit_scalers(X_train, y_train)
        Xtr, ytr = self._tx(X_train), self._tx_y(y_train)
        Xca, yca = self._tx(X_cal), self._tx_y(y_cal)
        self._Xtr, self._ytr = Xtr, ytr
        self._Xca, self._yca = Xca, yca

        for t in range(self.cfg.iterations):
            feedback, diagnostics, inject = "", None, []
            if self.feedback_fn is not None:
                out = self.feedback_fn(self, self.history)
                if isinstance(out, dict):
                    feedback, diagnostics = out.get("text", ""), out.get("diagnostics")
                    inject = list(out.get("inject") or [])
                else:
                    feedback, diagnostics = out if isinstance(out, tuple) else (out, None)
            prompt = base_prompt if not feedback else f"{base_prompt}\n\n{feedback}"

            record = IterationRecord(iteration=t + 1, prompt=prompt, feedback=feedback,
                                     diagnostics=diagnostics)
            injected = _validate(inject, var_names, max_params=12, source="critic_edit") if inject else None
            n_injected = len(injected.programs) if injected else 0
            generated = self.generator.generate(prompt, max(self.cfg.batch_hypotheses - n_injected, 1), var_names)
            record.raw_response = generated.raw_response
            record.system_message = generated.system_message

            for expr, reason in generated.invalid:
                record.candidates.append(Candidate(expr, accepted=False, reason=reason))

            sources = generated.sources or ["llm"] * len(generated.programs)
            for prog, src in zip(generated.programs, sources):
                self._evaluate_candidate(prog, Xtr, ytr, record, note=f"iter{t + 1}", source=src)
            if injected:
                for expr, reason in injected.invalid:
                    record.candidates.append(Candidate(expr, accepted=False, reason=reason, source="critic_edit"))
                for prog in injected.programs:
                    self._evaluate_candidate(prog, Xtr, ytr, record, note=f"iter{t + 1}", source="critic_edit")

            if self.cfg.auto_temperature:
                self.buffer.calibrate_temperature(self.cfg.target_effective_fraction)

            record.buffer_diagnostics = self.buffer.diagnostics()
            best = self.buffer.best()
            if best is not None:
                record.best_expression = best.program.expr
                record.best_log_evidence = self.buffer.choose_log_evidence(best.fit)

            self.history.append(record)

            if self.cfg.verbose:
                d = record.buffer_diagnostics
                head = (f"  [iter {t + 1:3d}/{self.cfg.iterations}] "
                        f"accepted={len(record.accepted)}/{len(record.candidates)}")
                if best is None:
                    print(f"{head}  no models in buffer yet")
                else:
                    print(f"{head}  buffer={d['n_items']} "
                          f"eff={d['effective_models']:.1f} "
                          f"best_mse={best.fit.train_mse:.4g}")

        self._calibrate(Xca, yca, X_target_for_cal)

    def _evaluate_candidate(self, prog, Xtr, ytr, record, note, source: str = "llm") -> None:
        try:
            fit = fit_program(
                prog, Xtr, ytr,
                n_restarts=self.cfg.n_restarts,
                max_nfev=self.cfg.max_nfev,
                seed=self.cfg.seed,
                time_budget=self.cfg.fit_time_budget or None,
            )
        except Exception as exc:
            record.candidates.append(
                Candidate(prog.expr, accepted=False, reason=f"fit failed: {exc}",
                          n_params=prog.n_params(), unused_variables=prog.unused_variables,
                          source=source)
            )
            return

        entered = self.buffer.add(prog, fit, note=note)
        record.candidates.append(
            Candidate(
                expression=prog.expr,
                accepted=True,
                train_mse=fit.train_mse,
                log_evidence=self.buffer.choose_log_evidence(fit),
                n_params=prog.n_params(),
                unused_variables=prog.unused_variables,
                weak_parameters=fit.weak_parameters,
                protection=fit.protection,
                entered_buffer=entered,
                source=source,
            )
        )

    def _calibrate(self, Xca, yca, X_target_for_cal) -> None:
        if not len(self.buffer):
            return
        if isinstance(self.conformal, NormalizedConformal):
            comp = self.bma.predict_components(Xca)
            self.conformal.fit_normalized(yca, comp["mean"], self.conformal.sigma(comp))
            return
        mu_cal, _ = self.bma.predict_mean_var(Xca)
        if isinstance(self.conformal, WeightedConformal) and X_target_for_cal is not None:
            try:
                w = estimate_density_ratio_logreg(Xca, self._tx(X_target_for_cal))
                self.conformal.fit_weighted(yca, mu_cal, w)
                return
            except Exception as exc:
                if self.cfg.verbose:
                    print(f"  weighted conformal unavailable ({exc}); falling back")
                self.conformal = SplitConformal(alpha=self.cfg.alpha)
        self.conformal.fit(yca, mu_cal)

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        Xz = self._tx(X)
        if len(self.buffer):
            comp = self.bma.predict_components(Xz)
        else:
            n = len(X)
            comp = {"mean": np.zeros(n), "var": np.ones(n), "var_noise": np.ones(n),
                    "var_param": np.zeros(n), "var_struct": np.zeros(n), "n_models": 0.0}
        mu_z, var_z = comp["mean"], comp["var"]

        if self.conformal.q is not None and isinstance(self.conformal, NormalizedConformal):
            lo_z, hi_z = self.conformal.predict_interval_scaled(mu_z, self.conformal.sigma(comp))
        elif self.conformal.q is not None:
            lo_z, hi_z = self.conformal.predict_interval(mu_z)
        else:
            sd = np.sqrt(var_z)
            lo_z, hi_z = mu_z - 1.96 * sd, mu_z + 1.96 * sd

        s2 = self._y_std ** 2
        return {
            "mean": self._inv_y(mu_z),
            "pi_lo": self._inv_y(lo_z),
            "pi_hi": self._inv_y(hi_z),
            "var": var_z * s2,
            "var_noise": comp["var_noise"] * s2,
            "var_param": comp["var_param"] * s2,
            "var_struct": comp["var_struct"] * s2,
        }

    def scalers(self) -> Dict:
        """The standardisation applied before fitting."""
        if self._x_mean is None:
            return {"standardized": False}
        return {
            "standardized": True,
            "x_mean": [float(v) for v in self._x_mean],
            "x_std": [float(v) for v in self._x_std],
            "y_mean": float(self._y_mean),
            "y_std": float(self._y_std),
        }

    def coefficient_uncertainty(self) -> Dict[str, float]:
        """How well determined the winning model's constants are."""
        best = self.buffer.best()
        if best is None:
            return {}
        fit = best.fit
        rel = []
        for name, mapv in fit.theta_map.items():
            sd = fit.param_sd.get(name)
            if sd is None or not np.isfinite(sd):
                continue
            rel.append(abs(sd) / max(abs(float(mapv)), 1e-12))
        sds = [v for v in fit.param_sd.values() if np.isfinite(v)]
        return {
            "n_params": int(fit.n_params),
            "n_weak_parameters": len(fit.weak_parameters),
            "frac_weak_parameters": len(fit.weak_parameters) / max(fit.n_params, 1),
            "mean_param_sd": float(np.mean(sds)) if sds else float("nan"),
            "max_param_sd": float(np.max(sds)) if sds else float("nan"),
            "mean_relative_sd": float(np.mean(rel)) if rel else float("nan"),
            "max_relative_sd": float(np.max(rel)) if rel else float("nan"),
            "sigma2_hat": float(fit.sigma2_hat),
        }

    def conversation_log(self) -> List[Dict]:
        """One dict per iteration: the prompt, the reply, and what was kept."""
        return [rec.as_dict() for rec in self.history]

    def best_expression(self) -> Optional[str]:
        best = self.buffer.best()
        return best.program.expr if best else None

    def posterior_structures(self) -> List[Dict]:
        """The buffer as a ranked distribution over *structural families*."""
        def coefficients(item) -> Dict:
            """The fitted constants and their Laplace standard deviations."""
            fit = item.fit
            return {
                "theta": {k: float(v) for k, v in fit.theta_map.items()},
                "param_sd": {k: float(v) for k, v in fit.param_sd.items()},
                "sigma2_hat": float(fit.sigma2_hat),
                "n_params": int(fit.n_params),
                "train_mse": float(fit.train_mse),
                "log_evidence_bic": float(fit.log_evidence_bic),
                "weak_parameters": list(fit.weak_parameters),
                "unidentified_parameters": list(fit.unidentified_parameters),
                "protection": {k: float(v) for k, v in fit.protection.items()},
            }

        by_family: Dict[str, Dict] = {}
        for item in self.buffer.items:
            entry = by_family.setdefault(item.family, {
                "family": item.family,
                "mass": 0.0,
                "expressions": [],
                "best_expression": item.program.expr,
                "best_log_weight": item.log_weight,
                **coefficients(item),
            })
            entry["mass"] += float(item.weight)
            entry["expressions"].append(item.program.expr)
            if item.log_weight > entry["best_log_weight"]:
                entry["best_expression"] = item.program.expr
                entry["best_log_weight"] = item.log_weight
                entry.update(coefficients(item))

        ranked = sorted(by_family.values(), key=lambda e: e["mass"], reverse=True)
        for i, entry in enumerate(ranked, 1):
            entry["rank"] = i
            entry["n_members"] = len(entry["expressions"])
        return ranked

    def search_diagnostics(self) -> Dict[str, float]:
        """Search-quality metrics."""
        if not self.history:
            return {}

        total = sum(len(r.candidates) for r in self.history)
        rejected = sum(len(r.rejected) for r in self.history)
        families = [len({c.expression for c in r.accepted}) for r in self.history]

        best_trace = [r.best_log_evidence for r in self.history if r.best_log_evidence is not None]
        iters_to_best = int(np.argmax(best_trace)) + 1 if best_trace else 0

        longest_plateau = current = 0
        for i in range(1, len(best_trace)):
            if best_trace[i] <= best_trace[i - 1] + 1e-9:
                current += 1
                longest_plateau = max(longest_plateau, current)
            else:
                current = 0

        repeated = self.buffer.repeated_families(min_count=2)

        llm_calls = sum(1 for r in self.history if r.raw_response is not None)
        n_fallback = sum(r.n_fallback for r in self.history)
        n_edits = sum(1 for r in self.history for c in r.candidates if c.source == "critic_edit")
        llm_proposals = total - n_fallback - n_edits
        short_iters = sum(1 for r in self.history if r.n_fallback > 0)

        return {
            "iterations": len(self.history),
            "candidates_total": total,
            "rejection_rate": rejected / max(total, 1),
            "mean_accepted_per_iter": float(np.mean(families)) if families else 0.0,
            "iterations_to_best": iters_to_best,
            "longest_plateau": longest_plateau,
            "distinct_structures_seen": len(self.buffer.seen_signatures),
            "distinct_families_seen": len(self.buffer.seen_families),
            "repeated_families": len(repeated),
            "effective_models": self.buffer.effective_model_count(),
            "llm_calls": llm_calls,
            "llm_proposals": llm_proposals,
            "critic_edit_candidates": n_edits,
            "proposals_per_llm_call": (llm_proposals / llm_calls) if llm_calls else None,
            "fallback_candidates": n_fallback,
            "fallback_rate": n_fallback / max(total, 1),
            "short_reply_iterations": short_iters,
        }
