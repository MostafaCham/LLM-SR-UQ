#!/usr/bin/env python3
"""Run an ablation rung over LLM-SRBench problems."""

from __future__ import annotations

import threadcap  # noqa: F401  (must precede NumPy)

import argparse
import datetime as dt
import json
import os
import re
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from benchmark import SUBSETS, Problem, load_subset
from benchmark.complexity import complexity_report
from benchmark.metrics import (
    calibration_report,
    evaluate_point,
    inclusion_probabilities,
    symbolic_accuracy,
    symbolic_accuracy_at_k,
)
from devset import load_dev_problems
from critic import CriticClient, CriticFeedback, budget_split, extra_split
from diagnostics import A1Feedback
from hypothesis_generators import LocalHeuristicGenerator, OpenAIHypothesisGenerator
from llmsr_pipeline import LLMSRUQPipeline, PipelineConfig
from reporting import export_run, write_conversations, write_results_csv
from utils import set_seed

DEFAULT_LOCAL_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")

SA_MIN_N = 24

SA_KEYS = ("symbolic_accuracy", "sa_argmax", "sa_at_1", "sa_at_3", "sa_at_5",
           "sa_at_10", "sa_at_all", "sa_in_credible_set",
           "sa_rank_when_found", "sa_mass_when_found")


def _withhold_sa(agg: Dict, reason: str) -> Dict:
    """Blank the SA aggregates, leaving everything else intact."""
    out = dict(agg)
    for k in SA_KEYS:
        if k in out:
            out[k] = None
    out["sa_withheld"] = reason
    return out


def build_prompt(problem: Problem) -> str:
    """Base prompt: the benchmark's own scientific context plus the grammar rules."""
    return (
        f"{problem.description.strip()}\n\n"
        f"Propose candidate closed-form expressions for {problem.output_var} "
        f"in terms of {', '.join(problem.input_vars)}.\n"
        "Prefer forms that are physically plausible for this domain, and keep "
        "them numerically well behaved across the whole input range."
    )


def _problem_seed(base: int, instance_id: str) -> int:
    """Stable per-problem seed, so runs are reproducible but not identical."""
    return (base + sum(ord(c) * (i + 1) for i, c in enumerate(instance_id))) % (2 ** 31)


def variance_report(out: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Where the predictive uncertainty on this split comes from."""
    total = float(np.mean(out["var"]))
    rep = {
        "pred_var_mean": total,
        "pred_sd_mean": float(np.mean(np.sqrt(out["var"]))),
        "var_noise_mean": float(np.mean(out["var_noise"])),
        "var_param_mean": float(np.mean(out["var_param"])),
        "var_struct_mean": float(np.mean(out["var_struct"])),
    }
    if total > 0 and np.isfinite(total):
        for part in ("noise", "param", "struct"):
            rep[f"var_frac_{part}"] = rep[f"var_{part}_mean"] / total
    else:
        for part in ("noise", "param", "struct"):
            rep[f"var_frac_{part}"] = float("nan")
    return rep


def build_generator(cfg: PipelineConfig, llm: str, llm_model: str, base_url: Optional[str]):
    """Pick the proposer backbone."""
    if llm == "none":
        return LocalHeuristicGenerator(seed=cfg.seed)
    if llm == "local":
        return OpenAIHypothesisGenerator(
            model=llm_model,
            temperature=0.9,
            base_url=base_url or DEFAULT_LOCAL_BASE_URL,
            request_timeout=300,
            fallback=cfg.fallback,
            seed=cfg.seed,
        )
    return OpenAIHypothesisGenerator(model=llm_model, temperature=0.9,
                                     fallback=cfg.fallback, seed=cfg.seed)


def build_critic_client(cfg: PipelineConfig):
    """The critic's endpoint for A2/A3. Module-level so a test can substitute a scripted stub."""
    return CriticClient(cfg.critic_model, base_url=cfg.critic_base_url, seed=cfg.seed,
                        max_tokens=cfg.critic_max_tokens)


def _critic_fields(feedback_fn, pipe) -> Dict:
    """The critic's mechanism summary and full log, for rungs that have a critic."""
    if not isinstance(feedback_fn, CriticFeedback):
        return {}
    return {"critic": feedback_fn.summary(pipe.history), "critic_log": feedback_fn.log}


def write_critic_log(rec: Dict, run_dir: str) -> None:
    """One JSONL per problem: blob, raw reply, guidance and every validation outcome."""
    out = os.path.join(run_dir, "critic")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{rec['instance_id']}.jsonl"), "w", encoding="utf-8") as fh:
        for entry in rec.get("critic_log") or []:
            fh.write(json.dumps({**entry, "instance_id": rec["instance_id"]}) + "\n")


def run_problem(problem: Problem, cfg: PipelineConfig, llm: str, llm_model: str,
                base_url: Optional[str] = None) -> Dict:
    generator = build_generator(cfg, llm, llm_model, base_url)

    feedback_fn = None
    if cfg.rung == "A1":
        feedback_fn = A1Feedback(problem.input_vars)
    elif cfg.rung in ("A2", "A3"):
        if cfg.critic_extra:
            _, critic_rounds = extra_split(cfg.iterations, cfg.critic_every)
        else:
            _, critic_rounds = budget_split(cfg.llm_calls_budget or cfg.iterations, cfg.critic_every)
        feedback_fn = CriticFeedback(problem.input_vars, build_critic_client(cfg), build_prompt(problem),
                                     cfg.batch_hypotheses, critic_rounds, calibration=cfg.rung == "A3")
    pipe = LLMSRUQPipeline(cfg, generator, feedback_fn=feedback_fn)
    wants_target = cfg.weighted_conformal or cfg.conformal == "weighted"
    target_for_cal = problem.ood.X if (wants_target and problem.has_ood) else None

    pipe.fit(
        problem.train.X, problem.train.y,
        problem.cal.X, problem.cal.y,
        var_names=problem.input_vars,
        base_prompt=build_prompt(problem),
        X_target_for_cal=target_for_cal,
    )

    if pipe.best_expression() is None:
        rejections = [
            c.reason for r in pipe.history for c in r.rejected
        ][:3]
        return {
            "instance_id": problem.instance_id,
            "subset": problem.subset,
            "ablation": cfg.ablation,
            "rung": cfg.rung,
            "seed": cfg.base_seed,
            "error": "no candidate entered the posterior buffer",
            "example_rejections": rejections,
            "search": pipe.search_diagnostics(),
            "conversations": pipe.conversation_log(),
            **_critic_fields(feedback_fn, pipe),
        }

    record: Dict = {
        "instance_id": problem.instance_id,
        "subset": problem.subset,
        "ablation": cfg.ablation,
        "rung": cfg.rung,
        "seed": cfg.base_seed,
        "problem_seed": cfg.seed,
        "llm": llm if llm == "none" else llm_model,
        "input_vars": problem.input_vars,
        "output_var": problem.output_var,
        "best_expression": pipe.best_expression(),
        "gt_expression": problem.gt_expression,
        "gt_is_numeric": problem.gt_is_numeric,
        "search": pipe.search_diagnostics(),
    }

    for name, split in (("test", problem.test), ("ood", problem.ood)):
        if split is None:
            continue
        out = pipe.predict(split.X)
        record[name] = {
            **evaluate_point(split.y, out["mean"]),
            **calibration_report(split.y, out["pi_lo"], out["pi_hi"],
                                 alpha=cfg.alpha, normalise_by=split.y),
            **variance_report(out),
        }

    best = pipe.best_expression()

    structures = pipe.posterior_structures()
    record["structures"] = [
        {k: v for k, v in s.items() if k != "best_log_weight"} for s in structures
    ]
    record["symbolic"] = symbolic_accuracy_at_k(
        structures, problem.gt_expression, problem.input_vars,
        problem.gt_is_numeric, alpha=cfg.alpha,
    )

    record["symbolic_accuracy"] = (
        symbolic_accuracy(best, problem.gt_expression, problem.input_vars,
                          problem.gt_is_numeric)
        if best else 0.0
    )
    record["symbolic"]["sa_argmax"] = record["symbolic_accuracy"]

    record["complexity"] = complexity_report(
        best, problem.input_vars, n_data=len(problem.train),
        complexity_penalty=cfg.complexity_penalty,
    )
    record["coefficients"] = pipe.coefficient_uncertainty()
    record["inclusion"] = inclusion_probabilities(structures, problem.input_vars)
    record["scalers"] = pipe.scalers()
    record["conversations"] = pipe.conversation_log()
    record.update(_critic_fields(feedback_fn, pipe))
    if "critic" in record:
        record["search"]["critic_calls"] = record["critic"]["critic_calls"]
        record["search"]["llm_calls_total"] = record["search"]["llm_calls"] + record["critic"]["critic_calls"]
    return record


class ProblemTimeout(BaseException):
    """Raised by the per-problem alarm."""


@contextmanager
def problem_deadline(timeout_s: Optional[float]):
    """Raise ProblemTimeout if the block outlives ``timeout_s`` seconds."""
    if not timeout_s or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(signum, frame):
        raise ProblemTimeout(f"problem exceeded {timeout_s}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _run_one(payload: Tuple[Problem, PipelineConfig, str, str, Optional[str], int]) -> Dict:
    """Worker entry point. Never raises: a crash on one problem must not sink the run."""
    problem, cfg, llm, llm_model, base_url, timeout_s = payload
    try:
        with problem_deadline(timeout_s):
            return run_problem(problem, cfg, llm, llm_model, base_url)
    except (Exception, ProblemTimeout) as exc:
        return {
            "instance_id": problem.instance_id,
            "subset": problem.subset,
            "ablation": cfg.ablation,
            "rung": cfg.rung,
            "seed": cfg.base_seed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def run_pool(payloads, keys, workers: int, budget: Optional[float], worker_fn,
             on_record: Callable[[Dict, int], None], straggler: Callable[[str], Dict]) -> None:
    """Run payloads across worker processes, up to a wall-clock budget for the run."""
    pool = ProcessPoolExecutor(max_workers=workers)
    futures = {pool.submit(worker_fn, pl): key for pl, key in zip(payloads, keys)}
    done = 0
    try:
        for i, fut in enumerate(as_completed(futures, timeout=budget), 1):
            on_record(fut.result(), i)
            done = i
    except FuturesTimeout:
        stuck = sorted(key for fut, key in futures.items() if not fut.done())
        print(f"\nrun deadline reached; {len(stuck)} problem(s) never returned: "
              f"{', '.join(stuck)}", file=sys.stderr)
        for j, key in enumerate(stuck, done + 1):
            on_record(straggler(key), j)
        for proc in list((getattr(pool, "_processes", None) or {}).values()):
            proc.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        return
    pool.shutdown(wait=True)


def _print_result(rec: Dict, i: int, total: int) -> None:
    head = f"[{i}/{total}] {rec['instance_id']}"
    if "error" in rec:
        print(f"{head}  FAILED: {rec['error']}")
        return
    t, o = rec["test"], rec.get("ood")
    print(f"{head}\n    best: {rec['best_expression']}")
    line = f"    test nmse={t['nmse']:.4g} cov={t['coverage']:.3f}"
    if o:
        line += f" | ood nmse={o['nmse']:.4g} cov={o['coverage']:.3f}"
    print(line + f" | sym_acc={rec['symbolic_accuracy']:.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=list(SUBSETS), default="lsr_synth_bio_pop_growth")
    ap.add_argument("--limit", type=int, default=None, help="first N problems of the subset")
    ap.add_argument("--dev", action="store_true",
                    help="use the frozen 20-problem dev set (see devset.py)")
    ap.add_argument("--workers", type=int, default=1,
                    help="problems fitted in parallel; BLAS is pinned to 1 thread each")
    ap.add_argument("--ablation", default="A0")
    ap.add_argument("--rung", choices=["A0", "A1", "A2", "A3"], default="A0",
                    help="the ablation rung to execute. A0 sends no feedback; A1 "
                         "appends mechanical feedback from diagnostics.py; A2 adds the "
                         "critic; A3 gives the critic label-free calibration evidence "
                         "(critic.py). Separate from --ablation, which is only a label.")
    ap.add_argument("--llm", choices=["none", "openai", "local"], default="none",
                    help="proposer backbone: none=templates, openai=hosted API, "
                         "local=OpenAI-compatible server (vLLM)")
    ap.add_argument("--llm_model", default="gpt-4o-mini",
                    help="with --llm local, the name the server was started with, "
                         "e.g. meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--llm_base_url", default=None,
                    help=f"endpoint for --llm local (default {DEFAULT_LOCAL_BASE_URL})")
    ap.add_argument("--critic_llm", choices=["local", "openai"], default="local",
                    help="A2/A3 critic endpoint: local = the proposer's server (--llm_base_url), "
                         "openai = the hosted API with OPENAI_API_KEY from the environment")
    ap.add_argument("--critic_model", default=None,
                    help="critic model; defaults to --llm_model with --critic_llm local, "
                         "required with --critic_llm openai (e.g. gpt-4o)")
    ap.add_argument("--critic_every", type=int, default=5,
                    help="one LLM call in this many goes to the critic; --iterations is then "
                         "the total LLM-call budget, shared")
    ap.add_argument("--critic_extra", action="store_true",
                    help="add the critic's calls on top of --iterations instead of taking them "
                         "out of it: --iterations proposer rounds, as in A0/A1, plus one critic "
                         "call after each --critic_every rounds")
    ap.add_argument("--critic_max_tokens", type=int, default=700,
                    help="cap on a critic reply; 700 fits the local 8B's 4,096-token context")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--fit_time_budget", type=float, default=120.0,
                    help="per-candidate backstop in seconds: no new restart starts after it "
                         "(0 disables)")
    ap.add_argument("--buffer_size", type=int, default=24)
    ap.add_argument("--target_effective", type=float, default=0.35,
                    help="hold effective model count at this fraction of the buffer")
    ap.add_argument("--fixed_temperature", type=float, default=None,
                    help="disable auto-tempering and pin this value instead")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--weighted_conformal", action="store_true",
                    help="alias for --conformal weighted")
    ap.add_argument("--conformal", choices=["split", "weighted", "normalized", "structural"], default="split",
                    help="interval calibration: constant half-width (split), density-ratio "
                         "reweighting (weighted), half-width scaled by structural "
                         "disagreement plus noise (normalized), or by "
                         "structural disagreement alone (structural)")
    ap.add_argument("--weighting", choices=["tempered", "hybrid"], default="tempered",
                    help="which weights make the BMA mean: the buffer's tempered weights, or raw "
                         "evidence for the mean with the tempered spread around it (hybrid). "
                         "Changes prediction only, never the search.")
    ap.add_argument("--standardize", action="store_true",
                    help="fit on standardised inputs and targets instead of physical units")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR",
                    help="continue an interrupted run: reuse RUN_DIR, skip every "
                         "problem already in its results.jsonl, and append. Records "
                         "are written per problem, so a preempted job loses only the "
                         "problem in flight, but only if it is resumed rather than "
                         "restarted.")
    ap.add_argument("--heldout", action="store_true",
                    help="run the held-out part of --subset: every problem not in the frozen "
                         "dev set")
    ap.add_argument("--only", default=None,
                    help="comma-separated instance_ids to run, filtered from the "
                         "selected set. Per-problem seeds derive from the run seed "
                         "and the instance_id alone, so a problem rerun this way "
                         "reproduces exactly what the full run would have given it.")
    ap.add_argument("--problem_timeout", type=int, default=3600,
                    help="wall-clock seconds one problem may take before it is "
                         "recorded as failed (0 disables). Guards the run against "
                         "a single problem stalling it indefinitely.")
    ap.add_argument("--fallback", action="store_true",
                    help="top up a short LLM reply with heuristic templates. Off by "
                         "default: those proposals cost no LLM call, so they break the "
                         "call-matched budget a backbone comparison rests on.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    named = re.match(r"^A(\d)", args.ablation)
    if named and f"A{named.group(1)}" != args.rung:
        ap.error(f"--ablation {args.ablation!r} names rung A{named.group(1)} "
                 f"but --rung is {args.rung}")
    if args.heldout and args.dev:
        ap.error("--heldout and --dev select disjoint problem sets; pick one")
    if args.weighted_conformal and args.conformal not in ("split", "weighted"):
        ap.error(f"--weighted_conformal conflicts with --conformal {args.conformal}")

    critic_model = critic_base_url = None
    proposer_rounds = llm_calls_budget = args.iterations
    if args.critic_extra and args.rung not in ("A2", "A3"):
        ap.error("--critic_extra needs a critic rung (--rung A2 or A3)")
    if args.rung in ("A2", "A3"):
        if args.critic_llm == "local":
            if args.llm != "local" and not args.critic_model:
                ap.error("--critic_llm local needs --llm local, or --critic_model for a server "
                         "at --llm_base_url")
            critic_model = args.critic_model or args.llm_model
            critic_base_url = args.llm_base_url or DEFAULT_LOCAL_BASE_URL
        else:
            if not args.critic_model:
                ap.error("--critic_llm openai needs --critic_model, e.g. gpt-4o")
            if not os.environ.get("OPENAI_API_KEY"):
                ap.error("--critic_llm openai needs OPENAI_API_KEY in the environment")
            critic_model = args.critic_model
        if args.critic_extra:
            proposer_rounds, critic_rounds = extra_split(args.iterations, args.critic_every)
            llm_calls_budget = proposer_rounds + len(critic_rounds)
        else:
            proposer_rounds, critic_rounds = budget_split(args.iterations, args.critic_every)
        print(f"budget: {llm_calls_budget} LLM calls = {proposer_rounds} proposer rounds "
              f"+ {len(critic_rounds)} critic calls")

    set_seed(args.seed)

    if args.dev:
        problems = load_dev_problems(seed=args.seed)
        label = "dev20"
    else:
        problems = load_subset(args.subset, seed=args.seed, limit=args.limit)
        label = args.subset

    if args.heldout:
        from devset import load_ids

        dev_ids = {pid for ids in load_ids().values() for pid in ids}
        problems = [p for p in problems if p.instance_id not in dev_ids]
        label = f"{args.subset}-heldout"

    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        problems = [p for p in problems if p.instance_id in wanted]
        missing = wanted - {p.instance_id for p in problems}
        if missing:
            print(f"--only: not in the selected set: {sorted(missing)}", file=sys.stderr)
            return 1
        label = f"{label}-only{len(problems)}"

    cfg = PipelineConfig(
        iterations=proposer_rounds,
        batch_hypotheses=args.batch,
        n_restarts=args.restarts,
        fit_time_budget=args.fit_time_budget or None,
        buffer_size=args.buffer_size,
        auto_temperature=args.fixed_temperature is None,
        target_effective_fraction=args.target_effective,
        evidence_temperature=args.fixed_temperature or 1.0,
        alpha=args.alpha,
        weighted_conformal=args.weighted_conformal,
        conformal=args.conformal,
        weighting=args.weighting,
        standardize=args.standardize,
        fallback=args.fallback,
        ablation=args.ablation,
        rung=args.rung,
        critic_model=critic_model,
        critic_base_url=critic_base_url,
        critic_every=args.critic_every,
        critic_extra=args.critic_extra,
        critic_max_tokens=args.critic_max_tokens,
        llm_calls_budget=llm_calls_budget,
        seed=args.seed,
        base_seed=args.seed,
        verbose=not args.quiet and args.workers == 1,
    )

    all_problem_ids = [p.instance_id for p in problems]

    done: Dict[str, Dict] = {}
    if args.resume:
        run_dir = args.resume
        jsonl = os.path.join(run_dir, "results.jsonl")
        if not os.path.exists(jsonl):
            print(f"--resume: no results.jsonl in {run_dir}", file=sys.stderr)
            return 1
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if "error" not in rec:
                        done[rec["instance_id"]] = rec
        remaining = [p for p in problems if p.instance_id not in done]
        print(f"resuming {run_dir}: {len(done)} done, {len(remaining)} to go")
        problems = remaining
        if not problems:
            print("nothing left to run")
            return 0
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(args.out_dir, f"{args.ablation}_{label}_s{args.seed}_{stamp}")
        run_dir, n = base, 1
        while True:
            try:
                os.makedirs(run_dir)
                break
            except FileExistsError:
                n += 1
                run_dir = f"{base}_{n}"
    os.makedirs(run_dir, exist_ok=True)

    backbone = args.llm if args.llm == "none" else f"{args.llm}:{args.llm_model}"
    print(f"{cfg.ablation} [{cfg.rung}] | {len(problems)} problems | llm={backbone} "
          f"| {args.iterations} iters x {args.batch} hypotheses | workers={args.workers}")
    print(f"writing to {run_dir}\n")

    payloads = []
    for p in problems:
        pcfg = replace(cfg, seed=_problem_seed(args.seed, p.instance_id))
        payloads.append((p, pcfg, args.llm, args.llm_model, args.llm_base_url,
                         args.problem_timeout))

    records: List[Dict] = list(done.values())
    jsonl_path = os.path.join(run_dir, "results.jsonl")
    with open(jsonl_path, "a" if args.resume else "w", encoding="utf-8") as fh:
        def record(rec: Dict, i: int) -> None:
            records.append(rec)
            slim = {k: v for k, v in rec.items() if k not in ("conversations", "critic_log")}
            fh.write(json.dumps(slim) + "\n")
            fh.flush()
            write_results_csv(records, run_dir)
            if rec.get("conversations"):
                write_conversations([rec], run_dir)
            if rec.get("critic_log"):
                write_critic_log(rec, run_dir)
            _print_result(rec, i, len(problems))

        if args.workers > 1:
            budget = (max(args.problem_timeout * len(payloads) / args.workers * 2, args.problem_timeout + 600)
                      if args.problem_timeout else None)
            run_pool(
                payloads, [pl[0].instance_id for pl in payloads], args.workers, budget,
                _run_one, record,
                lambda pid: {"instance_id": pid, "ablation": cfg.ablation, "rung": cfg.rung,
                             "seed": cfg.base_seed,
                             "error": "run deadline reached before this problem returned"},
            )
        else:
            for i, pl in enumerate(payloads, 1):
                record(_run_one(pl), i)

    summarise(records, run_dir, is_dev=args.dev)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({**asdict(cfg), "llm": args.llm, "llm_model": args.llm_model,
                   "llm_base_url": (args.llm_base_url or DEFAULT_LOCAL_BASE_URL)
                   if args.llm == "local" else None,
                   "problems": all_problem_ids,
                   "resumed": bool(args.resume)}, fh, indent=2)
    return 0


def _fmt(v: Optional[float], places: int = 3) -> str:
    """Format an aggregate that may legitimately be absent."""
    return "-" if v is None else f"{v:.{places}f}"


def _mean_if(vals: List[float]) -> Optional[float]:
    """Mean of a possibly empty list; None rather than nan so JSON stays clean."""
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _aggregate(rows: List[Dict]) -> Dict:
    """NMSE is summarised by median because a single divergent model can carry a mean into the thousands; coverage and width are means, which is what the conformal guarantee is stated over."""
    def med(path: str, key: str):
        vals = [r[path][key] for r in rows if path in r and np.isfinite(r[path][key])]
        return float(np.median(vals)) if vals else None

    def avg(path: str, key: str):
        vals = [r[path][key] for r in rows if path in r and np.isfinite(r[path][key])]
        return float(np.mean(vals)) if vals else None

    def block(name: str, key: str, median: bool = False):
        """Aggregate one field of a nested record block over problems."""
        vals = [r[name][key] for r in rows
                if name in r and r[name].get(key) is not None
                and np.isfinite(r[name][key])]
        if not vals:
            return None
        return float(np.median(vals) if median else np.mean(vals))

    def search(key: str):
        vals = [r["search"][key] for r in rows
                if "search" in r and r["search"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n": len(rows),
        "symbolic_accuracy": float(np.mean([r["symbolic_accuracy"] for r in rows])),
        "sa_argmax": avg("symbolic", "sa_argmax"),
        "sa_at_1": avg("symbolic", "sa_at_1"),
        "sa_at_3": avg("symbolic", "sa_at_3"),
        "sa_at_5": avg("symbolic", "sa_at_5"),
        "sa_at_10": avg("symbolic", "sa_at_10"),
        "sa_at_all": avg("symbolic", "sa_at_all"),
        "sa_in_credible_set": avg("symbolic", "sa_in_credible_set"),
        "credible_set_size": avg("symbolic", "credible_set_size"),
        "n_families": avg("symbolic", "n_families"),
        "sa_mass_when_found": _mean_if([r["symbolic"]["sa_mass"] for r in rows
                                        if r.get("symbolic", {}).get("sa_at_all")]),
        "sa_rank_when_found": _mean_if([r["symbolic"]["sa_rank"] for r in rows
                                        if r.get("symbolic", {}).get("sa_at_all")]),
        "test_nmse_median": med("test", "nmse"),
        "test_acc_tau_0.1": avg("test", "acc_tau_0.1"),
        "test_frac_within_0.1": avg("test", "frac_within_0.1"),
        "test_coverage": avg("test", "coverage"),
        "test_mean_width": avg("test", "mean_width"),
        "ood_nmse_median": med("ood", "nmse"),
        "ood_acc_tau_0.1": avg("ood", "acc_tau_0.1"),
        "ood_frac_within_0.1": avg("ood", "frac_within_0.1"),
        "ood_coverage": avg("ood", "coverage"),
        "ood_mean_width": avg("ood", "mean_width"),
        "test_var_frac_struct": avg("test", "var_frac_struct"),
        "ood_var_frac_struct": avg("ood", "var_frac_struct"),
        "n_nodes": block("complexity", "n_nodes"),
        "n_params": block("complexity", "n_params"),
        "mdl_total_bits": block("complexity", "mdl_total_bits"),
        "mean_relative_sd": block("coefficients", "mean_relative_sd", median=True),
        "frac_weak_parameters": block("coefficients", "frac_weak_parameters"),
        "rejection_rate": search("rejection_rate"),
        "proposals_per_llm_call": search("proposals_per_llm_call"),
        "fallback_rate": search("fallback_rate"),
        "effective_models": search("effective_models"),
        "longest_plateau": search("longest_plateau"),
    }


def summarise(records: List[Dict], run_dir: str, is_dev: bool = False) -> None:
    ok = [r for r in records if "test" in r]
    failed = [r for r in records if "error" in r]
    if not ok:
        print("no successful runs to summarise")
        for r in failed:
            print(f"  {r['instance_id']}: {r['error']}")
        export_run(records, run_dir, overall={}, by_domain={}, n_failed=len(failed))
        return

    overall = _aggregate(ok)
    by_domain = {}
    for subset in sorted({r["subset"] for r in ok}):
        by_domain[subset] = _aggregate([r for r in ok if r["subset"] == subset])

    if is_dev:
        overall = _withhold_sa(overall, "dev set")
        by_domain = {k: _withhold_sa(v, "dev set") for k, v in by_domain.items()}
    else:
        by_domain = {
            k: (v if v["n"] >= SA_MIN_N else _withhold_sa(v, f"n<{SA_MIN_N}"))
            for k, v in by_domain.items()
        }
        if overall["n"] < SA_MIN_N:
            overall = _withhold_sa(overall, f"n<{SA_MIN_N}")

    print("\n" + "=" * 96)
    print(f"SUMMARY  ({len(ok)} succeeded, {len(failed)} failed)")
    print("=" * 96)
    cols = ("n", "symbolic_accuracy", "sa_at_all", "test_acc_tau_0.1",
            "test_nmse_median", "test_coverage", "ood_acc_tau_0.1",
            "ood_nmse_median", "ood_coverage")
    header = f"{'domain':<30}" + "".join(f"{c.replace('_', ' '):>19}" for c in cols)
    print(header)
    print("-" * len(header))

    def row(name: str, agg: Dict) -> str:
        cells = ""
        for c in cols:
            v = agg[c]
            cells += f"{'-':>19}" if v is None else (
                f"{v:>19d}" if c == "n" else f"{v:>19.4g}")
        return f"{name:<30}{cells}"

    for subset, agg in by_domain.items():
        print(row(subset.replace("lsr_synth_", ""), agg))
    print("-" * len(header))
    print(row("ALL", overall))

    if overall.get("sa_withheld"):
        print(f"\nstructural posterior: WITHHELD ({overall['sa_withheld']})")
        if is_dev:
            print("  Dev-set SA is not reported: prompts are tuned on the dev set.")
        print("  Per-problem values remain in results.jsonl.")
    else:
        print("\nstructural posterior")
        print(f"  {'symbolic_accuracy':22} {_fmt(overall['symbolic_accuracy'])}"
              "   SA@1 on the emitted expression (comparable to published numbers)")
        for k in ("sa_at_1", "sa_at_3", "sa_at_5", "sa_at_10", "sa_at_all"):
            print(f"  {k:22} {_fmt(overall[k])}")
        print(f"  {'sa_in_credible_set':22} {_fmt(overall['sa_in_credible_set'])}"
              "   structural analogue of interval coverage, same alpha")
        print(f"  {'credible_set_size':22} {_fmt(overall['credible_set_size'], 1)}"
              f"   of {_fmt(overall['n_families'], 1)} families in the buffer")
        if overall["sa_rank_when_found"] is not None:
            print(f"  {'rank when found':22} {_fmt(overall['sa_rank_when_found'], 1)}"
                  f"   mass {_fmt(overall['sa_mass_when_found'])}")

    withheld = [k for k, v in by_domain.items() if v.get("sa_withheld")]
    if withheld and not is_dev:
        print(f"\n  per-domain SA withheld for n < {SA_MIN_N}: "
              + ", ".join(w.replace("lsr_synth_", "") for w in withheld))

    print("\npoint fit (mean over problems)")
    print(f"  {'frac within 10% (id)':22} {overall['test_frac_within_0.1']:.3f}"
          "   graded companion to Acc_0.1, not comparable to published numbers")
    if overall["ood_frac_within_0.1"] is not None:
        print(f"  {'frac within 10% (ood)':22} {overall['ood_frac_within_0.1']:.3f}")

    print("\nsimplicity + coefficient uncertainty (mean over problems)")
    for label, key in (("expression nodes", "n_nodes"),
                       ("fitted parameters", "n_params"),
                       ("MDL total (bits)", "mdl_total_bits"),
                       ("rel. coef. SD (median)", "mean_relative_sd"),
                       ("frac weak params", "frac_weak_parameters")):
        if overall.get(key) is not None:
            print(f"  {label:22} {overall[key]:.3f}")

    print("\nvariance decomposition (share from disagreement between structures)")
    for label, key in (("id", "test_var_frac_struct"), ("ood", "ood_var_frac_struct")):
        if overall.get(key) is not None:
            print(f"  {label:22} {overall[key]:.3f}")

    print("\nsearch behaviour (mean over problems)")
    for k in ("rejection_rate", "effective_models", "longest_plateau"):
        print(f"  {k:20} {overall[k]:.3f}")
    if overall["proposals_per_llm_call"] is not None:
        print(f"  {'proposals per call':20} {overall['proposals_per_llm_call']:.3f}")
    if overall["fallback_rate"]:
        print(f"  {'fallback rate':20} {overall['fallback_rate']:.3f}"
              "   <-- free proposals; budget is NOT call-matched")
    print(f"  {'test mean width':20} {overall['test_mean_width']:.3f}")
    if overall["ood_mean_width"] is not None:
        print(f"  {'ood mean width':20} {overall['ood_mean_width']:.3f}")

    if failed:
        print("\nfailures")
        for r in failed:
            print(f"  {r['instance_id']}: {r['error']}")

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"overall": overall, "by_domain": by_domain,
                   "n_failed": len(failed)}, fh, indent=2)
    export_run(records, run_dir, overall=overall, by_domain=by_domain, n_failed=len(failed))


if __name__ == "__main__":
    sys.exit(main())
