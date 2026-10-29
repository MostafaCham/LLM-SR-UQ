#!/usr/bin/env python3
"""CSV (and conversation) writers for a benchmark run."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

RESULT_COLUMNS = [
    "instance_id", "subset", "ablation", "rung", "seed", "llm", "error",
    "input_vars", "output_var",
    "best_expression", "gt_expression", "gt_is_numeric", "symbolic_accuracy",
    "symbolic_sa_argmax", "symbolic_sa_at_1", "symbolic_sa_at_3", "symbolic_sa_at_5",
    "symbolic_sa_at_10", "symbolic_sa_at_all",
    "symbolic_sa_rank", "symbolic_sa_mass",
    "symbolic_n_families", "symbolic_credible_set_size", "symbolic_sa_in_credible_set",
    "test_nmse", "test_r2", "test_mse", "test_rmse",
    "test_mae", "test_mape", "test_max_rel_error",
    "test_acc_tau_0.1", "test_acc_tau_0.01", "test_acc_tau_0.001",
    "test_frac_within_0.1", "test_frac_within_0.01", "test_frac_within_0.001",
    "test_coverage", "test_target_coverage", "test_coverage_gap",
    "test_mean_width", "test_sharpness_at_coverage",
    "test_pred_var_mean", "test_pred_sd_mean",
    "test_var_noise_mean", "test_var_param_mean", "test_var_struct_mean",
    "test_var_frac_noise", "test_var_frac_param", "test_var_frac_struct",
    "ood_nmse", "ood_r2", "ood_mse", "ood_rmse",
    "ood_mae", "ood_mape", "ood_max_rel_error",
    "ood_acc_tau_0.1", "ood_acc_tau_0.01", "ood_acc_tau_0.001",
    "ood_frac_within_0.1", "ood_frac_within_0.01", "ood_frac_within_0.001",
    "ood_coverage", "ood_target_coverage", "ood_coverage_gap",
    "ood_mean_width", "ood_sharpness_at_coverage",
    "ood_pred_var_mean", "ood_pred_sd_mean",
    "ood_var_noise_mean", "ood_var_param_mean", "ood_var_struct_mean",
    "ood_var_frac_noise", "ood_var_frac_param", "ood_var_frac_struct",
    "search_iterations", "search_candidates_total", "search_rejection_rate",
    "search_mean_accepted_per_iter", "search_iterations_to_best",
    "search_longest_plateau", "search_distinct_structures_seen",
    "search_distinct_families_seen", "search_repeated_families",
    "search_effective_models", "search_llm_calls",
    "search_llm_proposals", "search_proposals_per_llm_call",
    "search_fallback_candidates", "search_fallback_rate",
    "search_short_reply_iterations",
    "complexity_n_nodes", "complexity_n_operators", "complexity_n_binary_ops",
    "complexity_n_unary_ops", "complexity_n_functions", "complexity_n_terminals",
    "complexity_depth", "complexity_n_params", "complexity_n_variables_used",
    "complexity_n_constants", "complexity_expression_chars",
    "complexity_composite_penalty", "complexity_mdl_structure_bits",
    "complexity_mdl_param_bits", "complexity_mdl_total_bits",
    "coefficients_n_params", "coefficients_n_weak_parameters",
    "coefficients_frac_weak_parameters", "coefficients_mean_param_sd",
    "coefficients_max_param_sd", "coefficients_mean_relative_sd",
    "coefficients_max_relative_sd", "coefficients_sigma2_hat",
]

CONVERSATION_COLUMNS = [
    "instance_id", "subset", "ablation", "iteration",
    "n_accepted", "n_rejected", "n_fallback", "best_expression", "best_log_evidence",
    "proposed_expressions", "system_message", "feedback", "prompt",
    "raw_response",
]


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value != value:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def flatten_result(rec: Dict) -> Dict[str, Any]:
    """Collapse the nested JSONL record into a single spreadsheet row."""
    row: Dict[str, Any] = {}
    for key, value in rec.items():
        if key in ("conversations", "traceback", "example_rejections", "structures"):
            continue
        if key in ("test", "ood", "search", "symbolic", "complexity",
                   "coefficients", "inclusion", "scalers") and isinstance(value, dict):
            for inner, iv in value.items():
                row[f"{key}_{inner}"] = _cell(iv)
        else:
            row[key] = _cell(value)
    return row


def write_csv(path: str, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    seen = list(columns)
    extra = []
    for row in rows:
        for k in row:
            if k not in seen and k not in extra:
                extra.append(k)
    fieldnames = seen + extra
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_results_csv(records: Sequence[Dict], run_dir: str) -> str:
    path = os.path.join(run_dir, "results.csv")
    write_csv(path, [flatten_result(r) for r in records], RESULT_COLUMNS)
    return path


def write_summary_csv(overall: Dict, by_domain: Dict[str, Dict], n_failed: int, run_dir: str) -> str:
    path = os.path.join(run_dir, "summary.csv")
    rows = []
    for name, agg in list(by_domain.items()) + [("ALL", overall)]:
        row = {"domain": name, "n_failed": n_failed if name == "ALL" else ""}
        row.update(agg)
        rows.append(row)
    cols = ["domain", "n", "sa_withheld", "symbolic_accuracy", "sa_argmax",
            "sa_at_1", "sa_at_3", "sa_at_5", "sa_at_10", "sa_at_all",
            "sa_in_credible_set", "credible_set_size", "n_families",
            "sa_rank_when_found", "sa_mass_when_found",
            "test_nmse_median", "test_coverage",
            "test_mean_width", "ood_nmse_median", "ood_coverage", "ood_mean_width",
            "rejection_rate", "effective_models", "longest_plateau", "n_failed"]
    write_csv(path, rows, cols)
    return path


def flatten_conversation(instance_id: str, subset: str, ablation: str, turn: Dict) -> Dict[str, Any]:
    proposed = turn.get("proposed_expressions") or [
        c.get("expression", "") for c in turn.get("candidates") or []
    ]
    return {
        "instance_id": instance_id,
        "subset": subset,
        "ablation": ablation,
        "iteration": turn.get("iteration"),
        "n_accepted": turn.get("n_accepted"),
        "n_rejected": turn.get("n_rejected"),
        "n_fallback": turn.get("n_fallback"),
        "best_expression": turn.get("best_expression") or "",
        "best_log_evidence": turn.get("best_log_evidence") if turn.get("best_log_evidence") is not None else "",
        "proposed_expressions": " | ".join(str(x) for x in proposed),
        "system_message": turn.get("system_message") or "",
        "feedback": turn.get("feedback") or "",
        "prompt": turn.get("prompt") or "",
        "raw_response": turn.get("raw_response") or "",
    }


def write_conversations(records: Sequence[Dict], run_dir: str) -> Optional[str]:
    """Write one JSONL per problem and a combined conversations.csv."""
    conv_dir = os.path.join(run_dir, "conversations")
    for rec in records:
        log = rec.get("conversations") or []
        if not log:
            continue
        os.makedirs(conv_dir, exist_ok=True)
        path = os.path.join(conv_dir, f"{rec['instance_id']}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for turn in log:
                fh.write(json.dumps({
                    **turn,
                    "instance_id": rec.get("instance_id"),
                    "subset": rec.get("subset"),
                    "ablation": rec.get("ablation"),
                }) + "\n")

    if not os.path.isdir(conv_dir):
        return None

    rows: List[Dict[str, Any]] = []
    for fn in sorted(os.listdir(conv_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(conv_dir, fn), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                turn = json.loads(line)
                rows.append(flatten_conversation(
                    turn.get("instance_id") or fn[:-6],
                    turn.get("subset", ""),
                    turn.get("ablation", ""),
                    turn,
                ))

    if not rows:
        return None
    path = os.path.join(run_dir, "conversations.csv")
    write_csv(path, rows, CONVERSATION_COLUMNS)
    return path


def export_run(records: Sequence[Dict], run_dir: str,
               overall: Optional[Dict] = None,
               by_domain: Optional[Dict] = None,
               n_failed: int = 0) -> None:
    write_results_csv(records, run_dir)
    if overall is not None and by_domain is not None:
        write_summary_csv(overall, by_domain, n_failed, run_dir)
    write_conversations(records, run_dir)


def export_jsonl_dir(run_dir: str) -> None:
    """Rebuild CSV tables from an existing results.jsonl (no conversations)."""
    jsonl_path = os.path.join(run_dir, "results.jsonl")
    records = [json.loads(line) for line in open(jsonl_path, encoding="utf-8") if line.strip()]
    summary_path = os.path.join(run_dir, "summary.json")
    overall, by_domain, n_failed = {}, {}, 0
    if os.path.exists(summary_path):
        payload = json.load(open(summary_path, encoding="utf-8"))
        overall = payload.get("overall") or {}
        by_domain = payload.get("by_domain") or {}
        n_failed = int(payload.get("n_failed") or 0)
    export_run(records, run_dir, overall=overall, by_domain=by_domain, n_failed=n_failed)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python reporting.py <run_dir>")
    export_jsonl_dir(sys.argv[1])
