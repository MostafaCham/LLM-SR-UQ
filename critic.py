#!/usr/bin/env python3
"""LLM critic used by rungs A2 and A3."""

from __future__ import annotations

import ast
import json
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bma_conformal import NormalizedConformal, StructuralConformal
from canonical import family_signature
from diagnostics import MAX_CHARS, STAGNATION_K, TOP_K, A1Feedback, _additive_terms
from equation_program import EquationProgram

CRITIC_EVERY = 5
MAX_EDITS = 3
MAX_DIRECTIVES = 3
MAX_BANNED = 5
UNCONDITIONED_SLOTS = 2
MAX_INSTRUCTION_CHARS = 300
EDIT_ACTIONS = ("delete_term", "merge_coefficients", "replace_term")
BLOB_MAX_CHARS = 4500
REPLY_MAX_TOKENS = 700
PROBE_EXTEND = 0.25
PROBE_POINTS = 1000
RESIDUAL_ROWS = 2000


def budget_split(total_calls: int, every: int = CRITIC_EVERY) -> Tuple[int, List[int]]:
    """``(proposer_rounds, rounds_preceded_by_a_critic_call)`` for a budget of LLM calls."""
    proposer, before = 0, []
    for i in range(1, total_calls + 1):
        if every and i % every == 0 and i < total_calls:
            before.append(proposer + 1)
        else:
            proposer += 1
    return proposer, before


def extra_split(proposer_rounds: int, every: int = CRITIC_EVERY) -> Tuple[int, List[int]]:
    """``budget_split``'s return, with the critic's calls added on top of ``proposer_rounds``."""
    before = list(range(every + 1, proposer_rounds + 1, every)) if every else []
    return proposer_rounds, before


_PATH = re.compile(r"^[A-Za-z_]\w*(\[(\d+|[A-Za-z_]\w*)\])*(\.[A-Za-z_]\w*(\[(\d+|[A-Za-z_]\w*)\])*)*$")


def resolve(blob: Dict, path) -> bool:
    """True if ``path`` names a non-empty value in ``blob``."""
    if not isinstance(path, str) or not _PATH.match(path.strip()):
        return False
    cur = blob
    for part in path.strip().split("."):
        name = re.match(r"^[A-Za-z_]\w*", part).group(0)
        if not isinstance(cur, dict) or name not in cur:
            return False
        cur = cur[name]
        for idx in re.findall(r"\[([^\]]+)\]", part):
            if idx.isdigit():
                i = int(idx)
                if not isinstance(cur, list) or i >= len(cur):
                    return False
                cur = cur[i]
            else:
                if not isinstance(cur, dict) or idx not in cur:
                    return False
                cur = cur[idx]
    return cur is not None and not (isinstance(cur, (list, dict, str)) and len(cur) == 0)


def _round_floats(obj):
    if isinstance(obj, float):
        return float(f"{obj:.4g}") if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def _dump(blob: Dict) -> str:
    return json.dumps(blob, separators=(",", ":"))


def shrink(blob: Dict, max_chars: int = BLOB_MAX_CHARS) -> Dict:
    """Drop trailing list elements, longest list first, until the blob fits."""
    blob = json.loads(_dump(blob))

    def lists(node, out):
        if isinstance(node, dict):
            for v in node.values():
                lists(v, out)
        elif isinstance(node, list):
            if len(node) > 1:
                out.append(node)
            for v in node:
                lists(v, out)
        return out

    while len(_dump(blob)) > max_chars:
        candidates = lists(blob, [])
        if not candidates:
            break
        max(candidates, key=lambda l: len(_dump(l))).pop()
    return blob


def probe_box(X: np.ndarray, extend: float = PROBE_EXTEND, n: int = PROBE_POINTS, seed: int = 0) -> np.ndarray:
    lo, hi = X.min(0), X.max(0)
    width = hi - lo
    new_lo = np.where(lo >= 0, np.maximum(lo - extend * width, 0.0), lo - extend * width)
    new_hi = np.where(hi <= 0, np.minimum(hi + extend * width, 0.0), hi + extend * width)
    return np.random.default_rng(seed).uniform(new_lo, new_hi, size=(n, X.shape[1]))


def calibration_signals(pipe, var_names: Sequence[str], top_k: int = TOP_K) -> Dict:
    """Label-free calibration evidence for A3; no OOD or test labels are read."""
    X, y = getattr(pipe, "_Xtr", None), getattr(pipe, "_ytr", None)
    Xca, yca = getattr(pipe, "_Xca", None), getattr(pipe, "_yca", None)
    if X is None or y is None or not len(pipe.buffer):
        return {}
    if len(y) > RESIDUAL_ROWS:
        step = int(np.ceil(len(y) / RESIDUAL_ROWS))
        X, y = X[::step], y[::step]
    probe = probe_box(np.asarray(getattr(pipe, "_Xtr")), seed=0)
    y_var = max(float(np.var(y)), 1e-300)

    preds = []
    for s in pipe.posterior_structures()[:top_k]:
        try:
            prog = EquationProgram.from_expression(s["best_expression"], list(var_names), max_params=40)
            with np.errstate(all="ignore"):
                preds.append((int(s["rank"]), np.asarray(prog.predict(X, s["theta"]), float),
                              np.asarray(prog.predict(probe, s["theta"]), float)))
        except Exception:
            continue

    def nmse_between(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.mean((a[ok] - b[ok]) ** 2) / y_var) if ok.mean() >= 0.5 else None

    out: Dict = {
        "probe": f"inputs up to {PROBE_EXTEND:g} training-range widths beyond the data; no labels used",
        "disagreement": [
            {"ranks": [ra, rb], "train_nmse": nmse_between(ta, tb), "probe_nmse": nmse_between(pa, pb)}
            for i, (ra, ta, pa) in enumerate(preds) for (rb, tb, pb) in preds[i + 1:]
        ],
    }
    if Xca is not None and yca is not None and len(yca):
        cls = type(pipe.conformal) if isinstance(pipe.conformal, NormalizedConformal) else StructuralConformal
        comp_cal = pipe.bma.predict_components(Xca)
        comp_probe = pipe.bma.predict_components(probe)
        s_cal, s_probe = cls.sigma(comp_cal), cls.sigma(comp_probe)
        conf = cls(alpha=getattr(pipe.cfg, "alpha", 0.1))
        conf.fit_normalized(yca, comp_cal["mean"], s_cal)
        sd = float(np.sqrt(y_var))
        out["sigma_growth"] = float(np.median(s_probe) / max(float(np.median(s_cal)), 1e-300))
        if conf.q is not None and np.isfinite(conf.q):
            out["interval_halfwidth_rel"] = {"calibration": float(np.median(conf.q * s_cal) / sd),
                                             "probe": float(np.median(conf.q * s_probe) / sd)}
    return out


def system_message(var_names: Sequence[str], batch: int, calibration: bool) -> str:
    conditioned = batch - UNCONDITIONED_SLOTS
    text = (
        "You are the critic in an equation-discovery loop. A proposer writes candidate expressions; "
        "each is fitted to the data and scored by Bayesian evidence, and only the evidence decides what "
        "is kept. You never decide what is kept. You read the evidence JSON and write edits and "
        "directives for the proposer.\n\n"
        "Rules.\n"
        "1. Ground everything. Every edit and directive names an evidence_field: a path into the evidence "
        "JSON, with dots for keys and [n] for list positions, such as weak_coefficients[0], "
        "residual.patterns[1], variables.inclusion.t or stagnation. An item whose evidence_field does not "
        "exist, or points to an empty value, is discarded.\n"
        f"2. Edits (at most {MAX_EDITS}) are complete rewrites of one of top_structures, identified by "
        "of_rank, and are fitted like any proposal. The expression is always the WHOLE structure after the "
        "change — never a single term, and never the structure unchanged. Actions:\n"
        "   delete_term: remove one term whose coefficient the data does not determine (two, if the structure "
        "has four or more terms), or remove the part of one term that holds such a coefficient "
        "(c0*P/(c1 + c2*t) becomes c0*P when weak_coefficients names c1); keep everything else, and keep the "
        "names of the coefficients you keep.\n"
        "   merge_coefficients: rewrite with fewer coefficients where several only act together "
        "(c0*c1*P becomes c0*P); keep the same inputs.\n"
        "   replace_term: swap exactly one term the residual shows is misshaped for a better one; keep the "
        "other terms.\n"
        "   Example: top structure 1 is c0*P + c1*P**2 + c2*sin(c3*t) and weak_coefficients[0] names c2 in "
        "c2*sin(c3*t). A valid edit is {\"of_rank\": 1, \"action\": \"delete_term\", \"expression\": "
        "\"c0*P + c1*P**2\", \"evidence_field\": \"weak_coefficients[0]\"}.\n"
        "   An edit that does not match its action is discarded.\n"
        f"3. Directives (2 or {MAX_DIRECTIVES}) steer the proposer's next rounds. Make them pursue "
        f"different hypotheses. Each takes n_slots of the batch of {batch}; directive slots plus edits may "
        f"not exceed {conditioned}, so at least {UNCONDITIONED_SLOTS} proposals stay unconditioned.\n"
        f"4. banned_families (at most {MAX_BANNED}): expressions copied from repeated_families that the "
        "proposer should stop proposing. Nothing else may be banned.\n"
        "5. mode: \"exploit\" to refine the current best structures; \"explore\" when the search has "
        "stagnated or the best structures look physically implausible.\n"
    )
    if calibration:
        text += (
            "6. The evidence has a calibration block, computed without any held-out labels: how far the top "
            "structures' predictions diverge beyond the training inputs, and how much the prediction interval "
            "widens there. The goal is structures that extrapolate with narrow, honest intervals. Prefer edits "
            "and directives that would make the top structures agree beyond the data — forms that "
            "discriminate between them, or physically motivated forms — and never propose anything whose only "
            "effect would be a wider interval.\n"
        )
    else:
        text += ("6. The goal is a law, not a curve fit: compact, physically plausible forms that fit the "
                 "data.\n")
    text += (
        "\nExpressions: use only the variables " + ", ".join(var_names) + ", and coefficients named "
        "c0, c1, c2, ... Allowed functions: sin cos tan tanh sinh cosh exp log sqrt abs sign softplus "
        "minimum maximum. Write powers as **. Right-hand side only.\n"
        "Return JSON only."
    )
    return text


def response_format(strict: bool) -> Dict:
    """JSON schema for the critic reply. Caps are enforced in ``validate``."""
    edit = {"type": "object",
            "properties": {"of_rank": {"type": "integer"},
                           "action": {"type": "string", "enum": list(EDIT_ACTIONS)},
                           "expression": {"type": "string"},
                           "evidence_field": {"type": "string"}},
            "required": ["of_rank", "action", "expression", "evidence_field"],
            "additionalProperties": False}
    directive = {"type": "object",
                 "properties": {"instruction": {"type": "string"},
                                "evidence_field": {"type": "string"},
                                "n_slots": {"type": "integer"}},
                 "required": ["instruction", "evidence_field", "n_slots"],
                 "additionalProperties": False}
    schema = {"type": "object",
              "properties": {"mode": {"type": "string", "enum": ["explore", "exploit"]},
                             "edits": {"type": "array", "items": edit},
                             "directives": {"type": "array", "items": directive},
                             "banned_families": {"type": "array", "items": {"type": "string"}}},
              "required": ["mode", "edits", "directives", "banned_families"],
              "additionalProperties": False}
    body = {"name": "critic", "schema": schema}
    if strict:
        body["strict"] = True
    return {"type": "json_schema", "json_schema": body}


class CriticClient:
    """Chat-completions call for the critic, at temperature 0 with a fixed seed."""

    def __init__(self, model: str, base_url: Optional[str] = None, temperature: float = 0.0,
                 seed: int = 0, max_retries: int = 5, request_timeout: int = 300,
                 max_tokens: int = REPLY_MAX_TOKENS):
        from openai import OpenAI

        self.base_url = base_url or None
        if self.base_url:
            self.client = OpenAI(api_key=os.getenv("LOCAL_LLM_API_KEY") or "local", base_url=self.base_url)
        else:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("the hosted critic needs OPENAI_API_KEY")
            self.client = OpenAI(api_key=key)
        self.model = model
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_retries = int(max_retries)
        self.request_timeout = int(request_timeout)
        self.max_tokens = int(max_tokens)

    def __call__(self, system: str, user: str) -> Dict:
        delay, last = 2.0, None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, seed=self.seed,
                    max_tokens=self.max_tokens, timeout=self.request_timeout,
                    response_format=response_format(strict=not self.base_url),
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                usage = getattr(resp, "usage", None)
                return {"content": resp.choices[0].message.content,
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "attempts": attempt + 1}
            except Exception as exc:
                last = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(delay)
                delay *= 1.8
        raise RuntimeError(f"critic call failed: {last}")


def _term_families(expr: str, var_names: Sequence[str]) -> Counter:
    """Top-level additive terms as a multiset of families (constants wildcarded, sign dropped)."""
    out: Counter = Counter()
    for term in _additive_terms(expr):
        out[family_signature(ast.unparse(term), list(var_names)).replace("(-#)", "#")] += 1
    return out


_COEF = re.compile(r"c\d+")


def _atoms(node: ast.AST, var_names: Sequence[str]) -> Counter:
    """Everything in a term except its coefficients: operators, functions, variables, numbers."""
    out: Counter = Counter()
    for n in ast.walk(node):
        if isinstance(n, (ast.BinOp, ast.UnaryOp)):
            out[("op", type(n.op).__name__)] += 1
        elif isinstance(n, ast.Call):
            out[("call", ast.unparse(n.func))] += 1
        elif isinstance(n, ast.Name) and n.id in var_names:
            out[("var", n.id)] += 1
        elif isinstance(n, ast.Constant):
            out[("num", repr(n.value))] += 1
    return out


def _coefs(node: ast.AST) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and _COEF.fullmatch(n.id)}


def _factor_deletion(original: str, edit: str, var_names: Sequence[str], weak: set) -> bool:
    """True if ``edit`` simplifies exactly one term of ``original`` by removing a weak coefficient's part."""
    vars_ = list(var_names)
    fam = lambda t: family_signature(ast.unparse(t), vars_).replace("(-#)", "#")
    rest_o = list(_additive_terms(original))
    rest_e = []
    for t in _additive_terms(edit):
        i = next((k for k, u in enumerate(rest_o) if fam(u) == fam(t)), None)
        if i is None:
            rest_e.append(t)
        else:
            rest_o.pop(i)
    if len(rest_o) != 1 or len(rest_e) != 1:
        return False
    to, te = rest_o[0], rest_e[0]
    ao, ae = _atoms(to, vars_), _atoms(te, vars_)
    co, ce = _coefs(to), _coefs(te)
    return (ae != ao and all(ae[k] <= ao[k] for k in ae)
            and ce < co and bool((co - ce) & set(weak)))


def action_mismatch(action: str, original: str, edit: str, var_names: Sequence[str],
                    weak: Sequence[str] = ()) -> Optional[str]:
    """Why an edit does not do what its action says, or None."""
    o, e = _term_families(original, var_names), _term_families(edit, var_names)
    removed, added = sum((o - e).values()), sum((e - o).values())
    if action == "delete_term":
        max_removed = 2 if sum(o.values()) >= 4 else 1
        additive_ok = not added and 1 <= removed <= max_removed and bool(e)
        if not additive_ok and not _factor_deletion(original, edit, var_names, set(weak)):
            return ("delete_term must remove one term (two from four or more), or a weak coefficient's "
                    "factor inside one term, and keep the rest")
    elif action == "merge_coefficients":
        po = EquationProgram.from_expression(original, list(var_names), max_params=40)
        pe = EquationProgram.from_expression(edit, list(var_names), max_params=40)
        if set(po.unused_variables) != set(pe.unused_variables) or pe.n_params() >= po.n_params():
            return "merge_coefficients must keep the inputs and use fewer coefficients"
    elif action == "replace_term":
        if removed > 1 or added > 2:
            return "replace_term may swap out at most one term"
    return None


def validate(raw: str, blob: Dict, var_names: Sequence[str], batch: int) -> Tuple[Optional[Dict], Dict]:
    """``(guidance, report)``. Guidance is None when the reply cannot be used at all."""
    report: Dict = {"parsed": False, "dropped": [],
                    "edits_proposed": 0, "edits_valid": 0,
                    "directives_proposed": 0, "directives_valid": 0, "slots_trimmed": 0,
                    "banned_proposed": 0, "banned_valid": 0, "ungrounded": 0}

    def drop(kind, item, why):
        report["dropped"].append({"kind": kind, "item": item, "why": why})
        if why == "ungrounded":
            report["ungrounded"] += 1

    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("not an object")
    except Exception as exc:
        report["error"] = f"unparseable reply: {exc}"
        return None, report
    report["parsed"] = True

    top = blob.get("top_structures") or []
    top_sigs = set()
    for s in top:
        try:
            top_sigs.add(EquationProgram.from_expression(s["expression"], list(var_names), max_params=40).structure_signature())
        except Exception:
            pass

    edits, seen = [], set()
    raw_edits = obj.get("edits") if isinstance(obj.get("edits"), list) else []
    report["edits_proposed"] = len(raw_edits)
    for e in raw_edits:
        if not isinstance(e, dict):
            drop("edit", e, "not an object")
            continue
        if not resolve(blob, e.get("evidence_field")):
            drop("edit", e, "ungrounded")
            continue
        rank = e.get("of_rank")
        if not isinstance(rank, int) or not 1 <= rank <= len(top):
            drop("edit", e, "of_rank is not a top structure")
            continue
        if e.get("action") not in EDIT_ACTIONS:
            drop("edit", e, "unknown action")
            continue
        try:
            prog = EquationProgram.from_expression(str(e.get("expression", "")), list(var_names), max_params=12)
        except Exception as exc:
            drop("edit", e, f"does not parse: {str(exc)[:80]}")
            continue
        sig = prog.structure_signature()
        if sig in top_sigs:
            drop("edit", e, "identical to a top structure")
            continue
        try:
            weak = {w.get("coefficient") for w in blob.get("weak_coefficients") or [] if w.get("rank") == rank}
            why = action_mismatch(e["action"], top[rank - 1]["expression"], prog.expr, var_names, weak)
        except Exception as exc:
            why = f"cannot compare with the structure it edits ({str(exc)[:60]})"
        if why:
            drop("edit", e, f"action mismatch: {why}")
            continue
        if sig in seen:
            drop("edit", e, "repeats another edit")
            continue
        if len(edits) >= MAX_EDITS:
            drop("edit", e, "over the edit cap")
            continue
        seen.add(sig)
        edits.append({"of_rank": rank, "action": e["action"], "expression": prog.expr,
                      "evidence_field": e["evidence_field"]})
    report["edits_valid"] = len(edits)

    directives = []
    slots_left = max(batch - UNCONDITIONED_SLOTS - len(edits), 0)
    raw_dirs = obj.get("directives") if isinstance(obj.get("directives"), list) else []
    report["directives_proposed"] = len(raw_dirs)
    for d in raw_dirs:
        if not isinstance(d, dict):
            drop("directive", d, "not an object")
            continue
        if not resolve(blob, d.get("evidence_field")):
            drop("directive", d, "ungrounded")
            continue
        text = str(d.get("instruction", "")).strip()
        n = d.get("n_slots")
        if not text or not isinstance(n, int) or n < 1:
            drop("directive", d, "empty instruction or invalid n_slots")
            continue
        if len(directives) >= MAX_DIRECTIVES:
            drop("directive", d, "over the directive cap")
            continue
        if slots_left == 0:
            drop("directive", d, "no slots left")
            continue
        take = min(n, slots_left)
        report["slots_trimmed"] += n - take
        slots_left -= take
        directives.append({"instruction": text[:MAX_INSTRUCTION_CHARS], "n_slots": take,
                           "evidence_field": d["evidence_field"]})
    report["directives_valid"] = len(directives)

    repeated = {}
    for r in blob.get("repeated_families") or []:
        try:
            repeated[family_signature(r["expression"], list(var_names))] = r["expression"]
        except Exception:
            continue
    banned, banned_fams = [], set()
    raw_banned = obj.get("banned_families") if isinstance(obj.get("banned_families"), list) else []
    report["banned_proposed"] = len(raw_banned)
    for b in raw_banned:
        try:
            fam = family_signature(str(b), list(var_names))
        except Exception:
            drop("banned", b, "does not parse")
            continue
        if fam not in repeated:
            drop("banned", b, "ungrounded")
            continue
        if fam in banned_fams or len(banned) >= MAX_BANNED:
            drop("banned", b, "duplicate or over the cap")
            continue
        banned_fams.add(fam)
        banned.append(repeated[fam])
    report["banned_valid"] = len(banned)

    mode = obj.get("mode") if obj.get("mode") in ("explore", "exploit") else None
    if mode is None:
        drop("mode", obj.get("mode"), "not explore or exploit")
    guidance = {"mode": mode, "edits": edits, "directives": directives, "banned_families": banned,
                "banned_signatures": sorted(banned_fams)}
    return guidance, report


def render_guidance(guidance: Optional[Dict], n_proposals: int, edits_this_round: Sequence[str]) -> str:
    """The critic's block in the proposer prompt. Directives are addressed by position in the reply."""
    if not guidance:
        return ""
    lines = ["Critic guidance:"]
    if guidance.get("mode") == "explore":
        lines.append("The search should explore: prefer structurally new forms over variations of the best ones.")
    elif guidance.get("mode") == "exploit":
        lines.append("The search should exploit: refine the best structures rather than start over.")
    if edits_this_round:
        lines.append("These rewrites are already being fitted this round; do not repeat them:")
        lines += [f"  - {e}" for e in edits_this_round]
    dirs = guidance.get("directives") or []
    if dirs:
        used = 0
        lines.append(f"Write your {n_proposals} expressions in this order:")
        for d in dirs:
            k = min(d["n_slots"], max(n_proposals - used, 0))
            if k == 0:
                break
            lines.append(f"  - {k} expression{'s' if k > 1 else ''}: {d['instruction']}")
            used += k
        if n_proposals - used > 0:
            lines.append(f"  - the remaining {n_proposals - used}: your own choice.")
    if guidance.get("banned_families"):
        lines.append("Do not propose these families again:")
        lines += [f"  - {b}" for b in guidance["banned_families"]]
    return "\n".join(lines) if len(lines) > 1 else ""


class CriticFeedback:
    """``feedback_fn`` for A2 (``calibration=False``) and A3 (``calibration=True``)."""

    def __init__(self, var_names: Sequence[str], client, base_prompt: str, batch: int,
                 critic_rounds: Sequence[int], calibration: bool = False,
                 top_k: int = TOP_K, stagnation_k: int = STAGNATION_K, max_chars: int = MAX_CHARS):
        self.var_names = list(var_names)
        self.client = client
        self.base_prompt = base_prompt
        self.batch = int(batch)
        self.critic_rounds = sorted(set(int(r) for r in critic_rounds))
        self.calibration = bool(calibration)
        self.top_k = top_k
        self.a1 = A1Feedback(var_names, top_k, stagnation_k, max_chars)
        self.guidance: Optional[Dict] = None
        self.log: List[Dict] = []
        self.system = system_message(self.var_names, self.batch, self.calibration)

    def blob(self, pipe, diag: Optional[Dict]) -> Dict:
        blob = dict(diag or {})
        if self.calibration:
            try:
                blob["calibration"] = calibration_signals(pipe, self.var_names, self.top_k)
            except Exception as exc:
                blob["calibration"] = {"unavailable": str(exc)[:120]}
        return shrink(_round_floats(blob))

    def consult(self, pipe, diag: Optional[Dict], before_round: int) -> Dict:
        blob = self.blob(pipe, diag)
        user = f"{self.base_prompt}\n\nEvidence:\n{_dump(blob)}"
        entry: Dict = {"before_round": before_round, "blob": blob, "user_chars": len(user)}
        try:
            reply = self.client(self.system, user)
            entry.update({k: reply.get(k) for k in ("content", "prompt_tokens", "completion_tokens", "attempts")})
            guidance, report = validate(reply.get("content") or "", blob, self.var_names, self.batch)
        except Exception as exc:
            guidance, report = None, {"parsed": False, "error": str(exc)[:300]}
        entry.update({"guidance": guidance, "validation": report})
        self.log.append(entry)
        return entry

    def __call__(self, pipe, history: Sequence) -> Dict:
        text, diag = self.a1(pipe, history)
        round_no = len(history) + 1
        edits: List[str] = []
        consulted = False
        if history and round_no in self.critic_rounds:
            self.guidance = self.consult(pipe, diag, round_no)["guidance"]
            consulted = True
            edits = [e["expression"] for e in (self.guidance or {}).get("edits", [])]
        block = render_guidance(self.guidance, self.batch - len(edits), edits)
        full = "\n\n".join(t for t in (text, block) if t)
        return {"text": full, "inject": edits,
                "diagnostics": {"a1": diag, "critic_consulted": consulted,
                                "guidance_from_round": self.log[-1]["before_round"] if self.log and self.guidance else None}}

    def summary(self, history: Sequence) -> Dict:
        """What the critic proposed, what passed validation, and what it changed."""
        v = [e["validation"] for e in self.log]
        tot = lambda k: int(sum(r.get(k, 0) for r in v))
        edit_cands = [c for r in history for c in r.candidates if c.source == "critic_edit"]
        in_force, violations = 0, 0
        bounds = [e["before_round"] for e in self.log] + [len(history) + 1]
        for entry, start, end in zip(self.log, bounds[:-1], bounds[1:]):
            fams = set((entry.get("guidance") or {}).get("banned_signatures") or [])
            if not fams:
                continue
            for rec in history[start - 1:end - 1]:
                for c in rec.candidates:
                    if c.source != "llm":
                        continue
                    in_force += 1
                    try:
                        violations += family_signature(c.expression, self.var_names) in fams
                    except Exception:
                        pass
        return {
            "critic_calls": len(self.log),
            "critic_failed_calls": sum(1 for e in self.log if not e["validation"].get("parsed")),
            "critic_prompt_tokens": int(sum(e.get("prompt_tokens") or 0 for e in self.log)),
            "critic_completion_tokens": int(sum(e.get("completion_tokens") or 0 for e in self.log)),
            "edits_proposed": tot("edits_proposed"), "edits_valid": tot("edits_valid"),
            "edits_fitted": sum(1 for c in edit_cands if c.accepted),
            "edits_entered_buffer": sum(1 for c in edit_cands if c.entered_buffer),
            "directives_proposed": tot("directives_proposed"), "directives_valid": tot("directives_valid"),
            "slots_trimmed": tot("slots_trimmed"),
            "banned_proposed": tot("banned_proposed"), "banned_valid": tot("banned_valid"),
            "ungrounded": tot("ungrounded"),
            "ungrounded_rate": tot("ungrounded") / max(tot("edits_proposed") + tot("directives_proposed")
                                                       + tot("banned_proposed"), 1),
            "proposals_under_ban": in_force,
            "banned_family_violations": violations,
            "explore_calls": sum(1 for e in self.log if (e.get("guidance") or {}).get("mode") == "explore"),
        }
