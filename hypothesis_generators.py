#!/usr/bin/env python3
"""Hypothesis generators."""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Sequence

from equation_program import EquationProgram
from grammar import ExpressionError


class GenerationResult:
    """Candidates plus the record of what was thrown away and why."""

    def __init__(self):
        self.programs: List[EquationProgram] = []
        self.sources: List[str] = []
        self.raw: List[str] = []
        self.invalid: List[tuple] = []
        self.raw_response: Optional[str] = None
        self.system_message: Optional[str] = None

    def __len__(self) -> int:
        return len(self.programs)

    @property
    def n_fallback(self) -> int:
        return sum(1 for s in self.sources if s == "fallback")


def _validate(exprs: Sequence[str], var_names: Sequence[str], max_params: Optional[int],
              source: str = "llm") -> GenerationResult:
    out = GenerationResult()
    out.raw = list(exprs)
    for e in exprs:
        try:
            out.programs.append(EquationProgram.from_expression(e, var_names, max_params=max_params))
            out.sources.append(source)
        except (ExpressionError, Exception) as exc:
            out.invalid.append((e, str(exc)))
    return out


class HypothesisGenerator:
    def generate(self, prompt: str, n: int, var_names: Sequence[str]) -> GenerationResult:
        raise NotImplementedError


class LocalHeuristicGenerator(HypothesisGenerator):
    """LLM-free baseline. Also the fallback when an API call returns too little."""

    TEMPLATES = [
        "c0*{0} + c1",
        "c0*{0} + c1*{1} + c2",
        "c0*{0}**2 + c1*{0} + c2",
        "c0*exp(c1*{0}) + c2",
        "c0*log(1 + abs(c1*{0})) + c2",
        "c0*sin(c1*{0}) + c2",
        "c0*tanh(c1*{0}) + c2",
        "c0*{0}*{1} + c1*{0} + c2*{1} + c3",
        "c0*{0}/(c1 + abs({1})) + c2",
        "c0*abs({0})**c1 + c2",
        "c0*{0}*(1 - {0}/c1) + c2",
        "c0*{0} + c1*{1}**2 + c2*{0}*{1} + c3",
    ]

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def generate(self, prompt: str, n: int, var_names: Sequence[str],
                 source: str = "heuristic") -> GenerationResult:
        var_names = list(var_names)
        exprs = []
        for _ in range(n):
            tpl = self.rng.choice(self.TEMPLATES)
            slots = len(set(re.findall(r"\{(\d+)\}", tpl)))
            if slots <= len(var_names):
                picks = self.rng.sample(var_names, slots)
            else:
                picks = [self.rng.choice(var_names) for _ in range(slots)]
            exprs.append(tpl.format(*picks))
        return _validate(exprs, var_names, max_params=12, source=source)


try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


class OpenAIHypothesisGenerator(HypothesisGenerator):
    """LLM-backed generator returning a JSON list of expressions."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.9,
        max_retries: int = 3,
        request_timeout: int = 90,
        fallback: bool = False,
        seed: int = 0,
        base_url: Optional[str] = None,
    ):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("the openai package is not installed")
        self.base_url = base_url or None
        if self.base_url:
            key = api_key or os.getenv("LOCAL_LLM_API_KEY") or "local"
            self.client = OpenAI(api_key=key, base_url=self.base_url)
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("set OPENAI_API_KEY or pass api_key=...")
            self.client = OpenAI(api_key=key)
        self.model = model
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.request_timeout = int(request_timeout)
        self.fallback = bool(fallback)
        self._fallback_gen = LocalHeuristicGenerator(seed=seed)

    def system_message(self, n: int, var_names: Sequence[str]) -> str:
        return (
            "You propose candidate symbolic-regression expressions for a scientific "
            "equation-discovery task.\n"
            f"Return JSON with a single key 'expressions': a list of exactly {n} strings.\n\n"
            f"Variables — use ONLY these, spelled exactly: {', '.join(var_names)}.\n"
            "Coefficients — every fitted constant must be named c0, c1, c2, ... "
            "Numbering restarts at c0 in each expression.\n"
            "Any identifier that is neither a listed variable nor a c<number> "
            "coefficient is rejected, so do not invent symbol names.\n\n"
            "Allowed functions: sin cos tan tanh sinh cosh exp log sqrt abs sign "
            "softplus minimum maximum.\n"
            "Write exponentiation as ** (Python syntax), never ^. Give the "
            "right-hand side only, with no 'y =' prefix.\n"
            "Use softplus(...) where a quantity must stay positive, such as the base "
            "of a fractional power or the argument of a logarithm.\n\n"
            "Example of the required form: c0*exp(-c1*t) + c2\n"
            "Return JSON only, with no prose and no comments."
        )

    def _response_format(self, n: int) -> Dict:
        """The JSON schema, in whichever dialect the endpoint accepts."""
        schema = {
            "type": "object",
            "properties": {
                "expressions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": n,
                    "maxItems": n,
                }
            },
            "required": ["expressions"],
            "additionalProperties": False,
        }
        body = {"name": "sr_expressions", "schema": schema}
        if not self.base_url:
            body["strict"] = True
        return {"type": "json_schema", "json_schema": body}

    def _call(self, messages, n: int) -> List[str]:
        delay = 2.0
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format=self._response_format(n),
                    timeout=self.request_timeout,
                )
                content = resp.choices[0].message.content
                self._last_raw = content
                items = json.loads(content)["expressions"]
                seen, uniq = set(), []
                for s in items:
                    s = str(s).strip()
                    if s and s not in seen:
                        seen.add(s)
                        uniq.append(s)
                return uniq[:n]
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 1.8
        raise RuntimeError(f"LLM call failed: {last_exc}")

    def generate(self, prompt: str, n: int, var_names: Sequence[str]) -> GenerationResult:
        self._last_raw = None
        messages = [
            {"role": "system", "content": self.system_message(n, var_names)},
            {"role": "user", "content": prompt},
        ]
        exprs = self._call(messages, n)
        result = _validate(exprs, var_names, max_params=12)
        result.raw_response = self._last_raw
        result.system_message = messages[0]["content"]

        if self.fallback and len(result) < max(2, n // 2):
            topped_up = self._fallback_gen.generate(
                prompt, n - len(result), var_names, source="fallback")
            result.programs.extend(topped_up.programs)
            result.sources.extend(topped_up.sources)
        return result
