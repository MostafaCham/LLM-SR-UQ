#!/usr/bin/env python3
"""Parsing and canonicalisation of LLM-SRBench expressions."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

import sympy as sp

MATH_NAMES = {
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "log", "sqrt", "Abs", "abs",
    "sign", "Min", "Max", "erf", "pi", "E",
}

_CORRUPTED_COEFF = re.compile(r"\b(\d+\.?\d*)_([A-Za-z]\w*)\b")

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def repair_corrupted_coefficients(expr: str) -> str:
    """Rewrite ``<float>_<suffix>`` tokens into free coefficient symbols."""
    return _CORRUPTED_COEFF.sub(lambda m: f"c_{m.group(2)}", expr)


def defunctionalise(expr: str, variables: Iterable[str]) -> str:
    """Rewrite ``P(t)`` into ``P`` for declared state variables only."""
    for v in variables:
        expr = re.sub(rf"\b{re.escape(v)}\(t\)", v, expr)
    return expr


def parse_expression(expr: str, variables: Sequence[str]) -> sp.Expr:
    """Parse a benchmark expression into a SymPy expression."""
    text = repair_corrupted_coefficients(expr)
    text = defunctionalise(text, variables)
    names = set(_IDENTIFIER.findall(text)) - MATH_NAMES
    local = {n: sp.Symbol(n) for n in names}
    return sp.sympify(text, locals=local)


def free_constants(expr: sp.Expr, variables: Sequence[str]) -> List[str]:
    """Symbols that are not declared variables, i.e. unfitted constants."""
    return sorted({s.name for s in expr.free_symbols} - set(variables))


def is_numeric(expr: sp.Expr, variables: Sequence[str]) -> bool:
    """True when the expression can be evaluated from the variables alone."""
    return not free_constants(expr, variables)


def abstract_constants(expr: sp.Expr, variables: Sequence[str]) -> sp.Expr:
    """Replace numeric literals and free constants with canonical placeholders."""
    var_syms = {sp.Symbol(v) for v in variables}
    counter = [0]

    def placeholder() -> sp.Symbol:
        counter[0] += 1
        return sp.Symbol(f"C{counter[0]}")

    def walk(node: sp.Expr) -> sp.Expr:
        if node.is_Number:
            if node.is_Integer and abs(int(node)) <= 4:
                return node
            return placeholder()
        if node.is_Symbol:
            return node if node in var_syms else placeholder()
        if not node.args:
            return node
        if node.is_Pow:
            base, exponent = node.args
            new_exp = exponent if exponent.is_Integer and abs(int(exponent)) <= 4 else walk(exponent)
            return sp.Pow(walk(base), new_exp, evaluate=False)
        return node.func(*[walk(a) for a in node.args], evaluate=False)

    return walk(expr)


def lambdify_expression(expr: sp.Expr, variables: Sequence[str]):
    """Compile a SymPy expression into a NumPy callable over ``variables``."""
    return sp.lambdify([sp.Symbol(v) for v in variables], expr, "numpy")


def substitute_constants(expr: sp.Expr, values: Dict[str, float]) -> sp.Expr:
    return expr.subs({sp.Symbol(k): v for k, v in values.items()})


def describe(expr: str, variables: Sequence[str]) -> Optional[str]:
    """Best-effort human-readable normal form; ``None`` if unparsable."""
    try:
        return str(parse_expression(expr, variables))
    except Exception:
        return None
