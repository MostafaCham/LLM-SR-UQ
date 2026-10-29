#!/usr/bin/env python3
"""Structural canonicalisation of expressions."""

from __future__ import annotations

import ast
import re
from typing import Dict, Sequence

from grammar import FUNCTION_NAMES, parse

_NUMBER = re.compile(r"\d+\.\d+|\d+")


_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
_COMMUTATIVE = {"+", "*"}


def _render(node: ast.AST, reserved, rename) -> str:
    """Deterministic, fully parenthesised rendering."""
    if isinstance(node, ast.Expression):
        return _render(node.body, reserved, rename)

    if isinstance(node, ast.Constant):
        v = node.value
        return repr(round(float(v), 6)) if isinstance(v, (int, float)) else repr(v)

    if isinstance(node, ast.Name):
        return node.id if node.id in reserved else rename(node.id)

    if isinstance(node, ast.UnaryOp):
        op = "-" if isinstance(node.op, ast.USub) else "+"
        return f"({op}{_render(node.operand, reserved, rename)})"

    if isinstance(node, ast.BinOp):
        op = _BINOPS[type(node.op)]
        operands = [node.left, node.right]
        if op in _COMMUTATIVE:
            operands.sort(key=lambda n: _anonymous(n, reserved))
        left, right = (_render(n, reserved, rename) for n in operands)
        return f"({left}{op}{right})"

    if isinstance(node, ast.Call):
        args = ",".join(_render(a, reserved, rename) for a in node.args)
        return f"{node.func.id}({args})"

    raise ValueError(f"unsupported node {type(node).__name__}")


def _anonymous(node: ast.AST, reserved) -> str:
    """Rendering with every parameter collapsed to ``#``."""
    return _render(node, reserved, lambda _name: "#")


def structure_signature(expr: str, var_names: Sequence[str]) -> str:
    """Canonical form; falls back to whitespace stripping if unparsable."""
    try:
        tree = parse(expr)
        reserved = set(var_names) | set(FUNCTION_NAMES)
        mapping: Dict[str, str] = {}

        def rename(name: str) -> str:
            if name not in mapping:
                mapping[name] = f"p{len(mapping)}"
            return mapping[name]

        return _render(tree.body, reserved, rename)
    except Exception:
        return re.sub(r"\s+", "", expr)


def family_signature(expr: str, var_names: Sequence[str]) -> str:
    """Coarser signature: structure with all constants replaced by wildcards."""
    sig = structure_signature(expr, var_names)
    sig = re.sub(r"\bp\d+\b", "#", sig)
    return _NUMBER.sub("#", sig)
