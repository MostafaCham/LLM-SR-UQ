#!/usr/bin/env python3
"""Simplicity metrics for a discovered expression."""

from __future__ import annotations

import ast
import math
from typing import Dict, Optional, Sequence

from grammar import COEFFICIENT_PATTERN, FUNCTION_NAMES

_COEFF = COEFFICIENT_PATTERN

_BINOP = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
    ast.Div: "/", ast.Pow: "**",
}


def _tree(expr: str) -> Optional[ast.AST]:
    try:
        return ast.parse(str(expr).strip(), mode="eval").body
    except SyntaxError:
        return None


def expression_size(expr: str) -> Dict[str, float]:
    """Node, operator and depth counts for one expression."""
    tree = _tree(expr)
    if tree is None:
        return {k: float("inf") for k in
                ("n_nodes", "n_operators", "n_binary_ops", "n_unary_ops",
                 "n_functions", "n_terminals", "depth")}

    counts = dict(n_nodes=0, n_binary_ops=0, n_unary_ops=0,
                  n_functions=0, n_terminals=0)

    def walk(node: ast.AST, depth: int) -> int:
        counts["n_nodes"] += 1
        if isinstance(node, ast.BinOp):
            counts["n_binary_ops"] += 1
            return max(walk(node.left, depth + 1), walk(node.right, depth + 1))
        if isinstance(node, ast.UnaryOp):
            counts["n_unary_ops"] += 1
            return walk(node.operand, depth + 1)
        if isinstance(node, ast.Call):
            counts["n_functions"] += 1
            sub = [walk(a, depth + 1) for a in node.args]
            return max(sub) if sub else depth
        counts["n_terminals"] += 1
        return depth

    depth = walk(tree, 1)
    counts["n_operators"] = (counts["n_binary_ops"] + counts["n_unary_ops"]
                             + counts["n_functions"])
    counts["depth"] = depth
    return {k: float(v) for k, v in counts.items()}


def symbol_counts(expr: str, variables: Sequence[str]) -> Dict[str, float]:
    """Parameters, distinct variables used, and numeric literals."""
    tree = _tree(expr)
    if tree is None:
        return {"n_params": float("inf"), "n_variables_used": 0.0,
                "n_constants": float("inf")}

    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - called
    literals = sum(1 for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)))

    return {
        "n_params": float(len({n for n in names if _COEFF.match(n)})),
        "n_variables_used": float(len(names & set(variables))),
        "n_constants": float(literals),
    }


def description_length_bits(expr: str, variables: Sequence[str],
                            n_data: Optional[int] = None) -> Dict[str, float]:
    """Two-part MDL: bits to write the structure, plus bits for its constants."""
    size = expression_size(expr)
    syms = symbol_counts(expr, variables)
    if not math.isfinite(size["n_nodes"]):
        return {"mdl_structure_bits": float("inf"), "mdl_param_bits": float("inf"),
                "mdl_total_bits": float("inf")}

    alphabet = (len(_BINOP) + 2
                + len(FUNCTION_NAMES)
                + len(variables) + 1)
    structure_bits = size["n_nodes"] * math.log2(max(alphabet, 2))

    k = syms["n_params"]
    param_bits = 0.0
    if n_data and n_data > 1 and math.isfinite(k):
        param_bits = 0.5 * k * math.log2(n_data)

    return {
        "mdl_structure_bits": float(structure_bits),
        "mdl_param_bits": float(param_bits),
        "mdl_total_bits": float(structure_bits + param_bits),
    }


def complexity_report(expr: str, variables: Sequence[str],
                      n_data: Optional[int] = None,
                      complexity_penalty: float = 0.05) -> Dict[str, float]:
    """Every simplicity metric for one expression, in one dict."""
    out: Dict[str, float] = {}
    out.update(expression_size(expr))
    out.update(symbol_counts(expr, variables))
    out.update(description_length_bits(expr, variables, n_data))
    out["expression_chars"] = float(len(str(expr)))
    out["composite_penalty"] = -complexity_penalty * out["n_params"]
    return out
