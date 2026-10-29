#!/usr/bin/env python3
"""Expression grammar with protected operators."""

from __future__ import annotations

import threadcap  # noqa: F401  (must precede NumPy)

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

EPS = 1e-12

COEFFICIENT_PATTERN = re.compile(r"^c(?:\d+|_[A-Za-z0-9]+)$")


def is_coefficient_name(name: str) -> bool:
    return bool(COEFFICIENT_PATTERN.match(name))

FUNCTION_NAMES = frozenset({
    "sin", "cos", "tan", "tanh", "sinh", "cosh",
    "exp", "log", "sqrt", "abs", "sign", "softplus",
    "minimum", "maximum", "pow", "pi", "E",
})

_ALLOWED_NODES = (
    ast.Expression, ast.UnaryOp, ast.BinOp, ast.Name, ast.Load, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Constant,
)


class ExpressionError(ValueError):
    """Raised when an expression cannot be accepted into the grammar."""


@dataclass
class ProtectionLog:
    """Counts how often each protected operator had to intervene."""

    counts: Dict[str, int] = field(default_factory=dict)
    totals: Dict[str, int] = field(default_factory=dict)
    enabled: bool = True

    def record(self, op: str, activated, total: int) -> None:
        self.counts[op] = self.counts.get(op, 0) + int(activated)
        self.totals[op] = self.totals.get(op, 0) + int(total)

    def reset(self) -> None:
        self.counts.clear()
        self.totals.clear()

    @property
    def activation_rate(self) -> float:
        """Worst per-operator activation rate across the last evaluation."""
        rates = [self.counts[k] / self.totals[k] for k in self.counts if self.totals.get(k)]
        return max(rates) if rates else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            k: self.counts[k] / self.totals[k]
            for k in sorted(self.counts)
            if self.totals.get(k) and self.counts[k]
        }


def _size(x) -> int:
    return int(np.size(x))


def make_numpy_ops(log: ProtectionLog) -> Dict[str, Callable]:
    def p_div(a, b):
        b = np.asarray(b, dtype=float)
        unsafe = np.abs(b) < 1e-9
        if log.enabled:
            log.record("div", np.count_nonzero(unsafe), _size(b))
        safe_b = np.where(unsafe, np.where(b >= 0, 1e-9, -1e-9), b)
        return np.asarray(a, dtype=float) / safe_b

    def p_log(x):
        x = np.asarray(x, dtype=float)
        if log.enabled:
            log.record("log", np.count_nonzero(x <= EPS), _size(x))
        return np.log(np.abs(x) + EPS)

    def p_sqrt(x):
        x = np.asarray(x, dtype=float)
        if log.enabled:
            log.record("sqrt", np.count_nonzero(x < 0), _size(x))
        return np.sqrt(np.abs(x))

    def p_pow(a, b):
        a = np.asarray(a, dtype=float)
        if log.enabled:
            log.record("pow", np.count_nonzero(a < 0), _size(a))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out = np.power(np.abs(a) + EPS, np.clip(b, -30.0, 30.0))
        return out

    def p_exp(x):
        return np.exp(np.clip(np.asarray(x, dtype=float), -60.0, 60.0))

    def softplus(x):
        x = np.asarray(x, dtype=float)
        return np.logaddexp(0.0, np.clip(x, -60.0, 60.0))

    return {
        "p_div": p_div, "p_log": p_log, "p_sqrt": p_sqrt, "p_pow": p_pow,
        "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
        "sinh": np.sinh, "cosh": np.cosh, "exp": p_exp, "log": p_log,
        "sqrt": p_sqrt, "abs": np.abs, "sign": np.sign, "softplus": softplus,
        "minimum": np.minimum, "maximum": np.maximum, "pow": p_pow,
        "pi": math.pi, "E": math.e,
    }


def make_torch_ops(log: ProtectionLog) -> Dict[str, Callable]:
    def _t(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)

    def p_div(a, b):
        b = _t(b)
        unsafe = b.abs() < 1e-9
        if log.enabled:
            log.record("div", int(unsafe.sum()), int(b.numel()))
        return _t(a) / torch.where(unsafe, torch.full_like(b, 1e-9) * torch.sign(b + 1e-30), b)

    def p_log(x):
        x = _t(x)
        if log.enabled:
            log.record("log", int((x <= EPS).sum()), int(x.numel()))
        return torch.log(x.abs() + EPS)

    def p_sqrt(x):
        x = _t(x)
        if log.enabled:
            log.record("sqrt", int((x < 0).sum()), int(x.numel()))
        return torch.sqrt(x.abs() + EPS)

    def p_pow(a, b):
        a = _t(a)
        if log.enabled:
            log.record("pow", int((a < 0).sum()), int(a.numel()))
        return torch.pow(a.abs() + EPS, torch.clamp(_t(b), -30.0, 30.0))

    def p_exp(x):
        return torch.exp(torch.clamp(_t(x), -60.0, 60.0))

    return {
        "p_div": p_div, "p_log": p_log, "p_sqrt": p_sqrt, "p_pow": p_pow,
        "sin": torch.sin, "cos": torch.cos, "tan": torch.tan, "tanh": torch.tanh,
        "sinh": torch.sinh, "cosh": torch.cosh, "exp": p_exp, "log": p_log,
        "sqrt": p_sqrt, "abs": torch.abs, "sign": torch.sign,
        "softplus": torch.nn.functional.softplus,
        "minimum": torch.minimum, "maximum": torch.maximum, "pow": p_pow,
        "pi": math.pi, "E": math.e,
    }


class _Protector(ast.NodeTransformer):
    """Rewrites domain-restricted operations into their protected forms."""

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.Call(func=ast.Name(id="p_div", ctx=ast.Load()),
                            args=[node.left, node.right], keywords=[])
        if isinstance(node.op, ast.Pow):
            right = node.right
            if isinstance(right, ast.Constant) and isinstance(right.value, int):
                return node
            return ast.Call(func=ast.Name(id="p_pow", ctx=ast.Load()),
                            args=[node.left, node.right], keywords=[])
        return node


def validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("only direct function calls are allowed")
            if node.func.id not in FUNCTION_NAMES and not node.func.id.startswith("p_"):
                raise ExpressionError(f"unknown function: {node.func.id}")
            if node.keywords:
                raise ExpressionError("keyword arguments are not allowed")


_LHS_ASSIGNMENT = re.compile(r"^\s*[A-Za-z_]\w*(\([^)]*\))?\s*=(?!=)")


def normalise_expression(expr: str) -> str:
    """Repair notation that is unambiguous but not Python."""
    expr = expr.strip().rstrip(";").strip()
    expr = _LHS_ASSIGNMENT.sub("", expr, count=1).strip()
    return expr.replace("^", "**")


def parse(expr: str) -> ast.Expression:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from exc
    validate_ast(tree)
    return tree


def identifiers(expr: str) -> List[str]:
    """All Name nodes in the expression, excluding called functions."""
    tree = parse(expr)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - called)


def extract_param_names(expr: str, var_names: Sequence[str]) -> List[str]:
    """Identifiers that are neither declared variables nor reserved names."""
    reserved = set(var_names) | FUNCTION_NAMES
    return sorted(n for n in identifiers(expr) if n not in reserved)


def compile_numpy(expr: str, var_names: Sequence[str], param_names: Sequence[str]):
    """Compile to ``f(X, theta) -> ndarray`` plus the protection log it writes to."""
    tree = _Protector().visit(parse(expr))
    ast.fix_missing_locations(tree)
    code = compile(tree, "<expr>", "eval")
    log = ProtectionLog()
    ops = make_numpy_ops(log)

    var_names = list(var_names)
    param_names = list(param_names)
    empty_globals = {"__builtins__": {}}

    ctx = dict(ops)
    cache: Dict[str, object] = {"X_id": None, "n": 0}

    def f(X: np.ndarray, theta: Dict[str, float]) -> np.ndarray:
        if cache["X_id"] is not id(X):
            Xf = np.asarray(X, dtype=float)
            for i, v in enumerate(var_names):
                ctx[v] = Xf[:, i]
            cache["X_id"] = id(X)
            cache["n"] = Xf.shape[0]
        for p in param_names:
            ctx[p] = theta.get(p, 0.0)

        if log.enabled:
            log.reset()
        out = eval(code, empty_globals, ctx)
        if np.ndim(out) == 0:
            return np.full(cache["n"], float(out))
        return np.asarray(out, dtype=float)

    return f, log


def compile_torch(expr: str, var_names: Sequence[str], param_names: Sequence[str]):
    """Compile to a differentiable ``f(X, theta) -> Tensor``."""
    tree = _Protector().visit(parse(expr))
    ast.fix_missing_locations(tree)
    code = compile(tree, "<expr>", "eval")
    log = ProtectionLog()
    ops = make_torch_ops(log)

    def f(X: torch.Tensor, theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        log.reset()
        ctx = dict(ops)
        for i, v in enumerate(var_names):
            ctx[v] = X[:, i]
        for p in param_names:
            ctx[p] = theta.get(p, torch.zeros((), dtype=X.dtype, device=X.device))
        out = eval(code, {"__builtins__": {}}, ctx)
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out, dtype=X.dtype, device=X.device)
        return out.expand(X.shape[0]) if out.dim() == 0 else out

    return f, log
