#!/usr/bin/env python3
"""Compiled equation candidates with variable validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from canonical import family_signature, structure_signature
from grammar import (
    ExpressionError,
    ProtectionLog,
    compile_numpy,
    compile_torch,
    extract_param_names,
    identifiers,
    is_coefficient_name,
    normalise_expression,
)


def _probable_variable(name: str, var_names: Sequence[str]) -> Optional[str]:
    """The variable ``name`` was probably meant to be, for the error message."""
    lowered = {v.lower(): v for v in var_names}
    if name.lower() in lowered:
        return lowered[name.lower()]

    stripped = re.sub(r"\d+$", "", name)
    if stripped != name and stripped in var_names:
        return stripped
    return None


@dataclass
class EquationProgram:
    """A parsed, compiled, validated candidate equation."""

    expr: str
    var_names: List[str]
    param_names: List[str]

    f_np: Callable
    f_torch: Callable

    protection_np: ProtectionLog
    protection_torch: ProtectionLog

    used_variables: List[str] = field(default_factory=list)
    unused_variables: List[str] = field(default_factory=list)

    init_scale: float = 0.5
    prior_sigma: float = 10.0

    @classmethod
    def from_expression(
        cls,
        expr: str,
        var_names: Sequence[str],
        max_params: Optional[int] = 12,
        require_all_variables: bool = False,
    ) -> "EquationProgram":
        expr = normalise_expression(expr)
        if not expr:
            raise ExpressionError("empty expression")

        var_names = list(var_names)
        names = identifiers(expr)

        for n in names:
            if n in var_names or is_coefficient_name(n):
                continue
            intended = _probable_variable(n, var_names)
            hint = f"; did you mean the variable '{intended}'?" if intended else ""
            raise ExpressionError(
                f"'{n}' is neither a declared variable {var_names} nor a "
                f"coefficient (coefficients must be named c0, c1, ... or c_<name>)"
                f"{hint}"
            )

        param_names = extract_param_names(expr, var_names)
        if max_params is not None and len(param_names) > max_params:
            raise ExpressionError(
                f"{len(param_names)} parameters exceeds the budget of {max_params}"
            )

        used = [v for v in var_names if v in names]
        unused = [v for v in var_names if v not in names]
        if require_all_variables and unused:
            raise ExpressionError(f"expression omits declared variables {unused}")

        f_np, log_np = compile_numpy(expr, var_names, param_names)
        f_torch, log_torch = compile_torch(expr, var_names, param_names)

        return cls(
            expr=expr,
            var_names=var_names,
            param_names=param_names,
            f_np=f_np,
            f_torch=f_torch,
            protection_np=log_np,
            protection_torch=log_torch,
            used_variables=used,
            unused_variables=unused,
        )

    def structure_signature(self) -> str:
        return structure_signature(self.expr, self.var_names)

    def family_signature(self) -> str:
        return family_signature(self.expr, self.var_names)

    def n_params(self) -> int:
        return len(self.param_names)

    def initial_theta(self, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        rng = rng or np.random.default_rng()
        return {p: float(rng.normal(scale=self.init_scale)) for p in self.param_names}

    def predict(self, X: np.ndarray, theta: Dict[str, float]) -> np.ndarray:
        return self.f_np(X, theta)

    def protection_summary(self) -> Dict[str, float]:
        """Fraction of elements each protected operator had to clamp."""
        return self.protection_np.summary()

    def __repr__(self) -> str:
        return f"EquationProgram({self.expr!r}, params={self.param_names})"
