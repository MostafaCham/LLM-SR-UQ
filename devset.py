#!/usr/bin/env python3
"""The frozen dev set: 5 problems from each LSR-Synth domain."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

from benchmark import LSR_SYNTH_SUBSETS, Problem, dev_split, load_subset

DEV_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_set.json")
PER_DOMAIN = 5


def build(per_domain: int = PER_DOMAIN, seed: int = 0, **load_kwargs) -> Dict[str, List[str]]:
    """Resolve the dev membership by sampling each domain independently."""
    chosen: Dict[str, List[str]] = {}
    for subset in LSR_SYNTH_SUBSETS:
        problems = load_subset(subset, seed=seed, **load_kwargs)
        dev, _ = dev_split(problems, per_domain=per_domain, seed=seed)
        chosen[subset] = sorted(p.instance_id for p in dev)
    return chosen


def save(membership: Dict[str, List[str]], path: str = DEV_SET_PATH) -> None:
    payload = {
        "per_domain": PER_DOMAIN,
        "note": "Fixed dev set. Changing it invalidates cross-rung comparisons.",
        "subsets": membership,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_ids(path: str = DEV_SET_PATH) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["subsets"]


def load_dev_problems(path: str = DEV_SET_PATH, seed: int = 0, **load_kwargs) -> List[Problem]:
    """Load exactly the frozen dev problems, in a stable order."""
    membership = load_ids(path)
    out: List[Problem] = []
    for subset in sorted(membership):
        wanted = set(membership[subset])
        found = [p for p in load_subset(subset, seed=seed, **load_kwargs)
                 if p.instance_id in wanted]
        missing = wanted - {p.instance_id for p in found}
        if missing:
            raise RuntimeError(f"frozen dev ids missing from {subset}: {sorted(missing)}")
        out.extend(sorted(found, key=lambda p: p.instance_id))
    return out


def describe(problems: Sequence[Problem]) -> None:
    print(f"{'instance_id':<38} {'vars':<22} {'train':>6} {'cal':>5} "
          f"{'test':>6} {'ood':>6}  gt")
    print("-" * 110)
    for p in problems:
        sig = f"{','.join(p.input_vars)}->{p.output_var}"
        ood = str(len(p.ood)) if p.has_ood else "-"
        gt = "numeric" if p.gt_is_numeric else "structure-only"
        print(f"{p.instance_id:<38} {sig:<22} {len(p.train):>6} {len(p.cal):>5} "
              f"{len(p.test):>6} {ood:>6}  {gt}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.rebuild or not os.path.exists(DEV_SET_PATH):
        action = "rebuilding" if os.path.exists(DEV_SET_PATH) else "creating"
        print(f"{action} {DEV_SET_PATH}")
        save(build(seed=args.seed))

    problems = load_dev_problems(seed=args.seed)
    describe(problems)

    n_ood = sum(1 for p in problems if p.has_ood)
    n_numeric = sum(1 for p in problems if p.gt_is_numeric)
    print(f"\n{len(problems)} problems | {n_ood} with an OOD split | "
          f"{n_numeric} with numerically evaluable ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
