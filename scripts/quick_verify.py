#!/usr/bin/env python3
"""Memory-bounded determinism check for a candidate generator.

`cascade verify` builds the full screening corpus and gets OOM-killed on a host
with under a few GB of RAM, which leaves a small operator with no way to check
their generator at all. This checks the properties that a determinism failure
would break — same seed, same bytes; different seed, different series; the
shape and dtype contract — over a deliberately tiny number of series.

It is NOT a substitute for `cascade verify` before submission: it does not run
the static guard, the sandbox, or the packaging checks, and it never sees the
real corpus size. Use it to iterate on a small host, then verify properly.

    python scripts/quick_verify.py generators/candidate --n-series 32
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_generator_class(repo: Path):
    source = repo / "generator.py"
    if not source.is_file():
        raise SystemExit(f"{repo} has no generator.py")
    spec = importlib.util.spec_from_file_location(f"candidate_{repo.name}", source)
    module = importlib.util.module_from_spec(spec)
    # The module lives outside any package; make sibling imports work anyway.
    sys.path.insert(0, str(repo))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(repo))
    generator = getattr(module, "Generator", None)
    if generator is None:
        raise SystemExit(f"{source} defines no `Generator` class")
    return generator


def series_from(generator_cls, repo: Path, seed: int, n_series: int) -> list[np.ndarray]:
    generator = generator_cls(str(repo), seed=seed)
    produced = []
    for index, series in enumerate(generator.generate(n_series)):
        if index >= n_series:
            break
        produced.append(np.asarray(series))
    return produced


def check_contract(series: list[np.ndarray], config: dict, n_series: int) -> list[str]:
    problems = []
    if len(series) != n_series:
        problems.append(f"asked for {n_series} series, generator yielded {len(series)}")
    low = config.get("min_length")
    high = config.get("max_length")
    for index, item in enumerate(series):
        if item.ndim != 1:
            problems.append(f"series {index} has shape {item.shape}, expected 1-D")
            continue
        if item.dtype != np.float64:
            problems.append(f"series {index} has dtype {item.dtype}, expected float64")
        if not np.isfinite(item).all():
            problems.append(f"series {index} contains NaN or inf")
        if isinstance(low, int) and item.size < low:
            problems.append(f"series {index} is {item.size} long, below min_length {low}")
        if isinstance(high, int) and item.size > high:
            problems.append(f"series {index} is {item.size} long, above max_length {high}")
    return problems


def identical(a: np.ndarray, b: np.ndarray) -> bool:
    """Same seed, same bytes. Byte equality so a NaN matches itself."""
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def check_determinism(first: list[np.ndarray], second: list[np.ndarray]) -> list[str]:
    if len(first) != len(second):
        return [f"same seed yielded {len(first)} then {len(second)} series"]
    problems = []
    for index, (a, b) in enumerate(zip(first, second)):
        if a.shape != b.shape:
            problems.append(f"series {index} changed shape between runs: {a.shape} vs {b.shape}")
        elif not identical(a, b):
            problems.append(f"series {index} differs between runs at the same seed")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo_dir", type=Path)
    parser.add_argument("--n-series", type=int, default=32,
                        help="series per run; keep small, this is the memory knob")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.n_series <= 0:
        parser.error("--n-series must be positive")

    repo = args.repo_dir.resolve()
    config_path = repo / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    generator_cls = load_generator_class(repo)

    first = series_from(generator_cls, repo, args.seed, args.n_series)
    second = series_from(generator_cls, repo, args.seed, args.n_series)
    other = series_from(generator_cls, repo, args.seed + 1, args.n_series)

    problems = check_contract(first, config, args.n_series)
    problems += check_determinism(first, second)
    if first and other and len(first) == len(other) and all(
        identical(a, b) for a, b in zip(first, other)
    ):
        problems.append("seed is ignored: seeds "
                        f"{args.seed} and {args.seed + 1} produced identical corpora")

    label = f"{repo.name} n_series={args.n_series} seed={args.seed}"
    if problems:
        print(f"quick-verify FAILED for {label}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    points = sum(int(item.size) for item in first)
    print(f"quick-verify passed for {label}: "
          f"{len(first)} series, {points:,} points, deterministic and seed-sensitive")
    print("this is not `cascade verify`; run the full check before submitting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
