"""Train one candidate at one seed and save raw per-window score components.

The harness keeps the checkpoint so it can be re-evaluated without retraining,
and stores the components required by the paired cluster-bootstrap LCB:

* the checkpoint is written to a stable path and KEPT, so a model can be
  re-evaluated later without retraining;
* per-window score COMPONENTS are saved (qloss_per_q, abs_target, mase, source),
  not just the aggregate geomean. The KOTH verdict is a paired cluster bootstrap
  over those components, and it cannot be reconstructed from a geomean.

King and challenger must share windows to be paired, so the same `--seed` drives
both RoundSeeds and window selection — exactly as a real round does.

    python -m miner.evaluate <repo_dir> --seed 0 --chain-toml ... --pools-root ...
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

SNAPSHOT_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def snapshot_dirs(pools_root: Path) -> list[Path]:
    """Return dated dataset snapshots, rejecting a one-level-too-high root."""
    if not pools_root.is_dir():
        raise ValueError(f"pools root does not exist: {pools_root}")
    snapshots = sorted(
        path for path in pools_root.iterdir()
        if path.is_dir() and SNAPSHOT_NAME.fullmatch(path.name)
        and any(path.glob("*.npy"))
    )
    if not snapshots:
        nested = pools_root / "snapshots"
        hint = f"; use --pools-root {nested}" if nested.is_dir() else ""
        raise ValueError(
            f"pools root {pools_root} contains no dated snapshot directories with .npy files"
            f"{hint}"
        )
    return snapshots


def train_once(repo: Path, cfg, hours: float, seed: int, out_dir: Path, trainer_spec: str):
    from cascade.trainer.contract import RoundSeeds
    from cascade.trainer.main import _load_trainer
    from cascade.trainer.stream import open_round_stream

    contract = cfg.screen_contract().for_hours(
        hours,
        guard_factor=cfg.round.heat_guard_factor,
        guard_floor_seconds=cfg.round.heat_guard_floor_seconds,
    )
    token_budget = contract.train_tokens
    seeds = RoundSeeds.derive(seed, cfg.training)
    trainer = _load_trainer(trainer_spec)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open_round_stream(
        contract.corpus_mode, repo, seeds.generation_seed, cfg.generator,
        token_budget=token_budget, use_sandbox=False, blocked=cfg.static_guard.blocked,
    ) as rs:
        result = trainer.train(
            rs.series(), contract,
            training_seed=seeds.training_seed, token_budget=token_budget, out_dir=out_dir,
        )
        digest, n_series, points = rs.digest, rs.n_series, rs.total_points
    return result, {
        "corpus_digest": digest, "n_series": n_series, "total_points": points,
        "token_budget": token_budget, "train_seconds": result.train_seconds,
        "deadline_hit": points < token_budget,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_dir", type=Path)
    ap.add_argument("--chain-toml", type=Path, required=True)
    ap.add_argument("--pools-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", type=Path, default=Path("ckpts"))
    ap.add_argument("--scores-root", type=Path, default=Path("scores"))
    ap.add_argument("--train-hours", type=float, default=1.0)
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trainer", default="cascade.trainer.toto2_trainer:Toto2Trainer")
    args = ap.parse_args()
    snaps = snapshot_dirs(args.pools_root)

    from cascade.eval.scoring import global_geomean, stack_components
    from cascade.shared.config import load_chain_config
    from cascade.validator.evaluator import evaluate_checkpoint
    from cascade.validator.pool import window_source_from_dir

    cfg = load_chain_config(args.chain_toml)
    n_win = args.n_windows or min(cfg.round.heat_n_windows, cfg.eval.n_windows)
    name = args.repo_dir.name
    tag = f"{name}__seed{args.seed}"

    ckpt = args.ckpt_root / tag
    marker = ckpt / "TRAINED.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        print(f"[{tag}] reusing existing checkpoint", flush=True)
    else:
        print(f"[{tag}] training -> {ckpt}", flush=True)
        t0 = time.perf_counter()
        result, meta = train_once(
            args.repo_dir, cfg, args.train_hours, args.seed, ckpt, args.trainer
        )
        meta["wall_seconds"] = time.perf_counter() - t0
        ckpt = Path(result.local_dir)
        (ckpt / "TRAINED.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{tag}] trained {meta['train_seconds']:.0f}s "
              f"pts={meta['total_points']:,}/{meta['token_budget']:,} "
              f"deadline_hit={meta['deadline_hit']}", flush=True)

    out_root = args.scores_root / tag
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"candidate": name, "seed": args.seed, "n_windows": n_win,
               "checkpoint": str(ckpt), "meta": meta, "per_snapshot": []}

    for snap in snaps:
        try:
            src = window_source_from_dir(snap, cfg, label=f"dir={snap}")
            windows = src.windows_for_round(args.seed, n_win)
            scores = evaluate_checkpoint(
                ckpt, windows, num_samples=cfg.eval.num_samples, device=args.device
            )
            qloss, abs_t, mase = stack_components(scores)
            # `source` is the KOTH cluster key; series_id pins pairing order.
            np.savez_compressed(
                out_root / f"{snap.name}.npz",
                qloss=qloss, abs_target=abs_t, mase=mase,
                source=np.array([s.source or "" for s in scores], dtype=object),
                series_id=np.array([s.series_id for s in scores], dtype=object),
                channel=np.array([s.channel for s in scores], dtype=np.int32),
            )
            g = global_geomean(scores)
            summary["per_snapshot"].append(
                {"snapshot": snap.name, "geomean": g, "n_windows": len(scores)}
            )
            print(f"  {snap.name} geomean={g:.5f} n={len(scores)}", flush=True)
        except Exception as e:  # noqa: BLE001
            summary["per_snapshot"].append({"snapshot": snap.name, "error": repr(e)})
            print(f"  {snap.name} ERROR {e}", flush=True)

    ok = [r["geomean"] for r in summary["per_snapshot"] if "geomean" in r]
    summary["mean_geomean"] = (sum(ok) / len(ok)) if ok else None
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{tag}] mean={summary['mean_geomean']} -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
