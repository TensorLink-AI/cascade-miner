"""Train one candidate at one seed and save raw per-window score components.

The harness keeps the checkpoint so it can be re-evaluated without retraining,
and stores the components required by the paired cluster-bootstrap LCB:

* the checkpoint is written to a stable path and KEPT, so a model can be
  re-evaluated later without retraining;
* per-window score COMPONENTS are saved (qloss_per_q, abs_target, mase, source),
  not just the aggregate geomean. The KOTH verdict is a paired cluster bootstrap
  over those components, and it cannot be reconstructed from a geomean.

King and challenger must share windows to be paired, so the same `--seed` drives
both RoundSeeds and window selection — exactly as a real round does. Under the
warm-start regime both duellists also share the round's init; `--warm-init`
emulates that by handing the trainer an explicit checkpoint directory, and the
checkpoint records which init it was trained under so a re-run can never
silently pair arms trained from different inits.

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


def _dated_snapshots(base: Path) -> list[Path]:
    return sorted(
        path for path in base.iterdir()
        if path.is_dir() and SNAPSHOT_NAME.fullmatch(path.name)
        and any(path.glob("*.npy"))
    )


def snapshot_dirs(pools_root: Path) -> list[Path]:
    """Return dated dataset snapshots, with a layout-aware error otherwise.

    The recurring mistake: provisioning pushes the local pool to the pod's
    ``.../pools/snapshots/``, while this flag wants the directory whose
    CHILDREN are the dated snapshots — one `snapshots/` too many or too few
    in the path is invisible in the command line. The old hint guessed a
    literal ``snapshots/`` child whether or not it held snapshots, which sent
    people to paths that were just as wrong. So: report what the directory
    actually contains, and only suggest a --pools-root that has been verified
    to hold dated snapshots (one level down, or the parent when the given
    path IS a dated snapshot).
    """
    if not pools_root.is_dir():
        raise ValueError(f"pools root does not exist: {pools_root}")
    snapshots = _dated_snapshots(pools_root)
    if snapshots:
        return snapshots
    hints = []
    if SNAPSHOT_NAME.fullmatch(pools_root.name) and any(pools_root.glob("*.npy")):
        hints.append(f"{pools_root} is itself a dated snapshot — "
                     f"use --pools-root {pools_root.parent}")
    for child in sorted(path for path in pools_root.iterdir() if path.is_dir()):
        if _dated_snapshots(child):
            hints.append(f"dated snapshots found one level down — "
                         f"use --pools-root {child}")
            break
    contents = ", ".join(sorted(p.name for p in pools_root.iterdir())[:8]) or "<empty>"
    raise ValueError(
        f"pools root {pools_root} contains no dated snapshot directories with "
        f".npy files (it contains: {contents})"
        + ("; " + "; ".join(hints) if hints else "")
    )


def live_rule_block(cfg) -> int | None:
    """A block at which every armed, block-gated eval rule is active.

    The live window draw is block-gated: the jittered mix (`[eval]
    mix_from_block`, DEC-CA-0019) and the two-tier even-domain split
    (`mix_tier_from_block`, DEC-CA-0032) apply only to rounds at or past their
    activation blocks — and when active, ``mix_target_windows`` overrides the
    requested window count outright. A local eval that passes no block replays
    the legacy uniform draw the subnet no longer uses, so its window count,
    composition, and noise floor all mismatch the live decision statistic.
    Emulating "a round scored today" means the largest armed activation block;
    None (nothing armed) makes the legacy draw correct again.
    """
    armed = [b for b in (cfg.eval.mix_from_block, cfg.eval.mix_tier_from_block)
             if b and b > 0]
    return max(armed) if armed else None


# trainer.train keywords probed, in order, for the warm-start init directory.
WARM_KWARG_CANDIDATES = ("init_dir", "warm_init_dir", "warm_init",
                         "init_checkpoint", "resume_from")


def warm_train_kwarg(trainer, explicit: str = "") -> str:
    """The trainer.train keyword that receives the warm-start init directory.

    A requested warm start must take effect or fail loudly. Silently training
    from scratch would produce an arm whose numbers pair against a warm king
    — a broken comparison that looks like a huge (or disastrous) result.
    """
    import inspect
    params = inspect.signature(trainer.train).parameters
    accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    if explicit:
        if explicit not in params and not accepts_kwargs:
            raise ValueError(
                f"trainer {type(trainer).__name__}.train has no parameter "
                f"{explicit!r} (has: {sorted(params)})")
        return explicit
    for name in WARM_KWARG_CANDIDATES:
        if name in params:
            return name
    raise ValueError(
        f"--warm-init given but trainer {type(trainer).__name__}.train accepts "
        f"none of {WARM_KWARG_CANDIDATES} (has: {sorted(params)}); pass "
        "--warm-init-kwarg with the trainer's actual parameter name")


def train_once(repo: Path, cfg, hours: float, seed: int, out_dir: Path, trainer_spec: str,
               warm_init: Path | None = None, warm_kwarg: str = ""):
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

    warm_kwargs = {}
    if warm_init is not None:
        if not warm_init.is_dir() or not any(warm_init.iterdir()):
            raise ValueError(f"--warm-init {warm_init} is not a non-empty directory")
        warm_kwargs = {warm_train_kwarg(trainer, warm_kwarg): str(warm_init)}

    with open_round_stream(
        contract.corpus_mode, repo, seeds.generation_seed, cfg.generator,
        token_budget=token_budget, use_sandbox=False, blocked=cfg.static_guard.blocked,
    ) as rs:
        result = trainer.train(
            rs.series(), contract,
            training_seed=seeds.training_seed, token_budget=token_budget, out_dir=out_dir,
            **warm_kwargs,
        )
        digest, n_series, points = rs.digest, rs.n_series, rs.total_points
    return result, {
        "corpus_digest": digest, "n_series": n_series, "total_points": points,
        "token_budget": token_budget, "train_seconds": result.train_seconds,
        "deadline_hit": points < token_budget,
        "warm_init": warm_init.name if warm_init is not None else None,
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
    ap.add_argument("--block", type=int, default=None,
                    help="epoch-boundary block the draw emulates; default = the "
                         "largest armed [eval] activation block, so the jittered "
                         "mix applies exactly as it would in a round scored "
                         "today. Pass 0 to force the legacy uniform draw.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trainer", default="cascade.trainer.toto2_trainer:Toto2Trainer")
    ap.add_argument("--warm-init", type=Path, default=None,
                    help="checkpoint directory the training starts from, "
                         "emulating a warm-started round. Both arms of a "
                         "paired comparison must receive the SAME value for "
                         "the same --seed.")
    ap.add_argument("--warm-init-kwarg", default="",
                    help="trainer.train parameter that receives --warm-init; "
                         "default: auto-detect from the trainer signature and "
                         "fail loudly when nothing matches")
    args = ap.parse_args()
    snaps = snapshot_dirs(args.pools_root)

    from cascade.eval.scoring import global_geomean, stack_components
    from cascade.shared.config import load_chain_config
    from cascade.validator.evaluator import evaluate_checkpoint
    from cascade.validator.pool import window_source_from_dir

    cfg = load_chain_config(args.chain_toml)
    n_win = args.n_windows or min(cfg.round.heat_n_windows, cfg.eval.n_windows)
    block = args.block if args.block is not None else live_rule_block(cfg)
    if block and cfg.eval.mix_target_windows:
        print(f"jittered mix active (block={block}): draw is "
              f"mix_target_windows={cfg.eval.mix_target_windows}, "
              f"not the requested {n_win}", flush=True)
    name = args.repo_dir.name
    tag = f"{name}__seed{args.seed}"

    warm_name = args.warm_init.name if args.warm_init is not None else None
    ckpt = args.ckpt_root / tag
    marker = ckpt / "TRAINED.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        # A checkpoint trained under a different init is a different arm;
        # reusing it silently would break the paired comparison.
        if meta.get("warm_init") != warm_name:
            raise SystemExit(
                f"[{tag}] existing checkpoint was trained with "
                f"warm_init={meta.get('warm_init')!r} but this run asks for "
                f"{warm_name!r}; delete {ckpt} or use a different --ckpt-root")
        print(f"[{tag}] reusing existing checkpoint", flush=True)
    else:
        print(f"[{tag}] training -> {ckpt}"
              + (f" (warm init: {warm_name})" if warm_name else ""), flush=True)
        t0 = time.perf_counter()
        result, meta = train_once(
            args.repo_dir, cfg, args.train_hours, args.seed, ckpt, args.trainer,
            warm_init=args.warm_init, warm_kwarg=args.warm_init_kwarg,
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
               "draw_block": block, "checkpoint": str(ckpt),
               "warm_init": warm_name, "meta": meta, "per_snapshot": []}

    for snap in snaps:
        try:
            src = window_source_from_dir(snap, cfg, label=f"dir={snap}")
            windows = src.windows_for_round(args.seed, n_win, block=block)
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
