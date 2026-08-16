#!/usr/bin/env python3
"""Extract the miner-facing state of upstream cascade into a checked-in snapshot.

The competition moves as `chain.toml` keys and `decisions/DEC-CA-*.md` nodes
land in `TensorLink-AI/cascade`, not as GitHub issues. Before this script the
only record of "what the subnet does today" was prose in `notes/CONTRACT.md`,
hand-folded after a sync — so it drifted silently (warm-start went live on
2026-08-05 while the notes still said the trainer trains from random init).

This splits that knowledge in two:

  * FACTS  — every miner-facing key and decision node, read straight from the
             reference clone into `notes/upstream-state.json` (+ a rendered
             `notes/UPSTREAM.md`). Mechanical, regenerated on a schedule, no
             judgement involved.
  * PROSE  — `notes/CONTRACT.md`, still written by hand, but now pinned to the
             facts by `tests/test_stale_references.py`, which reads the snapshot
             and fails when a sentence contradicts it.

So a change upstream can no longer pass unnoticed: it either shows up as a diff
in the snapshot, or it breaks a test that names the stale sentence.

Stdlib only (tomllib, 3.11+) — the offline test suite has no venv and no network.

    python scripts/upstream_state.py                 # regenerate from $CASCADE_DIR
    python scripts/upstream_state.py --check         # exit 3 if the snapshot is stale
    python scripts/upstream_state.py --print         # dump to stdout, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "notes/upstream-state.json"
RENDERED = ROOT / "notes/UPSTREAM.md"

SCHEMA = 1
UPSTREAM_REPO = "TensorLink-AI/cascade"

# Exit codes: 0 current / written, 1 error, 2 usage, 3 drift (--check).
EXIT_DRIFT = 3

# Every key below is one the miner competes against — it changes what we may
# submit, what the trainer does with it, or what statistic decides the round.
# `(group, key, why)`; the group headings are what `notes/UPSTREAM.md` renders.
# A key that disappears upstream is recorded in `missing_keys` rather than
# crashing the run: a key vanishing is itself a change we want to see.
KEYS: tuple[tuple[str, str, str], ...] = (
    ("Subnet", "subnet.netuid", "which chain we compete on"),

    ("Submission limits", "generator.corpus_n_series", "series drawn per training run"),
    ("Submission limits", "generator.min_length", "per-series length band, low"),
    ("Submission limits", "generator.max_length", "per-series length band, high"),
    ("Submission limits", "generator.max_total_points", "corpus cap"),
    ("Submission limits", "generator.max_generate_seconds", "wall-clock budget to drain the generator"),
    ("Submission limits", "generator.max_memory_mb", "rlimit on the generation sandbox"),
    ("Submission limits", "generator.max_repo_mb", "cap on the fetched submission (code only)"),
    ("Submission limits", "generator.max_channels", "variates per series"),
    ("Submission limits", "generator.max_abs_value", "peak |value| reject bar; 0.0 = float32 ceiling"),
    ("Submission limits", "generator.reject_constant", "flat series rejected?"),
    ("Submission limits", "generator.max_dup_fraction", "byte-duplicate series allowed (cache_reuse only)"),
    ("Submission limits", "dependencies.max_packages", "requirements.txt package cap"),
    ("Submission limits", "dependencies.allowed", "the dependency allowlist"),
    ("Submission limits", "static_guard.blocked", "imports the AST scan rejects"),

    ("Round mechanics", "round.epoch_blocks", "round length in blocks"),
    ("Round mechanics", "round.epoch_blocks_prev", "pre-activation round length"),
    ("Round mechanics", "round.epoch_activation_block", "block the new length takes effect"),
    ("Round mechanics", "round.round_hours", "wall-clock span of an epoch"),
    ("Round mechanics", "round.reveal_margin_blocks", "timed-reveal margin before the boundary"),
    ("Round mechanics", "round.heat_train_hours", "screening budget per competitor"),
    ("Round mechanics", "round.heat_n_windows", "eval windows the heat screens on"),
    ("Round mechanics", "round.heat_num_samples", "sample forecasts per heat window"),
    ("Round mechanics", "round.finalists", "challengers promoted from heat to duel"),
    ("Round mechanics", "round.screen_size", "arch size the heat screens at"),
    ("Round mechanics", "round.throne_sizes", "size(s) the duel decides on"),
    ("Round mechanics", "round.commit_floor_block", "reveals before this never compete"),
    ("Round mechanics", "round.one_submission_per_hotkey", "a heat entry burns the hotkey"),
    ("Round mechanics", "round.dedup_mode", "pre-heat content dedup enforcement"),
    ("Round mechanics", "round.dedup_probe_mode", "behavioural (identical-output) probe"),

    ("Training contract", "training.base_arch", "backbone family"),
    ("Training contract", "training.arch_preset", "backbone size"),
    ("Training contract", "training.target_train_hours", "final-stage budget"),
    ("Training contract", "training.ref_throughput_tokens_per_s", "throughput the token budget is calibrated to"),
    ("Training contract", "training.max_train_seconds", "hard wall-clock cap (a stop is deadline_hit)"),
    ("Training contract", "training.corpus_mode", "how the corpus is fed to the trainer"),
    ("Training contract", "training.expected_gpu", "pinned SKU for king and challenger"),
    ("Training contract", "training.train_image_digest", "pinned worker image; folded into contract_digest"),
    ("Training contract", "training.context_length", "input length"),
    ("Training contract", "training.horizon", "forecast horizon"),

    ("Decision statistic", "eval.n_windows", "eval windows drawn per round"),
    ("Decision statistic", "eval.num_samples", "sample forecasts per window"),
    ("Decision statistic", "scoring.win_margin_start", "margin the challenger's LCB must clear"),
    ("Decision statistic", "scoring.win_margin_end", "margin at full tenure warmup"),
    ("Decision statistic", "scoring.margin_warmup_rounds", "tenure ramp; 0 = flat margin"),
    ("Decision statistic", "scoring.bootstrap_B", "paired-bootstrap resamples"),
    ("Decision statistic", "scoring.bootstrap_alpha", "one-sided LCB level"),
    ("Decision statistic", "scoring.min_windows", "below this, no decision"),
    ("Decision statistic", "scoring.min_clusters", "min distinct window clusters"),
    ("Decision statistic", "scoring.dethrone_cp", "consecutive round wins to take the throne"),
    ("Decision statistic", "scoring.gift_gate_mode", "public-benchmark no-regression gate"),
    ("Decision statistic", "scoring.gift_gate_tolerance", "slack before that gate fails a challenger"),
    ("Decision statistic", "scoring.reward_prior_kings", "prior kings still paid"),
    ("Decision statistic", "scoring.king_decay", "geometric decay across paid kings"),

    ("Warm start", "scoring.cascade_enabled", "warm-start promotion armed? decides whether the trainer starts from random init"),
    ("Warm start", "scoring.cascade_reign_rounds", "undethroned rounds before a promotion fires"),
    ("Warm start", "scoring.cascade_top_k", "member checkpoints one promotion may declare"),
    ("Warm start", "scoring.cascade_quality_epsilon", "quality floor members must sit within"),
)


class ExtractError(RuntimeError):
    """Raised for an unusable reference clone — the caller turns it into exit 1."""


def cascade_dir(explicit: str | None = None) -> Path:
    return Path(explicit or os.environ.get("CASCADE_DIR") or "/root/cascade").expanduser()


def _git(repo: Path, *args: str) -> str | None:
    """`git -C repo ...`, or None when git or the checkout is unavailable.

    A tarball/`actions/checkout` copy with no git metadata still yields a usable
    snapshot — the revision fields just read "unknown", which the workflow
    surfaces rather than silently pinning a wrong sha.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _lookup(table: dict[str, Any], dotted: str) -> tuple[bool, Any]:
    node: Any = table
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def read_decisions(root: Path) -> list[dict[str, str]]:
    """`id`/`status`/`date`/`title` from each decision node's YAML frontmatter.

    Deliberately a line scanner, not a YAML parse: the suite runs with numpy as
    its only dependency, and these four keys are flat scalars in every node.
    """
    out: list[dict[str, str]] = []
    directory = root / "decisions"
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not text or text[0].strip() != "---":
            continue
        entry: dict[str, str] = {}
        for line in text[1:]:
            if line.strip() == "---":
                break
            key, sep, value = line.partition(":")
            if not sep or key != key.strip():   # nested/continuation line
                continue
            if key.strip() in ("id", "status", "date", "title"):
                entry[key.strip()] = value.strip().strip('"').strip("'")
        if entry.get("id"):
            out.append({k: entry.get(k, "") for k in ("id", "status", "date", "title")})
    return out


def build(cascade: Path) -> dict[str, Any]:
    chain_path = cascade / "chain.toml"
    if not chain_path.is_file():
        raise ExtractError(
            f"no chain.toml under {cascade} — point --cascade-dir (or $CASCADE_DIR) "
            "at the read-only cascade reference clone"
        )
    with chain_path.open("rb") as handle:
        chain = tomllib.load(handle)

    keys: dict[str, Any] = {}
    missing: list[str] = []
    for _group, dotted, _why in KEYS:
        found, value = _lookup(chain, dotted)
        if found:
            keys[dotted] = value
        else:
            missing.append(dotted)

    return {
        "schema": SCHEMA,
        "generated_by": "scripts/upstream_state.py",
        "generated_from": {
            "repo": UPSTREAM_REPO,
            "revision": _git(cascade, "rev-parse", "--short", "HEAD") or "unknown",
            "revision_date": _git(cascade, "log", "-1", "--format=%ad", "--date=short") or "unknown",
            "chain_schema_version": chain.get("schema_version"),
        },
        "keys": keys,
        "missing_keys": missing,
        "decisions": read_decisions(cascade),
    }


def _cell(text: str) -> str:
    """A bare `|` inside a cell splits the markdown table — escape it."""
    return text.replace("|", "\\|")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, list):
        return ", ".join(f"`{_cell(str(v))}`" for v in value) or "*(empty)*"
    return f"`{_cell(str(value))}`"


def render(state: dict[str, Any]) -> str:
    src = state["generated_from"]
    lines = [
        "# Upstream cascade — miner-facing state",
        "",
        "<!-- GENERATED by scripts/upstream_state.py — do not edit by hand. -->",
        "",
        f"Read from `{src['repo']}` at `{src['revision']}` ({src['revision_date']}),",
        f"chain.toml schema {src['chain_schema_version']}.",
        "",
        "These are the **facts**, refreshed mechanically by the `upstream-sync`",
        "workflow. The **interpretation** lives in `notes/CONTRACT.md`, which",
        "`tests/test_stale_references.py` pins to the values below — if prose and",
        "table disagree, the test suite says which sentence is stale.",
        "",
    ]
    seen: set[str] = set()
    for group, dotted, why in KEYS:
        if dotted not in state["keys"]:
            continue
        if group not in seen:
            seen.add(group)
            lines += ["", f"## {group}", "", "| key | value | why it matters |", "|---|---|---|"]
        lines.append(f"| `{dotted}` | {_fmt(state['keys'][dotted])} | {_cell(why)} |")

    if state["missing_keys"]:
        lines += [
            "",
            "## Keys that vanished upstream",
            "",
            "Tracked here but no longer in `chain.toml` — either renamed or removed,",
            "and worth a look either way:",
            "",
        ]
        lines += [f"- `{k}`" for k in state["missing_keys"]]

    if state["decisions"]:
        lines += ["", "## Decision nodes", "", "| id | status | date | title |", "|---|---|---|---|"]
        for d in state["decisions"]:
            lines.append(
                f"| `{d['id']}` | {d['status']} | {d['date']} | {_cell(d['title'])} |"
            )

    lines.append("")
    return "\n".join(lines)


def serialize(state: dict[str, Any]) -> str:
    return json.dumps(state, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def load_snapshot(path: Path = SNAPSHOT) -> dict[str, Any]:
    """The committed snapshot — the tests' view of what upstream does today."""
    return json.loads(path.read_text(encoding="utf-8"))


def changed_keys(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Keys whose value differs between two snapshots (added/removed included)."""
    o, n = old.get("keys", {}), new.get("keys", {})
    absent = object()
    return sorted(k for k in set(o) | set(n) if o.get(k, absent) != n.get(k, absent))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate notes/upstream-state.json + notes/UPSTREAM.md from the cascade reference clone.",
    )
    parser.add_argument("--cascade-dir", help="reference clone (default: $CASCADE_DIR, else /root/cascade)")
    parser.add_argument("--check", action="store_true",
                        help=f"report drift and change nothing; exit {EXIT_DRIFT} when stale")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="write the JSON to stdout instead of the snapshot files")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        state = build(cascade_dir(args.cascade_dir))
    except ExtractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    payload, rendered = serialize(state), render(state)

    if args.print_only:
        sys.stdout.write(payload)
        return 0

    old_payload = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.is_file() else ""
    old_rendered = RENDERED.read_text(encoding="utf-8") if RENDERED.is_file() else ""
    drifted = payload != old_payload or rendered != old_rendered

    if args.check:
        if not drifted:
            print(f"current: snapshot matches cascade {state['generated_from']['revision']}")
            return 0
        print(f"STALE: snapshot does not match cascade {state['generated_from']['revision']}")
        try:
            old_state = json.loads(old_payload) if old_payload else {}
        except json.JSONDecodeError:
            print("    (committed snapshot is not valid JSON — regenerate it)")
            return EXIT_DRIFT
        for key in changed_keys(old_state, state):
            print(f"    {key}: {old_state.get('keys', {}).get(key, '(absent)')!r}"
                  f" -> {state['keys'].get(key, '(absent)')!r}")
        return EXIT_DRIFT

    SNAPSHOT.write_text(payload, encoding="utf-8")
    RENDERED.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {SNAPSHOT.relative_to(ROOT)} + {RENDERED.relative_to(ROOT)} "
        f"from cascade {state['generated_from']['revision']}"
        + (" (unchanged)" if not drifted else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
