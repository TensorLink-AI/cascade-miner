"""One-look miner status assembled from the controller's ``runs/`` files.

Everything here is a read: the controller state, the approval queue, the event
history tail, and the candidate working-tree digest. ``--json`` emits the same
summary as data, which is also what the MCP ``miner_status`` tool returns.

    python -m miner.status
    python -m miner.status --json

``--doctor`` runs the onboarding checklist instead: every stage between a
fresh clone and a submittable miner, each with the exact command that fixes
it. Read-only, offline, and exit status 1 until the host is ready.

    python -m miner.status --doctor
    python -m miner.status --doctor --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Any

from miner import experiments
from miner.controller import candidate_digest, candidate_dirty, read_json

STATE_FILE = Path("runs/controller-state.json")
APPROVALS_FILE = Path("runs/approvals.json")
EVENTS_FILE = Path("runs/controller-events.jsonl")
EVENT_TAIL = 10


def tail_events(path: Path, limit: int = EVENT_TAIL) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines: deque[str] = deque(maxlen=limit)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                lines.append(line)
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def summarize(root: Path) -> dict[str, Any]:
    state = read_json(root / STATE_FILE)
    receipt = state.get("last_receipt", {}) if isinstance(state, dict) else {}
    summary = receipt.get("summary", {}) if isinstance(receipt, dict) else {}
    approvals = read_json(root / APPROVALS_FILE).get("requests", [])
    if not isinstance(approvals, list):
        approvals = []
    try:
        candidate = {
            "digest": candidate_digest(root),
            "dirty": candidate_dirty(root),
        }
    except Exception:  # noqa: BLE001 - status must render without git
        candidate = {"digest": None, "dirty": None}
    ledger = experiments.list_entries(root, limit=5)
    return {
        "initialized": bool(state.get("initialized")),
        "updated_at": state.get("updated_at"),
        "round": {
            "id": state.get("last_round_id"),
            "status": receipt.get("status"),
            "king_gen_ref": summary.get("king_gen_ref"),
            "miner_participant": receipt.get("miner_participant"),
            "miner_heat": receipt.get("miner_heat"),
        },
        "eval_pool": {
            "repo": state.get("eval_pool_repo"),
            "revision": state.get("eval_pool_revision"),
            "snapshot": state.get("eval_pool_snapshot"),
            "local_snapshots": state.get("eval_pool_local_snapshots", []),
        },
        "cascade_head": state.get("cascade_head"),
        "candidate": candidate,
        "approvals": {
            "pending": [r for r in approvals if r.get("status") == "pending"],
            "counts": {
                status: sum(1 for r in approvals if r.get("status") == status)
                for status in sorted({str(r.get("status")) for r in approvals})
            },
        },
        "experiments": ledger,
        "recent_events": tail_events(root / EVENTS_FILE),
    }


def render(summary: dict[str, Any]) -> str:
    lines = []
    round_info = summary["round"]
    lines.append(f"round:       {round_info['id'] or '(none)'}"
                 f"  status={round_info['status'] or '?'}")
    lines.append(f"king ref:    {round_info['king_gen_ref'] or '(unknown)'}")
    heat = round_info.get("miner_heat")
    if heat:
        lines.append(f"our heat:    {json.dumps(heat, sort_keys=True)}")
    pool = summary["eval_pool"]
    revision = str(pool["revision"] or "")
    lines.append(f"eval pool:   {pool['repo'] or '(unsynced)'}"
                 f"@{revision[:12]}  local={','.join(pool['local_snapshots']) or 'none'}")
    candidate = summary["candidate"]
    digest = str(candidate["digest"] or "")
    dirty = candidate["dirty"]
    lines.append(f"candidate:   {digest[:12] or '(unknown)'}"
                 f"  {'dirty (changes awaiting review)' if dirty else 'clean' if dirty is not None else ''}")
    pending = summary["approvals"]["pending"]
    if pending:
        lines.append(f"approvals:   {len(pending)} pending")
        for request in pending:
            lines.append(f"  [{request.get('id')}] {request.get('action')}: "
                         f"{request.get('reason', '')}")
        lines.append("  approve with: .venv/bin/python -m miner.controller --approve <id>")
    else:
        lines.append("approvals:   none pending")
    entries = summary["experiments"]
    if entries:
        lines.append(f"experiments: last {len(entries)} ledger entries")
        for entry in entries:
            lines.append(f"  {entry.get('id')} {entry.get('status'):<9} "
                         f"{entry.get('hypothesis', '')}")
    lines.append(f"updated at:  {summary['updated_at'] or '(controller has not run)'}")
    return "\n".join(lines)


REQUIRED_ENV_KEYS = (
    "HF_TOKEN", "HIPPIUS_HUB_TOKEN", "LIUM_API_KEY", "CASCADE_MINER_HOTKEY",
)
CANDIDATE_FILES = ("generator.py", "config.json", "requirements.txt")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def parse_env_file(path: Path) -> tuple[dict[str, str], list[str]]:
    """Read KEY=value pairs from a .env file, collecting quoting problems.

    An unquoted value containing a space makes ``source .env`` execute the
    second word as a command — the exact footgun docs/SETUP.md warns about —
    so it is reported rather than silently accepted.
    """
    values: dict[str, str] = {}
    problems: list[str] = []
    if not path.is_file():
        return values, problems
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        key, sep, value = line.partition("=")
        if not sep or not key.isidentifier():
            continue
        if value[:1] in {'"', "'"}:
            value = value[1:].partition(value[0])[0]
        else:
            value = value.partition(" #")[0].strip()
            if any(c.isspace() for c in value):
                problems.append(
                    f".env line {lineno}: unquoted value with a space in {key} — "
                    "quote it, or `source .env` executes the second word as a command")
                continue
        values[key] = value
    return values, problems


def _env_value(name: str, env_file_values: dict[str, str]) -> str:
    value = os.environ.get(name) or env_file_values.get(name, "")
    return "" if "replace_me" in value else value.strip()


def doctor(root: Path) -> dict[str, Any]:
    """Onboarding checklist: what is ready, what is not, and the fix for each.

    Every check is a local read — no network, no chain, nothing paid — so the
    report is instant and safe to re-run anywhere, but it also means on-chain
    facts (registration, balance) can only ever be notes here.
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str, fix: str | None = None,
              required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail,
                       "fix": None if ok else fix, "required": required})

    cascade_dir = Path(os.environ.get("CASCADE_DIR") or "/root/cascade")
    env_ok = _module_available("cascade") and _module_available("huggingface_hub")
    check("python env", env_ok,
          "this interpreter imports cascade and huggingface_hub" if env_ok
          else "cascade or huggingface_hub not importable from this interpreter",
          f"bash scripts/setup.sh --cascade-dir {cascade_dir} "
          "(then re-run doctor with .venv/bin/python)")

    env_file = root / ".env"
    env_values, env_problems = parse_env_file(env_file)
    missing = [k for k in REQUIRED_ENV_KEYS if not _env_value(k, env_values)]
    if not env_file.is_file() and not any(os.environ.get(k) for k in REQUIRED_ENV_KEYS):
        check("credentials", False, "no .env file",
              "cp example.env .env, fill in the tokens, then: set -a; source .env; set +a")
    elif env_problems:
        check("credentials", False, "; ".join(env_problems),
              "quote every value containing a space in .env")
    else:
        check("credentials", not missing,
              "HF_TOKEN, HIPPIUS_HUB_TOKEN, LIUM_API_KEY, CASCADE_MINER_HOTKEY set"
              if not missing else
              f"still placeholder or unset: {', '.join(missing)}",
              "edit .env and fill in: " + ", ".join(missing) if missing else None)

    cascade_ok = (cascade_dir / "chain.toml").is_file()
    check("cascade checkout", cascade_ok,
          f"read-only reference at {cascade_dir}" if cascade_ok
          else f"no chain.toml under {cascade_dir}",
          f"git clone https://github.com/TensorLink-AI/cascade.git {cascade_dir} "
          "(or export CASCADE_DIR=/path/to/cascade)")

    lium = shutil.which("lium") or (
        str(Path.home() / ".lium/bin/lium")
        if (Path.home() / ".lium/bin/lium").exists() else None)
    check("lium CLI", lium is not None,
          f"found at {lium}" if lium else "not on PATH",
          "curl -fsSL https://lium.io/install.sh | bash  "
          "(then add ~/.lium/bin to PATH)")

    ssh_key = _ssh_key_path()
    check("SSH key", ssh_key.is_file(),
          f"pod-access key at {ssh_key}" if ssh_key.is_file()
          else f"no pod-access key at {ssh_key}",
          f"ssh-keygen -t ed25519 -f {ssh_key} -N '' "
          "(and register the .pub with Lium)")

    snapshots = root / "pools/snapshots"
    have_pool = snapshots.is_dir() and any(snapshots.iterdir())
    check("eval pool", have_pool,
          "snapshot(s) present under pools/snapshots" if have_pool
          else "no local eval-pool snapshot",
          ".venv/bin/python -m miner.controller --sync-pool")

    state = read_json(root / STATE_FILE)
    seeded = bool(state.get("initialized"))
    check("controller state", seeded,
          "runs/controller-state.json seeded" if seeded
          else "runs/controller-state.json missing or not initialized",
          "set -a; source .env; set +a; .venv/bin/python -m miner.controller --once")

    candidate_dir = root / "generators/candidate"
    missing_files = [f for f in CANDIDATE_FILES if not (candidate_dir / f).is_file()]
    check("candidate", not missing_files,
          "generators/candidate has generator.py, config.json, requirements.txt"
          if not missing_files else
          f"generators/candidate missing: {', '.join(missing_files)}",
          "restore the starter candidate (git checkout generators/candidate) "
          "or fetch the king as a base: see 'The loop' in README.md")

    wallet_name = os.environ.get("CASCADE_WALLET_NAME", "").strip()
    wallet_hotkey = os.environ.get("CASCADE_WALLET_HOTKEY", "default").strip()
    wallet_path = Path(os.environ.get("BT_WALLET_PATH")
                       or Path.home() / ".bittensor/wallets")
    if not wallet_name:
        check("wallet", False, "CASCADE_WALLET_NAME not set",
              "export CASCADE_WALLET_NAME=<wallet> so doctor can check the "
              "coldkey/hotkey (creation stays with your reviewed ops/ wrapper)")
    else:
        coldkey = wallet_path / wallet_name / "coldkeypub.txt"
        hotkey = wallet_path / wallet_name / "hotkeys" / wallet_hotkey
        wallet_ok = coldkey.is_file() and hotkey.is_file()
        check("wallet", wallet_ok,
              (f"coldkey and hotkey {wallet_name}/{wallet_hotkey} "
               f"{'found' if wallet_ok else 'not found'} under {wallet_path}"),
              "create the coldkey/hotkey with your reviewed operator wrapper "
              "(ops/create-next-hotkey); registration burns TAO and stays manual")

    approvals = read_json(root / APPROVALS_FILE).get("requests", [])
    pending = [r for r in approvals if isinstance(r, dict)
               and r.get("status") == "pending"]
    check("approvals", not pending,
          "none pending" if not pending else
          f"{len(pending)} pending: " + ", ".join(
              f"[{r.get('id')}] {r.get('action')}" for r in pending),
          ".venv/bin/python -m miner.controller --list-approvals  "
          "(then --approve/--reject <id>)",
          required=False)

    failing = [c for c in checks if not c["ok"] and c["required"]]
    return {
        "checks": checks,
        "ready": not failing,
        "next": failing[0]["fix"] if failing else None,
        "notes": [
            "registration and submission cannot be verified offline: "
            "registration burns TAO, and submissions are one-shot per hotkey",
        ],
    }


def _ssh_key_path() -> Path:
    """Resolve the pod-access key the way miner/pods.py does, without importing
    it — pods.py stays an explicit operator library nothing loads automatically."""
    override = os.environ.get("LIUM_SSH_KEY", "").strip()
    if override:
        return Path(override).expanduser()
    config = Path.home() / ".lium/config.ini"
    if config.is_file():
        import configparser
        parser = configparser.ConfigParser()
        try:
            parser.read(config)
        except configparser.Error:
            parser = None
        if parser is not None:
            wanted = ("ssh_key_path", "ssh_private_key_path", "private_key_path",
                      "ssh_key", "key_path")
            for section in parser.sections():
                for name in wanted:
                    value = parser[section].get(name, "").strip()
                    if value:
                        return Path(value).expanduser()
    return Path.home() / ".ssh/id_ed25519"


def render_doctor(report: dict[str, Any]) -> str:
    lines = ["cascade-miner doctor — "
             + ("ready" if report["ready"] else "not ready")]
    for check in report["checks"]:
        marker = "+" if check["ok"] else ("-" if check["required"] else "!")
        lines.append(f"  {marker} {check['name']:<17} {check['detail']}")
        if not check["ok"] and check["fix"]:
            lines.append(f"        fix: {check['fix']}")
    for note in report["notes"]:
        lines.append(f"  * {note}")
    if report["next"]:
        lines.append(f"next: {report['next']}")
    else:
        lines.append("next: nothing — this host is ready to mine")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--doctor", action="store_true",
                        help="run the onboarding checklist: every stage with "
                             "its fix command; exit 1 until the host is ready")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.doctor:
        report = doctor(root)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_doctor(report))
        return 0 if report["ready"] else 1
    summary = summarize(root)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
