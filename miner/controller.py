"""Continuous Cascade update, round-result, and improvement controller.

The controller detects new information and invokes a user-supplied improvement
command. Human mode routes paid evaluation through a persistent approval queue;
autonomous mode runs a preconfigured evaluator within a hard iteration cap.
On-chain submission is never automatic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOOK_TIMEOUT = 2 * 3600


class CascadeDirty(RuntimeError):
    """The read-only Cascade reference checkout has local modifications."""

    def __init__(self, repo: Path, paths: list[str]):
        self.repo = repo
        self.paths = paths
        joined = ", ".join(paths)
        super().__init__(
            f"Cascade reference checkout {repo} has local modifications: {joined}. "
            "Revert those changes; never commit changes to the reference checkout."
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def candidate_dirty(root: Path) -> bool:
    result = run(
        ["git", "status", "--porcelain", "--", "generators/candidate",
         "notes/EXPERIMENTS.md"],
        cwd=root,
    )
    return bool(result.stdout.strip())


def candidate_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (Path("generators/candidate"), Path("notes/EXPERIMENTS.md")):
        path = root / relative
        if path.is_file():
            digest.update(str(relative).encode())
            digest.update(path.read_bytes())
        elif path.is_dir():
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                digest.update(str(child.relative_to(root)).encode())
                digest.update(child.read_bytes())
    return digest.hexdigest()


def queue_approval(path: Path, *, action: str, reason: str,
                   context: dict[str, Any]) -> dict[str, Any]:
    doc = read_json(path)
    requests = doc.get("requests", [])
    if not isinstance(requests, list):
        requests = []
    fingerprint = hashlib.sha256(json.dumps(
        {"action": action, "context": context}, sort_keys=True,
    ).encode()).hexdigest()[:12]
    for request in requests:
        if request.get("id") == fingerprint and request.get("status") == "pending":
            return request
    request = {
        "id": fingerprint,
        "action": action,
        "reason": reason,
        "context": context,
        "status": "pending",
        "created_at": utc_now(),
    }
    requests.append(request)
    write_json(path, {"requests": requests})
    return request


def set_approval_status(path: Path, request_id: str, status: str) -> dict[str, Any]:
    doc = read_json(path)
    requests = doc.get("requests", [])
    for request in requests:
        if request.get("id") == request_id:
            if request.get("status") != "pending":
                raise RuntimeError(
                    f"approval {request_id} is already {request.get('status')}"
                )
            request["status"] = status
            request[f"{status}_at"] = utc_now()
            write_json(path, {"requests": requests})
            return request
    raise RuntimeError(f"unknown approval id {request_id}")


def run(argv: list[str], *, cwd: Path, timeout: int = 900,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env,
    )


def absolute_path(path: Path) -> Path:
    """Make a path absolute without dereferencing a venv interpreter symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def assert_runtime(python: Path, root: Path,
                   modules: tuple[str, ...] = ("cascade", "huggingface_hub")) -> None:
    imports = ", ".join(modules)
    try:
        result = run([str(python), "-c", f"import {imports}"], cwd=root, timeout=30)
    except OSError as exc:
        raise RuntimeError(
            f"Cascade runtime check failed for interpreter {python}: {exc}"
        ) from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(
            f"Cascade runtime check failed for interpreter {python}: {detail}"
        )


def git_head(repo: Path) -> str:
    result = run(["git", "rev-parse", "HEAD"], cwd=repo)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "cannot read Cascade git HEAD")
    return result.stdout.strip()


def refresh_cascade(repo: Path) -> tuple[str, str]:
    """Fast-forward the read-only Cascade checkout, rejecting local changes."""
    old = git_head(repo)
    status = run(["git", "status", "--porcelain"], cwd=repo)
    if status.returncode:
        raise RuntimeError(status.stderr.strip() or "cannot inspect Cascade checkout")
    if status.stdout.strip():
        paths = [line[3:] for line in status.stdout.splitlines()[:25]]
        raise CascadeDirty(repo, paths)
    fetch = run(["git", "fetch", "--prune", "origin"], cwd=repo, timeout=300)
    if fetch.returncode:
        raise RuntimeError(f"Cascade fetch failed: {fetch.stderr.strip()}")
    branch = run(["git", "branch", "--show-current"], cwd=repo).stdout.strip()
    if not branch:
        raise RuntimeError("Cascade checkout is detached; refusing to update it")
    merge = run(["git", "merge", "--ff-only", f"origin/{branch}"], cwd=repo, timeout=300)
    if merge.returncode:
        raise RuntimeError(f"Cascade fast-forward failed: {merge.stderr.strip()}")
    return old, git_head(repo)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    out: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        out.update(flatten(child, name))
    return out


def chain_snapshot(path: Path) -> dict[str, Any]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return {"digest": file_digest(path), "values": flatten(raw)}


def changed_values(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old = before.get("values", {}) if isinstance(before, dict) else {}
    new = after.get("values", {}) if isinstance(after, dict) else {}
    keys = set(old) | set(new)
    return {
        key: {"before": old.get(key), "after": new.get(key)}
        for key in sorted(keys)
        if old.get(key) != new.get(key)
    }


def audit_latest(python: Path, cascade_dir: Path, chain_toml: Path,
                 network: str, hotkey: str = "") -> dict[str, Any]:
    result = run(
        [str(python), "-m", "cascade.audit.main", "latest", "--tier", "0",
         "--json", "--config", str(chain_toml), "--network", network],
        cwd=cascade_dir, timeout=600,
    )
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"latest receipt audit failed: {detail}") from exc
    if not isinstance(payload, dict) or not payload.get("round_id"):
        raise RuntimeError("latest receipt audit returned no round id")
    payload["audit_exit_code"] = result.returncode
    detail_script = """
import json
import sys
from cascade.audit.main import fetch_receipt_text
from cascade.shared.config import load_chain_config
from cascade.shared.receipt import load_receipt, summarize_receipt
cfg = load_chain_config(sys.argv[1])
doc = json.loads(fetch_receipt_text(cfg, None))
receipt = load_receipt(json.dumps(doc))
out = {"summary": summarize_receipt(receipt)}
hotkey = sys.argv[2]
if hotkey:
    out["miner_participant"] = next(
        (p for p in doc.get("participants", []) if p.get("hotkey") == hotkey), None
    )
    heat = doc.get("manifest", {}).get("heat") or {}
    out["miner_heat"] = next(
        (p for p in heat.get("entrants", []) if p.get("hotkey") == hotkey), None
    )
print(json.dumps(out))
"""
    detail = run(
        [str(python), "-c", detail_script, str(chain_toml), hotkey],
        cwd=cascade_dir, timeout=600,
    )
    if detail.returncode == 0:
        try:
            payload.update(json.loads(detail.stdout.strip().splitlines()[-1]))
        except (IndexError, ValueError):
            pass
    return payload


def sync_eval_pool(python: Path, root: Path, repo_id: str, dest: Path,
                   known_revision: str = "") -> dict[str, Any]:
    """Check the dataset revision and download only when it changed."""
    script = """
import json
import os
import sys
from huggingface_hub import HfApi, snapshot_download
repo_id, dest, known = sys.argv[1], sys.argv[2], sys.argv[3]
token = os.environ.get("HF_TOKEN") or None
info = HfApi(token=token).dataset_info(repo_id=repo_id)
downloaded = info.sha != known
if downloaded:
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=info.sha,
                      local_dir=dest, token=token)
print(json.dumps({"repo_id": repo_id, "revision": info.sha, "dest": dest,
                  "downloaded": downloaded}))
"""
    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    result = run(
        [str(python), "-c", script, repo_id, str(dest), known_revision],
        cwd=root, timeout=3600, env=env,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(f"eval-pool sync failed: {detail}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("eval-pool sync returned no revision") from exc
    return payload


def run_hook(command: str, *, root: Path, event: dict[str, Any],
             state_file: Path, chain_toml: Path,
             timeout: int = DEFAULT_HOOK_TIMEOUT) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({
        "CASCADE_MINER_EVENT": json.dumps(event, sort_keys=True),
        "CASCADE_MINER_EVENT_TYPE": str(event.get("type", "")),
        "CASCADE_MINER_ROUND_ID": str(event.get("round_id", "")),
        "CASCADE_MINER_STATE": str(state_file.resolve()),
        "CASCADE_CHAIN_TOML": str(chain_toml.resolve()),
        "CASCADE_MINER_MODE": str(event.get("mode", "human")),
    })
    try:
        result = run(shlex.split(command), cwd=root, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": 124,
            "stdout": (exc.stdout or "")[-4000:],
            "stderr": f"timed out after {timeout}s",
        }
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


@dataclass
class Controller:
    root: Path
    cascade_dir: Path
    chain_toml: Path
    cascade_python: Path
    state_file: Path
    events_file: Path
    approvals_file: Path
    network: str = "finney"
    hotkey: str = ""
    eval_pool_repo: str = "Tensor-Link/cascade-eval-pool"
    eval_pool_dir: Path = Path("pools")
    improve_command: str = ""
    sync_command: str = ""
    notify_command: str = ""
    approved_eval_command: str = ""
    create_hotkey_command: str = ""
    register_hotkey_command: str = ""
    submit_command: str = ""
    autonomous_actions: tuple[str, ...] = ("gpu_evaluation",)
    mode: str = "human"
    hook_timeout: int = DEFAULT_HOOK_TIMEOUT
    max_improvements_per_round: int = 1

    def action_command(self, action: str) -> str:
        return {
            "gpu_evaluation": self.approved_eval_command,
            "create_hotkey": self.create_hotkey_command,
            "register_hotkey": self.register_hotkey_command,
            "submit_candidate": self.submit_command,
        }.get(action, "")

    def notify(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if not self.notify_command:
            return None
        return run_hook(
            self.notify_command, root=self.root, event=event,
            state_file=self.state_file, chain_toml=self.chain_toml, timeout=60,
        )

    def process_approved(self) -> list[dict[str, Any]]:
        doc = read_json(self.approvals_file)
        requests = doc.get("requests", [])
        processed: list[dict[str, Any]] = []
        changed = False
        for request in requests:
            if request.get("status") != "approved":
                continue
            changed = True
            action = str(request.get("action", ""))
            command = self.action_command(action)
            if action == "candidate_review":
                request["status"] = "completed"
            elif not command:
                request["status"] = "blocked"
                request["error"] = f"no command is configured for {action}"
            else:
                result = run_hook(
                    command, root=self.root,
                    event={"type": "approved_action", "approval": request},
                    state_file=self.state_file, chain_toml=self.chain_toml,
                    timeout=24 * 3600,
                )
                request["result"] = result
                request["status"] = "completed" if result["exit_code"] == 0 else "failed"
            request["finished_at"] = utc_now()
            event = {"type": "approval_result", "at": utc_now(), "approval": request}
            append_event(self.events_file, event)
            self.notify(event)
            processed.append(event)
        if changed:
            write_json(self.approvals_file, {"requests": requests})
        return processed

    def cycle(self) -> list[dict[str, Any]]:
        state = read_json(self.state_file)
        first_run = not bool(state.get("initialized"))
        events: list[dict[str, Any]] = []
        self.process_approved()
        previous_chain = state.get("chain", {})

        old_head, new_head = refresh_cascade(self.cascade_dir)
        current_chain = chain_snapshot(self.chain_toml)
        if new_head != state.get("cascade_head") or (
            previous_chain and current_chain["digest"] != previous_chain.get("digest")
        ):
            event = {
                "type": "cascade_update",
                "at": utc_now(),
                "old_head": state.get("cascade_head") or old_head,
                "new_head": new_head,
                "chain_changes": changed_values(previous_chain, current_chain),
            }
            events.append(event)

        pool = sync_eval_pool(
            self.cascade_python, self.root, self.eval_pool_repo, self.eval_pool_dir,
            (str(state.get("eval_pool_revision", ""))
             if (self.eval_pool_dir / "snapshots").is_dir() else ""),
        )
        if pool["revision"] != state.get("eval_pool_revision"):
            events.append({
                "type": "eval_pool_update",
                "at": utc_now(),
                "repo_id": pool["repo_id"],
                "old_revision": state.get("eval_pool_revision"),
                "new_revision": pool["revision"],
                "path": str(self.eval_pool_dir),
            })

        receipt = audit_latest(
            self.cascade_python, self.cascade_dir, self.chain_toml, self.network, self.hotkey,
        )
        round_id = str(receipt["round_id"])
        if round_id != str(state.get("last_round_id", "")):
            events.append({
                "type": "round_result",
                "at": utc_now(),
                "round_id": round_id,
                "status": receipt.get("status"),
                "audit_ok": receipt.get("ok"),
                "audit_exit_code": receipt.get("audit_exit_code"),
                "checks": receipt.get("checks", []),
                "summary": receipt.get("summary", {}),
                "miner_participant": receipt.get("miner_participant"),
                "miner_heat": receipt.get("miner_heat"),
            })

        for event in events:
            append_event(self.events_file, event)
            print(json.dumps(event, indent=2, sort_keys=True), flush=True)

        # A restart must not rediscover these events even if a hook crashes.
        state.update({
            "initialized": True,
            "updated_at": utc_now(),
            "cascade_head": new_head,
            "chain": current_chain,
            "eval_pool_repo": pool["repo_id"],
            "eval_pool_revision": pool["revision"],
            "last_round_id": round_id,
            "last_receipt": receipt,
        })
        write_json(self.state_file, state)

        if first_run:
            seeded = {
                "type": "state_seeded",
                "at": utc_now(),
                "events_observed": [event["type"] for event in events],
            }
            append_event(self.events_file, seeded)
            return events

        if self.sync_command and old_head != new_head:
            sync_event = {
                "type": "cascade_sync_started", "at": utc_now(),
                "old_head": old_head, "new_head": new_head,
            }
            append_event(self.events_file, sync_event)
            result = run_hook(
                self.sync_command, root=self.root, event=sync_event,
                state_file=self.state_file, chain_toml=self.chain_toml,
                timeout=self.hook_timeout,
            )
            append_event(self.events_file, {
                "type": "cascade_sync_result", "at": utc_now(), "result": result,
            })

        if events and self.improve_command:
            key = round_id or hashlib.sha256(
                json.dumps(events, sort_keys=True).encode()
            ).hexdigest()[:12]
            passes = state.setdefault("improvement_passes", {})
            count = int(passes.get(key, 0))
            if candidate_dirty(self.root) and self.mode == "human":
                request = queue_approval(
                    self.approvals_file, action="candidate_review",
                    reason="Candidate changes are already awaiting human review.",
                    context={"round_id": round_id,
                             "candidate_digest": candidate_digest(self.root)},
                )
                approval_event = {"type": "approval_required", "approval": request}
                append_event(self.events_file, approval_event)
                self.notify(approval_event)
            while (self.mode == "autonomous" or not candidate_dirty(self.root)) and (
                count < self.max_improvements_per_round
            ):
                batch = {
                    "type": "improvement_batch",
                    "mode": self.mode,
                    "at": utc_now(),
                    "round_id": round_id,
                    "iteration": count + 1,
                    "events": events,
                }
                passes[key] = count + 1
                write_json(self.state_file, state)
                append_event(self.events_file, {
                    "type": "improvement_started",
                    "at": utc_now(),
                    "round_id": round_id,
                    "iteration": count + 1,
                })
                improvement = run_hook(
                    self.improve_command, root=self.root, event=batch,
                    state_file=self.state_file, chain_toml=self.chain_toml,
                    timeout=self.hook_timeout,
                )
                count += 1
                result_event = {
                    "type": "improvement_result",
                    "at": utc_now(),
                    "round_id": round_id,
                    "iteration": count,
                    "result": improvement,
                }
                append_event(self.events_file, result_event)
                if improvement["exit_code"] != 0:
                    break

                request_path = self.root / "runs/agent-request.json"
                request_data = read_json(request_path)
                request_path.unlink(missing_ok=True)
                dirty = candidate_dirty(self.root)
                action = str(request_data.get("action") or
                             ("gpu_evaluation" if dirty else ""))
                if not action:
                    break
                allowed_actions = {
                    "gpu_evaluation", "create_hotkey", "register_hotkey",
                    "submit_candidate",
                }
                if action not in allowed_actions:
                    error_event = {
                        "type": "agent_request_rejected", "at": utc_now(),
                        "action": action, "reason": "unknown action",
                    }
                    append_event(self.events_file, error_event)
                    self.notify(error_event)
                    break
                reason = str(request_data.get("reason") or
                             f"Agent requested {action}.")
                context = {
                    "round_id": round_id,
                    "pass": passes[key],
                    "candidate_digest": candidate_digest(self.root),
                    "candidate_path": request_data.get(
                        "candidate_path", "generators/candidate"
                    ),
                    "estimated_hours": request_data.get("estimated_hours"),
                }
                if self.mode == "human" or action not in self.autonomous_actions:
                    request = queue_approval(
                        self.approvals_file, action=action, reason=reason,
                        context=context,
                    )
                    approval_event = {"type": "approval_required", "approval": request}
                    append_event(self.events_file, approval_event)
                    print(json.dumps(approval_event, indent=2), flush=True)
                    self.notify(approval_event)
                    break

                evaluation_event = {
                    "type": "autonomous_action",
                    "mode": self.mode,
                    "at": utc_now(),
                    "action": action,
                    "reason": reason,
                    "context": context,
                }
                command = self.action_command(action)
                if not command:
                    evaluation_event["result"] = {
                        "exit_code": 2,
                        "stderr": f"no command is configured for {action}",
                    }
                else:
                    evaluation_event["result"] = run_hook(
                        command, root=self.root,
                        event=evaluation_event, state_file=self.state_file,
                        chain_toml=self.chain_toml, timeout=24 * 3600,
                    )
                append_event(self.events_file, evaluation_event)
                print(json.dumps(evaluation_event, indent=2), flush=True)
                self.notify(evaluation_event)
                if evaluation_event["result"]["exit_code"] != 0:
                    break
                # Never invoke the improvement hook more than once in one poll.
                # Further bounded passes require a later, newly detected event.
                break

        write_json(self.state_file, state)
        return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cascade-dir", type=Path, default=Path("/root/cascade"))
    parser.add_argument("--chain-toml", type=Path, default=None)
    parser.add_argument("--cascade-python", type=Path, default=None)
    parser.add_argument("--state-file", type=Path, default=Path("runs/controller-state.json"))
    parser.add_argument("--events-file", type=Path, default=Path("runs/controller-events.jsonl"))
    parser.add_argument("--approvals-file", type=Path, default=Path("runs/approvals.json"))
    parser.add_argument("--network", default="finney")
    parser.add_argument("--mode", choices=("human", "autonomous"),
                        default=os.environ.get("CASCADE_MODE", "human"),
                        help="Human approval queue or bounded autonomous evaluation.")
    parser.add_argument("--hotkey", default=os.environ.get("CASCADE_MINER_HOTKEY", ""),
                        help="Miner SS58 hotkey used to extract your heat result.")
    parser.add_argument("--eval-pool-repo", default="Tensor-Link/cascade-eval-pool")
    parser.add_argument("--eval-pool-dir", type=Path, default=Path("pools"))
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--hook-timeout", "--agent-timeout", dest="hook_timeout",
                        type=int, default=DEFAULT_HOOK_TIMEOUT)
    parser.add_argument("--max-improvements-per-round", "--max-improvements-per-event",
                        dest="max_improvements_per_round", type=int, default=1)
    parser.add_argument("--list-approvals", action="store_true")
    parser.add_argument("--approve", metavar="ID")
    parser.add_argument("--reject", metavar="ID")
    parser.add_argument(
        "--improve-command", default=os.environ.get("CASCADE_IMPROVE_COMMAND", ""),
        help="Command run once for each Cascade update or new round receipt.",
    )
    parser.add_argument(
        "--sync-command", default=os.environ.get("CASCADE_SYNC_COMMAND", ""),
        help="Optional dependency reinstall command run after Cascade code updates.",
    )
    parser.add_argument(
        "--notify-command", default=os.environ.get("CASCADE_NOTIFY_COMMAND", ""),
        help="Optional command that surfaces approval requests/results to a human.",
    )
    parser.add_argument(
        "--approved-eval-command",
        default=os.environ.get("CASCADE_APPROVED_EVAL_COMMAND", ""),
        help="GPU evaluation command: approved-only in human mode, automatic in autonomous mode.",
    )
    parser.add_argument(
        "--create-hotkey-command", default=os.environ.get("CASCADE_CREATE_HOTKEY_COMMAND", ""),
        help="Command that creates the next configured Bittensor hotkey.",
    )
    parser.add_argument(
        "--register-hotkey-command",
        default=os.environ.get("CASCADE_REGISTER_HOTKEY_COMMAND", ""),
        help="Command that registers the configured hotkey on the subnet.",
    )
    parser.add_argument(
        "--submit-command", default=os.environ.get("CASCADE_SUBMIT_COMMAND", ""),
        help="Command that verifies, uploads, and submits the candidate on-chain.",
    )
    parser.add_argument(
        "--autonomous-actions",
        default=os.environ.get("CASCADE_AUTONOMOUS_ACTIONS", "gpu_evaluation"),
        help="Comma-separated named actions autonomous mode may execute.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    cascade_dir = args.cascade_dir.resolve()
    approvals_file = ((root / args.approvals_file).resolve()
                      if not args.approvals_file.is_absolute() else args.approvals_file)
    if args.list_approvals:
        requests = read_json(approvals_file).get("requests", [])
        print(json.dumps(requests, indent=2, sort_keys=True))
        return 0
    if args.approve or args.reject:
        request = set_approval_status(
            approvals_file, args.approve or args.reject,
            "approved" if args.approve else "rejected",
        )
        print(json.dumps(request, indent=2, sort_keys=True))
        return 0
    supported_actions = {
        "gpu_evaluation", "create_hotkey", "register_hotkey", "submit_candidate",
    }
    autonomous_actions = tuple(
        action.strip() for action in args.autonomous_actions.split(",") if action.strip()
    )
    unknown_actions = set(autonomous_actions) - supported_actions
    if unknown_actions:
        raise SystemExit(
            "unknown autonomous actions: " + ", ".join(sorted(unknown_actions))
        )
    cascade_python = absolute_path(args.cascade_python or root / ".venv/bin/python")
    assert_runtime(cascade_python, root)
    controller = Controller(
        root=root,
        cascade_dir=cascade_dir,
        chain_toml=(args.chain_toml or cascade_dir / "chain.toml").resolve(),
        cascade_python=cascade_python,
        state_file=(root / args.state_file).resolve() if not args.state_file.is_absolute()
                   else args.state_file,
        events_file=(root / args.events_file).resolve() if not args.events_file.is_absolute()
                    else args.events_file,
        approvals_file=approvals_file,
        network=args.network,
        hotkey=args.hotkey,
        eval_pool_repo=args.eval_pool_repo,
        eval_pool_dir=(root / args.eval_pool_dir).resolve()
                      if not args.eval_pool_dir.is_absolute() else args.eval_pool_dir,
        improve_command=args.improve_command,
        sync_command=args.sync_command,
        notify_command=args.notify_command,
        approved_eval_command=args.approved_eval_command,
        create_hotkey_command=args.create_hotkey_command,
        register_hotkey_command=args.register_hotkey_command,
        submit_command=args.submit_command,
        autonomous_actions=autonomous_actions,
        mode=args.mode,
        hook_timeout=max(60, args.hook_timeout),
        max_improvements_per_round=max(0, args.max_improvements_per_round),
    )
    while True:
        try:
            controller.cycle()
        except CascadeDirty as exc:
            event = {
                "type": "cascade_dirty",
                "at": utc_now(),
                "error": str(exc),
                "paths": exc.paths,
                "action": (
                    f"revert local changes under {exc.repo} and restart the controller; "
                    "never commit them"
                ),
            }
            append_event(controller.events_file, event)
            print(json.dumps(event), file=sys.stderr, flush=True)
            return 1
        except Exception as exc:  # noqa: BLE001 - daemon must survive transient failures
            event = {"type": "controller_error", "at": utc_now(), "error": repr(exc)}
            append_event(controller.events_file, event)
            print(json.dumps(event), file=sys.stderr, flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
