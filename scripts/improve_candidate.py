"""Run one bounded generator-improvement pass with a supported agent backend.

Most backends spawn a non-interactive CLI. ``hermes-native`` is different: an
agent that is *already* the session has no CLI to call itself with, so the pass
is handed over as a file request and this process waits for the reply.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

REQUEST_NAME = "runs/improve-request.json"
RESPONSE_NAME = "runs/improve-response.json"
DEFAULT_RESPONSE_TIMEOUT = 3600
DEFAULT_RESPONSE_POLL = 5
TERMINAL_STATUSES = {"completed", "failed", "rejected", "skipped"}


def build_prompt(root: Path, event: str, mode: str = "human") -> str:
    return f"""You are maintaining the Cascade miner harness in {root}.

Read AGENTS.md and CLAUDE.md before acting.

A new controller event arrived:
{event}

Controller mode: {mode}. In human mode, paid evaluation pauses for approval. In
autonomous mode, the controller may run its preconfigured evaluation command.
In both modes, you only request evaluation; never rent infrastructure directly.

Perform one bounded improvement pass:
1. Read the current Cascade chain contract at $CASCADE_CHAIN_TOML and relevant
   upstream miner documentation.
2. Inspect the newest round outcome and local eval-pool revision from the event
   and controller state.
3. Create or improve a single deployable candidate under generators/candidate.
   A candidate must contain generator.py, config.json, and requirements.txt.
4. Run static verification and inexpensive deterministic checks.
5. Screen the corpus before asking for paid work:
   python -m miner.screen generators/candidate --king generators/king-control
   --claim <family>+ for each property you believe the change adds. A `blocked`
   verdict must be fixed; `undosed` means the change is too small for a paired
   eval to resolve, so raise the dose instead of requesting one; a failing claim
   means the generator does not do what you think. A `measurable` verdict
   licenses the request and is not evidence of an improvement. Quote its numbers
   in the request's reason.
6. When another capability is needed, write runs/agent-request.json with one action from
   gpu_evaluation, create_hotkey, register_hotkey, or submit_candidate, plus a
   reason, candidate_path, and optional estimated_hours. Example:
   {{"action":"gpu_evaluation","reason":"...",
     "candidate_path":"generators/candidate","estimated_hours":1}}.
   Never execute these privileged actions yourself.
7. Record the hypothesis and checks in notes/EXPERIMENTS.md without claiming an
   unmeasured score improvement.

Keep the repository focused and leave the changes in the working tree for
human review. Never read or print wallet secrets, deploy, submit on-chain,
commit, push, or expose private evaluation data yourself.
"""


def read_json(path: Path) -> dict:
    """Tolerate a partially-written file; the writer is another process."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    """Write atomically so a poller never observes a half-written request."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def request_improvement(root: Path, prompt: str, event: str, mode: str,
                        *, timeout: int, poll: int, say=print) -> int:
    """Hand the pass to the surrounding agent session and wait for its reply.

    ``CASCADE_AGENT=hermes-native`` exists because the improvement hook used to
    shell out to a ``hermes`` binary that does not exist inside a Hermes
    session — the agent cannot invoke itself as a CLI. Instead the prompt is
    published to ``runs/improve-request.json``; the session picks it up and
    answers in ``runs/improve-response.json``.
    """
    request_path, response_path = root / REQUEST_NAME, root / RESPONSE_NAME
    request_id = f"{int(time.time())}-{os.getpid()}"
    # Never let a previous cycle's reply satisfy this request.
    response_path.unlink(missing_ok=True)
    write_json(request_path, {
        "id": request_id,
        "status": "pending",
        "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "root": str(root),
        "event": read_json_string(event),
        "prompt": prompt,
    })
    say(f"improvement request {request_id} written to {REQUEST_NAME}; "
        f"waiting up to {timeout}s for {RESPONSE_NAME}", flush=True)

    deadline = time.monotonic() + timeout
    while True:
        response = read_json(response_path)
        status = str(response.get("status", "")).strip().lower()
        if response.get("id") == request_id and status in TERMINAL_STATUSES:
            detail = str(response.get("detail", "")).strip()
            say(f"improvement request {request_id}: {status}"
                + (f" — {detail}" if detail else ""), flush=True)
            request_path.unlink(missing_ok=True)
            return 0 if status == "completed" else 1
        if time.monotonic() >= deadline:
            # Leave the request in place: the session may still be working, and
            # a stale pending file is the only evidence of what was asked.
            say(f"improvement request {request_id} timed out after {timeout}s; "
                f"{REQUEST_NAME} left in place", flush=True)
            return 124
        time.sleep(poll)


def read_json_string(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def agent_command(agent: str, root: Path, prompt: str) -> tuple[list[str], str | None]:
    if agent == "codex":
        return (["codex", "exec", "--ephemeral", "--sandbox", "workspace-write",
                 "-C", str(root), "-"], prompt)
    if agent == "claude":
        return (["claude", "--print", "--permission-mode", "acceptEdits",
                 "--no-session-persistence", prompt], None)
    if agent == "hermes":
        return (["hermes", "chat", "--toolsets", "terminal", "-q", prompt], None)
    if agent == "custom":
        template = os.environ.get("CASCADE_AGENT_COMMAND", "").strip()
        if not template:
            raise RuntimeError("CASCADE_AGENT=custom requires CASCADE_AGENT_COMMAND")
        argv = shlex.split(template)
        if any("{prompt}" in arg for arg in argv):
            return ([arg.replace("{prompt}", prompt) for arg in argv], None)
        return (argv, prompt)
    raise RuntimeError(f"unsupported CASCADE_AGENT={agent!r}")


def choose_agent(requested: str) -> str:
    if requested != "auto":
        return requested
    for candidate in ("claude", "codex", "hermes"):
        if shutil.which(candidate):
            return candidate
    if os.environ.get("CASCADE_AGENT_COMMAND"):
        return "custom"
    # Deliberately not auto-detected: "no CLI on PATH" is a broken install far
    # more often than it is an in-session agent, and the file handshake blocks
    # for an hour before anyone finds out.
    raise RuntimeError(
        "no supported agent CLI found; configure CASCADE_AGENT_COMMAND, or set "
        "CASCADE_AGENT=hermes-native when running inside an agent session"
    )


def positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    event = os.environ.get("CASCADE_MINER_EVENT", "{}")
    agent = choose_agent(os.environ.get("CASCADE_AGENT", "auto").strip().lower())
    mode = os.environ.get("CASCADE_MINER_MODE", "human").strip().lower()
    prompt = build_prompt(root, event, mode)
    if agent == "hermes-native":
        print("requesting improvement pass from the surrounding agent session",
              flush=True)
        return request_improvement(
            root, prompt, event, mode,
            timeout=positive_int("CASCADE_IMPROVE_RESPONSE_TIMEOUT",
                                 DEFAULT_RESPONSE_TIMEOUT),
            poll=positive_int("CASCADE_IMPROVE_RESPONSE_POLL_SECONDS",
                              DEFAULT_RESPONSE_POLL),
        )
    argv, stdin = agent_command(agent, root, prompt)
    print(f"running improvement pass with {agent}", flush=True)
    result = subprocess.run(argv, input=stdin, text=True, cwd=root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
