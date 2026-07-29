"""Run one bounded generator-improvement pass with a supported agent CLI."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


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
4. Run static verification and inexpensive deterministic checks. When another
   capability is needed, write runs/agent-request.json with one action from
   gpu_evaluation, create_hotkey, register_hotkey, or submit_candidate, plus a
   reason, candidate_path, and optional estimated_hours. Example:
   {{"action":"gpu_evaluation","reason":"...",
     "candidate_path":"generators/candidate","estimated_hours":1}}.
   Never execute these privileged actions yourself.
5. Record the hypothesis and checks in notes/EXPERIMENTS.md without claiming an
   unmeasured score improvement.

Keep the repository focused and leave the changes in the working tree for
human review. Never read or print wallet secrets, deploy, submit on-chain,
commit, push, or expose private evaluation data yourself.
"""


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
    raise RuntimeError("no supported agent CLI found; configure CASCADE_AGENT_COMMAND")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    event = os.environ.get("CASCADE_MINER_EVENT", "{}")
    agent = choose_agent(os.environ.get("CASCADE_AGENT", "auto").strip().lower())
    mode = os.environ.get("CASCADE_MINER_MODE", "human").strip().lower()
    argv, stdin = agent_command(agent, root, build_prompt(root, event, mode))
    print(f"running improvement pass with {agent}", flush=True)
    result = subprocess.run(argv, input=stdin, text=True, cwd=root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
