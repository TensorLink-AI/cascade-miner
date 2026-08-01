"""Rent and provision Lium GPU pods, with guards for every way this bit us.

Each guard below exists because its absence cost real time or money on the
cascade subnet during development:

* ``-c 1`` is ALWAYS passed. Renting ``--gpu RTX4090`` alone matched an 8-GPU
  node at $2.56/h for a strictly single-GPU job (~$6 wasted before it was
  spotted).
* Remote commands use ABSOLUTE paths. ssh lands in ``/root``, not the project
  dir, so ``./.venv/bin/python`` and ``uv pip install --python .venv/...``
  silently resolve to nothing. This produced three separate incidents,
  including one where ``uv pip install`` "succeeded" into a non-existent
  interpreter and three pods sat idle for an hour.
* ``assert_deps`` runs ``import torch`` on the pod and REFUSES to launch
  otherwise. A silent no-op install is otherwise invisible until the run
  errors on every seed.
* ``check_ttl_covers`` refuses to start work that cannot finish before the
  pod's TTL. Two full experiments were lost to pods expiring mid-batch.
* Transfers use ``tar`` over ssh rather than ``rsync``. Minimal container
  images ship without rsync and often cannot install it.

The results puller is deliberately NOT optional; see :func:`start_puller`.
"""

from __future__ import annotations

import configparser
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def _lium_configured_key() -> Path | None:
    """Read the private key the lium CLI already uses, if it recorded one."""
    config = Path.home() / ".lium/config.ini"
    if not config.is_file():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(config)
    except configparser.Error:
        return None
    wanted = ("ssh_key_path", "ssh_private_key_path", "private_key_path",
              "ssh_key", "key_path")
    for section in parser.sections():
        for name in wanted:
            value = parser[section].get(name, "").strip()
            if value:
                return Path(value).expanduser()
    return None


def ssh_key_path() -> Path:
    """Private key for pod access: env override, then lium's own, then default.

    Hardcoding ``/root/.ssh/id_ed25519`` locked out every non-root operator.
    """
    override = os.environ.get("LIUM_SSH_KEY", "").strip()
    if override:
        return Path(override).expanduser()
    configured = _lium_configured_key()
    if configured is not None:
        return configured
    return Path.home() / ".ssh/id_ed25519"


SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=30",
    "-i", str(ssh_key_path()),
]
REMOTE_ROOT = "/root/cascade-miner"   # absolute, always
REMOTE_PY = f"{REMOTE_ROOT}/.venv/bin/python"


@dataclass(frozen=True)
class Pod:
    name: str
    ip: str
    port: str
    price_per_hour: float
    ttl_iso: str

    def ssh(self, cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
        """Run a command on the pod. `cmd` MUST use absolute paths."""
        return subprocess.run(
            ["ssh", *SSH_OPTS, "-p", self.port, f"root@{self.ip}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )

    def push(self, local: str, remote: str, timeout: int = 900) -> None:
        ssh_opts = ' '.join(shlex.quote(o) for o in SSH_OPTS)
        local_clean = local.rstrip('/')
        remote_clean = remote.rstrip('/')
        cmd = (f"tar -cf - --exclude=__pycache__ -C {shlex.quote(local_clean)} . | "
               f"ssh {ssh_opts} -p {self.port} root@{self.ip} "
               f"'mkdir -p {remote_clean} && tar -xf - -C {remote_clean}'")
        subprocess.run(["bash", "-c", cmd], check=True, timeout=timeout)

    def pull(self, remote: str, local: str, timeout: int = 900) -> None:
        ssh_opts = ' '.join(shlex.quote(o) for o in SSH_OPTS)
        os.makedirs(local, exist_ok=True)
        cmd = (f"ssh {ssh_opts} -p {self.port} root@{self.ip} "
               f"'tar -cf - -C {remote} .' | tar -xf - -C {shlex.quote(local)}")
        subprocess.run(["bash", "-c", cmd], timeout=timeout, check=False)


def _lium(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    executable = shutil.which("lium")
    if executable is None:
        raise RuntimeError("lium CLI not found — see README setup")
    return subprocess.run(
        [executable, *args], capture_output=True, text=True, timeout=timeout,
    )


def rent(name: str, ttl_hours: int = 10, gpu: str = "RTX4090", retries: int = 8) -> None:
    """Rent ONE single-GPU pod. `-c 1` is not optional — see module docstring.

    Lium serialises rentals account-wide, so concurrent calls fail with
    "Another rental is already in progress"; retry rather than parallelise.
    """
    for attempt in range(1, retries + 1):
        out = _lium("up", "--gpu", gpu, "-c", "1", "--name", name,
                    "--ttl", f"{ttl_hours}h", "-y")
        blob = (out.stdout + out.stderr).lower()
        if any(s in blob for s in ("already in progress", "not found", "no nodes available")):
            time.sleep(45)
            continue
        if out.returncode:
            raise RuntimeError(
                f"lium failed to rent {name}: {(out.stderr or out.stdout).strip()[-2000:]}"
            )
        return
    raise RuntimeError(f"could not rent {name} after {retries} attempts")


def stop(name: str) -> None:
    """Terminate exactly one named pod and fail if Lium does not confirm it."""
    out = _lium("rm", name, timeout=300)
    if out.returncode:
        raise RuntimeError(
            f"lium failed to stop {name}: {(out.stderr or out.stdout).strip()[-2000:]}"
        )


def list_pods() -> list[Pod]:
    out = _lium("ps", "--format", "json", timeout=300)
    pods = []
    for r in json.loads(out.stdout or "[]"):
        cmd = r.get("ssh_cmd") or ""
        if " -p " not in cmd:
            continue
        pods.append(Pod(
            name=r["name"],
            ip=cmd.split("root@")[1].split(" ")[0],
            port=cmd.split(" -p ")[-1].strip(),
            price_per_hour=float(r.get("price_per_hour", 0.0)),
            ttl_iso=str(r.get("removal_scheduled_at", "")),
        ))
    return pods


def assert_single_gpu() -> None:
    """Fail loudly if any pod is multi-GPU — we only ever use one."""
    bad = []
    for r in json.loads(_lium("ps", "--format", "json", timeout=300).stdout or "[]"):
        if int(r.get("gpu_count", 1)) > 1:
            bad.append((r["name"], r.get("config"), r.get("price_per_hour")))
    if bad:
        raise RuntimeError(f"multi-GPU pods rented (wasting money): {bad}")


def provision(pod: Pod, cascade_dir: str, gens: list[str], pools_dir: str) -> None:
    """Sync the project and build the venv. All remote paths absolute."""
    pod.ssh(f"mkdir -p {REMOTE_ROOT}/generators {REMOTE_ROOT}/pools")
    pod.push(f"{cascade_dir}/", "/root/cascade/")
    for g in gens:
        pod.push(f"{g}/", f"{REMOTE_ROOT}/generators/{g.rstrip('/').split('/')[-1]}/")
    pod.push(f"{pools_dir}/", f"{REMOTE_ROOT}/pools/snapshots/")

    pod.ssh(
        "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; "
        f"export PATH=/root/.local/bin:$PATH; "
        f"uv venv --python 3.11 {REMOTE_ROOT}/.venv >/dev/null 2>&1; "
        f"uv pip install --python {REMOTE_PY} '/root/cascade[train]' "
        "pandas scikit-learn gpytorch networkx >/dev/null 2>&1",
        timeout=2400,
    )
    assert_deps(pod)


def assert_deps(pod: Pod) -> None:
    """Refuse to proceed unless torch actually imports ON THE POD.

    `uv pip install --python <relative path>` silently installs nowhere; the
    failure is otherwise invisible until every training seed errors out.
    """
    r = pod.ssh(f"{REMOTE_PY} -c 'import torch, numpy; print(\"DEPS OK\")'", timeout=300)
    if "DEPS OK" not in r.stdout:
        raise RuntimeError(f"{pod.name}: deps missing — {r.stdout.strip()} {r.stderr.strip()[:200]}")


def check_ttl_covers(pod: Pod, needed_hours: float) -> None:
    """Refuse to start work the pod's TTL cannot cover.

    Two experiments were lost to pods expiring mid-batch; the second time the
    TTL had under an hour left when a 7.5h batch was launched.
    """
    from datetime import datetime, timezone
    if not pod.ttl_iso:
        return
    ttl_str = pod.ttl_iso.replace("Z", "+00:00")
    if "+" not in ttl_str and "T" in ttl_str:
        ttl_str = ttl_str + "+00:00"
    ttl = datetime.fromisoformat(ttl_str)
    if ttl.tzinfo is None:
        ttl = ttl.replace(tzinfo=timezone.utc)
    left = (ttl - datetime.now(timezone.utc)).total_seconds() / 3600
    if left < needed_hours:
        raise RuntimeError(
            f"{pod.name}: TTL leaves {left:.1f}h but the run needs {needed_hours:.1f}h"
        )


def start_puller(pods: list[Pod], local_scores: str, every_s: int = 600) -> subprocess.Popen:
    """Continuously pull results — start this WITH the run, never after.

    Pulling only on completion has cost two experiments. This copies whatever
    exists every cycle, so an expired pod costs at most one interval.
    """
    lines = ["#!/bin/bash", "while true; do"]
    for p in pods:
        opts = ' '.join(shlex.quote(o) for o in SSH_OPTS)
        lines.append(
            f'  timeout 600 ssh {opts} -p {p.port} root@{p.ip} '
            f"'mkdir -p {REMOTE_ROOT}/scores && tar -cf - -C {REMOTE_ROOT}/scores/ .' 2>/dev/null | "
            f'tar -xf - -C {shlex.quote(local_scores)}/ 2>/dev/null || true'
        )
    lines += [f'  sleep {every_s}', "done"]
    path = "/tmp/cascade_miner_puller.sh"
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    subprocess.run(["chmod", "+x", path], check=True)
    return subprocess.Popen([path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
