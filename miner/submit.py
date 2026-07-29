"""Deploy a generator: prepare -> verify -> upload -> commit -> confirm reveal.

Two non-obvious things are encoded here, both learned the hard way:

1. ``cascade deploy`` CANNOT upload. ``cascade.shared.hippius.upload_dir_to_hub``
   passes an explicit ``token=`` that the registry rejects for blob upload with
   ``401 (blob upload init failed)``. Calling ``hippius_hub.upload_folder`` with
   ``token=None`` and letting the SDK resolve auth works. So we upload
   ourselves and hand ``cascade deploy`` a ``--ref``.

Submission is deliberately NOT autonomous. ``one_submission_per_hotkey`` means a
wrong call burns a hotkey plus the registration cost, and the local eval has a
~3.3pp noise floor with a heavy tail (a byte-identical rerun once differed by
5.97%). An automated picker would confidently deploy on noise. Callers must pass
``confirm=True``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import secrets
import subprocess
import sys
from pathlib import Path


def corpus_digest(repo: Path, n: int = 256, seed: int = 0) -> str:
    """sha256 over the first `n` emitted series — the behaviour fingerprint."""
    repo = Path(repo)
    spec = importlib.util.spec_from_file_location(f"g_{repo.name}", repo / "generator.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(repo))
    h = hashlib.sha256()
    for s in mod.Generator(str(repo), seed=seed).generate(n):
        h.update(s.tobytes())
    return h.hexdigest()


def prepare(src: Path, dst: Path) -> str:
    """Copy deployable generator files into `dst` and return the corpus digest."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        subprocess.run(["rm", "-rf", str(dst)], check=True)
    dst.mkdir(parents=True)
    for f in ("generator.py", "config.json", "requirements.txt"):
        subprocess.run(["cp", str(src / f), str(dst / f)], check=True)
    subprocess.run(["rm", "-rf", str(dst / "__pycache__")], check=False)
    return corpus_digest(dst)


def verify(repo: Path, chain_toml: Path, cascade_bin: str) -> None:
    r = subprocess.run([cascade_bin, "verify", str(repo), "--chain-toml", str(chain_toml)],
                       capture_output=True, text=True, timeout=3600)
    if "would be accepted by the trainer" not in r.stdout:
        raise RuntimeError(f"cascade verify FAILED:\n{r.stdout[-800:]}\n{r.stderr[-400:]}")


def upload(repo: Path, namespace: str) -> str:
    """Upload to a FRESH random repo and return `ns/name@sha256:...`.

    A predictable repo name lets competitors watch the namespace and copy the
    generator before the on-chain pointer reveals. Random names keep content as
    undiscoverable as the pointer (the registry catalog is not enumerable).
    """
    from hippius_hub import upload_folder
    name = f"{namespace}/gen-{secrets.token_hex(6)}"
    res = upload_folder(
        repo_id=name, folder_path=str(repo),
        allow_patterns=["generator.py", "config.json", "requirements.txt"],
        token=None,                      # MUST be None — see module docstring
    )
    oid = getattr(res, "oid", "")
    if not oid.startswith("sha256:"):
        raise RuntimeError(f"upload returned no digest: {res!r}")
    return f"{name}@{oid}"


def check_pullable(ref: str, chain_toml: Path, cascade_bin: str, out: Path) -> None:
    """Fetch the ref with NO credentials — exactly as the trainer does.

    A private project returns 401 and the submission is rejected every round as
    `generator_artifact_unreachable`, while the on-chain commit looks fine.
    """
    import os
    env = {k: v for k, v in os.environ.items()
           if k not in ("HIPPIUS_HUB_TOKEN", "HIPPIUS_HUB_USERNAME", "HIPPIUS_HUB_PASSWORD")}
    subprocess.run(["rm", "-rf", str(out)], check=False)
    r = subprocess.run([cascade_bin, "fetch", ref, "--chain-toml", str(chain_toml),
                        "--out", str(out)], capture_output=True, text=True, env=env, timeout=900)
    if not (out / "generator.py").exists():
        raise RuntimeError(f"ref NOT anonymously pullable:\n{r.stdout[-400:]}{r.stderr[-400:]}")


def commit(ref: str, repo: Path, chain_toml: Path, cascade_bin: str,
           wallet: str, hotkey: str, network: str = "finney",
           confirm: bool = False) -> str:
    """Write the on-chain pointer. Requires an explicit `confirm=True`."""
    if not confirm:
        raise RuntimeError("refusing to submit without confirm=True — this burns the hotkey")
    r = subprocess.run(
        [cascade_bin, "deploy", str(repo), "--chain-toml", str(chain_toml),
         "--network", network, "--wallet-name", wallet, "--wallet-hotkey", hotkey,
         "--ref", ref, "--skip-verify"],
        capture_output=True, text=True, timeout=1800,
    )
    if "committed:" not in r.stdout:
        raise RuntimeError(f"commit failed:\n{r.stdout[-800:]}\n{r.stderr[-400:]}")
    return r.stdout


def reveal_status(hotkey_ss58: str, boundary: int, chain_toml: Path,
                  cascade_bin: str, network: str = "finney") -> str:
    """A missed reveal auto-rolls to the next round and does NOT burn the
    submission — the burn persists only once a heat has screened the field."""
    r = subprocess.run(
        [cascade_bin, "reveal-status", hotkey_ss58, "--network", network,
         "--expect-boundary", str(boundary), "--chain-toml", str(chain_toml)],
        capture_output=True, text=True, timeout=600,
    )
    return r.stdout.strip()
