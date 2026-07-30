"""Offline checks for scripts/setup.sh — no network, no venv, no wallet."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/setup.sh"


class SetupScriptTests(TestCase):
    def run_setup(self, *args: str, home: str | None = None) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.pop("CASCADE_DIR", None)
        env.pop("CASCADE_WALLET_NAME", None)
        if home is not None:
            env["HOME"] = home
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True, text=True, cwd=ROOT, env=env, timeout=120,
        )

    def test_script_is_executable_and_parses(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        syntax = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_help_lists_every_documented_step(self):
        result = self.run_setup("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for fragment in ("venv", "Lium CLI", "SSH key", "wallet status",
                         "eval-pool snapshot", "controller state", "cascade verify",
                         "--cascade-dir", "--eval-pool-snapshot", "--dry-run"):
            self.assertIn(fragment, result.stdout)

    def test_unknown_option_fails_loudly(self):
        result = self.run_setup("--nope")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown option", result.stderr)

    def test_dry_run_plans_every_step_and_changes_nothing(self):
        with TemporaryDirectory() as home:
            result = self.run_setup(
                "--dry-run", "--cascade-dir", f"{home}/cascade",
                "--venv", f"{home}/venv", home=home,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(Path(home, "venv").exists())
            self.assertFalse(Path(home, ".ssh/id_ed25519").exists())
        for fragment in ("uv venv --python 3.11", "uv pip install",
                         "miner.controller --sync-pool", "miner.controller --once",
                         "cascade verify", "ssh-keygen -t ed25519",
                         "would check", "needs action", "=== summary"):
            self.assertIn(fragment, result.stdout)
        # A plan must never be reported as a verified outcome.
        self.assertIn("(unverified, dry run)", result.stdout)

    def test_selected_snapshot_reaches_both_controller_calls(self):
        with TemporaryDirectory() as home:
            result = self.run_setup(
                "--dry-run", "--eval-pool-snapshot", "2026-07-16",
                "--cascade-dir", f"{home}/cascade", home=home,
            )
        self.assertIn("--sync-pool --root", " ".join(result.stdout.split()))
        self.assertEqual(result.stdout.count("--eval-pool-snapshot 2026-07-16"), 2)

    def test_skips_are_honoured(self):
        with TemporaryDirectory() as home:
            result = self.run_setup(
                "--dry-run", "--cascade-dir", f"{home}/cascade",
                "--skip-venv", "--skip-lium", "--skip-ssh-key", "--skip-pool",
                "--skip-seed", "--skip-verify", home=home,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("skipped"), 6)
        self.assertNotIn("would run:", result.stdout)
        for fragment in ("uv pip install", "ssh-keygen", "--sync-pool",
                         "cascade verify"):
            self.assertNotIn(fragment, result.stdout)

    def test_wallet_creation_is_delegated_and_never_handles_secrets(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("new-coldkey", "new-hotkey", "regen-coldkey",
                          "bittensor.Wallet", "create_new_coldkey",
                          "coldkey.json", "--password"):
            self.assertNotIn(forbidden, source)
        self.assertIn("CASCADE_CREATE_HOTKEY_COMMAND", source)
        with TemporaryDirectory() as home:
            result = self.run_setup(
                "--with-wallet", "--skip-venv", "--skip-lium", "--skip-ssh-key",
                "--skip-pool", "--skip-seed", "--skip-verify",
                "--cascade-dir", f"{home}/cascade", home=home,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/create-next-hotkey", result.stdout)
        # The shipped wrapper is a refusing stub, so this must be reported as
        # outstanding operator work rather than silently swallowed.
        self.assertIn("refused or failed", result.stdout)
        self.assertIn("registration burns TAO", result.stdout)

    def test_extras_are_configurable_for_disk_constrained_hosts(self):
        with TemporaryDirectory() as home:
            result = self.run_setup(
                "--dry-run", "--extras", "hippius,chain",
                "--cascade-dir", f"{home}/cascade", home=home,
            )
        self.assertIn("[hippius,chain]", result.stdout)
        self.assertNotIn("train,hippius,chain", result.stdout)

    def test_readme_documents_the_setup_entry_point(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("bash scripts/setup.sh --cascade-dir", readme)
        self.assertIn("docs/SETUP.md", readme)


class SetupGuideTests(TestCase):
    """docs/SETUP.md must drive the shipped tooling, not re-implement it."""

    GUIDE = (ROOT / "docs/SETUP.md").read_text(encoding="utf-8")

    def test_guide_uses_the_controller_for_pool_sync_and_seeding(self):
        self.assertIn("miner.controller --sync-pool", self.GUIDE)
        self.assertIn("miner.controller --once", self.GUIDE)
        self.assertIn("--eval-pool-snapshot", self.GUIDE)
        self.assertIn("bash scripts/setup.sh --cascade-dir", self.GUIDE)

    def test_guide_does_not_hand_write_controller_state(self):
        """A hand-seeded state file fires hooks with no king_gen_ref to fetch."""
        for stale in ("'initialized': True", '"initialized": true',
                      "'last_receipt': {}", "snapshot_download("):
            self.assertNotIn(stale, self.GUIDE)

    def test_guide_never_recommends_overwriting_an_existing_coldkey(self):
        """`overwrite=True` destroys an existing coldkey and its funds."""
        self.assertNotIn("use_password=False, overwrite=True", self.GUIDE)
        self.assertIn("use_password=False, overwrite=False", self.GUIDE)
