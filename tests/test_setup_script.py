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
    def run_setup(self, *args: str, home: str | None = None,
                  env_file: str | None = None) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.pop("CASCADE_DIR", None)
        env.pop("CASCADE_WALLET_NAME", None)
        # Point at a non-existent env file by default so a developer's real
        # .env cannot change what these offline tests observe.
        env["CASCADE_ENV_FILE"] = env_file or "/nonexistent/.env"
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
            # Operator actions remain (no cascade checkout, no .env), and the
            # exit status now says so.
            self.assertEqual(result.returncode, 1, result.stderr)
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
        self.assertEqual(result.returncode, 1, result.stderr)
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
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("ops/create-next-hotkey", result.stdout)
        # The shipped wrapper is a refusing stub, so this must be reported as
        # outstanding operator work rather than silently swallowed.
        self.assertIn("refused or failed", result.stdout)
        self.assertIn("registration burns TAO", result.stdout)

    def test_readme_documents_the_setup_entry_point(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("bash scripts/setup.sh --cascade-dir", readme)

    def test_check_mode_verifies_without_installing_or_downloading(self):
        with TemporaryDirectory() as home:
            result = self.run_setup(
                "--check", "--cascade-dir", f"{home}/cascade",
                "--venv", f"{home}/venv", home=home,
            )
            self.assertFalse(Path(home, "venv").exists())
            self.assertFalse(Path(home, ".ssh/id_ed25519").exists())
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("check only", result.stdout)
        # Nothing mutating may appear, not even as a plan.
        for fragment in ("would run:", "uv pip install", "ssh-keygen -t ed25519 -N",
                         "--sync-pool --root"):
            self.assertNotIn(fragment, result.stdout)

    def test_check_mode_never_runs_the_wallet_wrapper(self):
        with TemporaryDirectory() as home:
            result = self.run_setup(
                "--check", "--with-wallet",
                "--cascade-dir", f"{home}/cascade", home=home,
            )
        self.assertIn("skipping the create-hotkey wrapper", result.stdout)
        self.assertNotIn("refused or failed", result.stdout)

    def test_env_lint_flags_unquoted_spaces_and_placeholders(self):
        with TemporaryDirectory() as home:
            env_file = Path(home, ".env")
            env_file.write_text(
                "HF_TOKEN=hf_replace_me\n"
                'SAFE="a b"\n'
                "DANGEROUS=a b\n"
                "TRAILING=plain # comment\n"
                "# COMMENTED=x y z\n",
                encoding="utf-8",
            )
            result = self.run_setup(
                "--check", "--cascade-dir", f"{home}/cascade",
                home=home, env_file=str(env_file),
            )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("HF_TOKEN is still a placeholder", result.stdout)
        self.assertIn("unquoted value with a space in DANGEROUS", result.stdout)
        self.assertNotIn("SAFE", result.stdout)
        self.assertNotIn("TRAILING", result.stdout)
        self.assertNotIn("COMMENTED", result.stdout)

    def test_exit_status_is_zero_only_when_nothing_needs_the_operator(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("setup incomplete", source)
        # The registration reminder must be a note, not an action, or the
        # script could never exit 0.
        self.assertIn("note \"registration burns TAO", source)
        with TemporaryDirectory() as home:
            result = self.run_setup("--check", "--cascade-dir", f"{home}/cascade",
                                    home=home)
        self.assertEqual(result.returncode, 1)
        self.assertIn("operator action(s) remain", result.stdout)
