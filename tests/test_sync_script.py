"""Offline checks for scripts/sync.sh — no network, no venv, no clone."""

import subprocess
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sync.sh"


class SyncScriptTests(TestCase):
    def run_sync(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_script_is_executable_and_parses(self):
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)
        syntax = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_help_documents_check_mode(self):
        result = self.run_sync("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--check", result.stdout)
        self.assertIn("reinstall", result.stdout)

    def test_unknown_option_fails_loudly(self):
        result = self.run_sync("--frobnicate")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown option", result.stderr)

    def test_missing_reference_clone_has_actionable_error(self):
        result = self.run_sync(env={"CASCADE_DIR": "/nonexistent/cascade", "PATH": "/usr/bin:/bin"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("not a git checkout", result.stderr)
        self.assertIn("CASCADE_DIR", result.stderr)

    def test_reinstall_happens_and_notes_document_the_script(self):
        # A pull without a reinstall silently scores with the old metric; the
        # script and the notes must both carry that invariant.
        source = SCRIPT.read_text()
        self.assertIn("uv pip install", source)
        contract = (ROOT / "notes/CONTRACT.md").read_text()
        self.assertIn("scripts/sync.sh", contract)
        self.assertIn("Last synced:", contract)
