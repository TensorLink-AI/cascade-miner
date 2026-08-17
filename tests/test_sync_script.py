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

    def test_sync_regenerates_the_upstream_snapshot(self):
        # CI and the operator host must run the SAME extractor, or they end up
        # with different pictures of upstream.
        source = SCRIPT.read_text()
        self.assertIn("scripts/upstream_state.py", source)
        self.assertIn("--check", source)

    def test_tests_run_before_the_stamp_and_gate_it(self):
        # "Last synced" means PROSE REVIEWED at that revision, and the prose
        # pins live in the suite — so stamping a failing tree would forge the
        # one claim the stamp makes.
        source = SCRIPT.read_text()
        self.assertLess(
            source.index('step "tests"'), source.index('step "stamp notes/CONTRACT.md"'),
            "the suite must run before the stamp, not after",
        )
        stamp_section = source[source.index('step "stamp notes/CONTRACT.md"'):]
        self.assertIn('if [ "$TESTS_PASSED" = 0 ]', stamp_section)
        self.assertIn("NOT stamping", stamp_section)

    def test_reinstall_happens_and_notes_document_the_script(self):
        # A pull without a reinstall silently scores with the old metric; the
        # script and the notes must both carry that invariant.
        source = SCRIPT.read_text()
        self.assertIn("uv pip install", source)
        contract = (ROOT / "notes/CONTRACT.md").read_text()
        self.assertIn("scripts/sync.sh", contract)
        self.assertIn("Last synced:", contract)
