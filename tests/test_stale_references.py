from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from miner import pods


ROOT = Path(__file__).resolve().parents[1]


class StaleReferenceTests(TestCase):
    def test_remote_root_uses_current_project_name(self):
        self.assertEqual(pods.REMOTE_ROOT, "/root/cascade-miner")

    def test_removed_tool_names_are_not_documented(self):
        evaluate = (ROOT / "miner/evaluate.py").read_text()
        self.assertNotIn("tools/run_matrix.py", evaluate)
        self.assertNotIn("eval_all_snapshots.py", evaluate)

    def test_round_docs_do_not_claim_a_hardcoded_cadence_change(self):
        rounds = (ROOT / "miner/rounds.py").read_text()
        self.assertNotIn("7200 -> 3600", rounds)
        self.assertIn("chain.toml", rounds)

    def test_controller_does_not_resolve_the_venv_interpreter(self):
        source = (ROOT / "miner/controller.py").read_text()
        self.assertNotIn("cascade_python=(args.cascade_python", source)
        self.assertIn("cascade_python = absolute_path", source)

    def test_missing_lium_cli_has_actionable_error(self):
        with patch("miner.pods.shutil.which", return_value=None), \
                patch("miner.pods.subprocess.run") as run:
            with self.assertRaisesRegex(
                RuntimeError, "lium CLI not found.*README setup"
            ):
                pods._lium("ps", "--format", "json")
        run.assert_not_called()

    def test_lium_cli_is_resolved_from_path_at_call_time(self):
        with patch("miner.pods.shutil.which", return_value="/opt/lium/bin/lium"), \
                patch("miner.pods.subprocess.run") as run:
            pods._lium("ps", "--format", "json", timeout=123)
        run.assert_called_once_with(
            ["/opt/lium/bin/lium", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=123,
        )

    def test_readme_documents_explicit_lium_boundary(self):
        readme = (ROOT / "README.md").read_text()
        prose = " ".join(readme.split())
        self.assertIn("https://lium.io/install.sh", readme)
        self.assertIn("nothing imports or runs it automatically", prose)
        self.assertIn("controller deliberately never rents GPUs", prose)

    def test_hf_token_is_templated_and_exported_before_controller(self):
        template = (ROOT / "example.env").read_text()
        readme = (ROOT / "README.md").read_text()
        self.assertIn("HF_TOKEN=hf_replace_me", template)
        self.assertIn("set -a; source .env; set +a", readme)
