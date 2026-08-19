"""Offline regression checks for the evaluation-control documentation."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MethodDocumentationTests(unittest.TestCase):
    def test_method_preserves_all_verdict_inputs_and_pairing(self):
        method = (ROOT / "notes" / "METHOD.md").read_text(encoding="utf-8")
        for term in ("RoundSeeds", "qloss", "abs_target", "mase", "source",
                     "series_id", "cluster bootstrap", "lower confidence bound"):
            self.assertIn(term, method)
        self.assertIn("fresh control", method.lower())
        self.assertIn("metric versions", method.lower())
        self.assertIn("diagnose by source", method.lower())

    def test_readme_fetches_receipt_king_and_scores_paired_arms(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn('["last_receipt"]["summary"]["king_gen_ref"]', readme)
        self.assertIn('cascade fetch "$KING_REF" --out generators/king-control', readme)
        self.assertIn("miner.evaluate ./generators/king-control", readme)
        self.assertIn("miner.evaluate ./generators/<name>", readme)
        self.assertIn("--king king-control", readme)
        self.assertNotIn("--king <king_dir>", readme)

    def test_method_states_what_screening_can_and_cannot_do(self):
        method = (ROOT / "notes" / "METHOD.md").read_text(encoding="utf-8").lower()
        self.assertIn("seed-to-seed", method)
        self.assertIn("falsifier, never a predictor", method)
        self.assertIn("coverage traded away", method)

    def test_agent_entry_points_route_to_the_screen_before_paid_work(self):
        for name in ("AGENTS.md", "CLAUDE.md", "llms.txt",
                     "skills/cascade-miner/SKILL.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("miner.screen", text, f"{name} does not mention the screen")
        brief = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("does not predict the duel", brief)

    def test_llms_index_points_at_files_that_exist(self):
        import re

        text = (ROOT / "llms.txt").read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            self.assertTrue((ROOT / target).exists(), f"llms.txt links missing {target}")

    def test_analyze_docstring_uses_real_module(self):
        source = (ROOT / "miner" / "analyze.py").read_text(encoding="utf-8")
        self.assertIn("python -m miner.analyze", source)
        self.assertNotIn("tools/analyze_matrix.py", source)


if __name__ == "__main__":
    unittest.main()
