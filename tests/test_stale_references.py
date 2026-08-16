import re
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from miner import pods
from scripts import upstream_state


ROOT = Path(__file__).resolve().parents[1]


def _digits(text: str) -> str:
    """Normalise `3_700_000` / `3,700,000` to `3700000` so prose can group digits."""
    return re.sub(r"(?<=\d)[_,](?=\d)", "", text)


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

    def test_hippius_requires_only_the_hub_token(self):
        template = (ROOT / "example.env").read_text()
        readme = (ROOT / "README.md").read_text()
        self.assertIn("HIPPIUS_HUB_TOKEN=replace_me", template)
        self.assertNotIn("HIPPIUS_S3", template)
        self.assertNotIn("POOL_S3", template)
        self.assertIn("no Hippius S3 or pool-S3 credentials are required", readme)


class ContractMatchesUpstreamTests(TestCase):
    """Pin the prose to the generated snapshot of what cascade actually does.

    `notes/upstream-state.json` refreshes itself (scripts/upstream_state.py,
    run by the upstream-sync workflow); the notes do not. These assertions are
    what turns a silent divergence into a failing build that names the stale
    sentence. Each one exists because the claim it guards is load-bearing for
    how we design or score a generator — add one whenever a sync catches
    something the notes said wrongly.
    """

    @classmethod
    def setUpClass(cls):
        cls.state = upstream_state.load_snapshot()
        cls.keys = cls.state["keys"]
        cls.contract = _digits((ROOT / "notes/CONTRACT.md").read_text())
        cls.readme = _digits((ROOT / "README.md").read_text())
        cls.claude = _digits((ROOT / "CLAUDE.md").read_text())

    # assertIn/assertNotIn would print the whole document on failure; the sync
    # workflow pastes this output into a PR body, so failures say what is wrong
    # and nothing else.
    def assert_states(self, claim: str, doc: str, doc_name: str, why: str):
        self.assertTrue(claim in doc, f"{doc_name} does not state {why} ({claim!r})")

    def assert_omits(self, claim: str, doc: str, doc_name: str, why: str):
        self.assertFalse(claim in doc, f"{doc_name} still says {claim!r} — {why}")

    def assert_key_stated(self, key: str, doc: str | None = None, doc_name: str = "notes/CONTRACT.md"):
        """`name = value`, exactly as upstream has it, must appear in the prose."""
        name = key.split(".", 1)[1]
        self.assert_states(
            f"{name} = {self.keys[key]}",
            self.contract if doc is None else doc,
            doc_name,
            f"the live {name}",
        )

    def test_warm_start_regime_is_described_correctly(self):
        # The one that bit us: warm-start armed 2026-08-05 and the notes still
        # said the trainer starts from random init ten days later. That is not
        # a wording nit — "the model starts from noise, so it learns only from
        # us" is the premise the whole corpus design rests on.
        if not self.keys["scoring.cascade_enabled"]:
            self.skipTest("warm-start disarmed upstream; the from-scratch framing is correct again")
        stale_claims = (
            "backbone\nfrom random init",
            "from random initialisation on your corpus",
            "trains a fixed forecaster from scratch on",
            "The model starts from **noise**",
        )
        for doc_name, doc in (
            ("notes/CONTRACT.md", self.contract),
            ("README.md", self.readme),
            ("CLAUDE.md", self.claude),
        ):
            for claim in stale_claims:
                self.assert_omits(
                    claim, doc, doc_name,
                    "scoring.cascade_enabled is true upstream, so the run is warm-started",
                )
            self.assert_states("warm-start", doc.lower(), doc_name, "the live warm-start regime")

    def test_warm_start_envelope_values_are_current(self):
        if not self.keys["scoring.cascade_enabled"]:
            self.skipTest("warm-start disarmed upstream")
        for key in (
            "scoring.cascade_reign_rounds",
            "scoring.cascade_top_k",
            "scoring.cascade_quality_epsilon",
        ):
            self.assert_key_stated(key)

    def test_round_mechanics_values_are_current(self):
        for key in (
            "round.finalists",
            "round.heat_n_windows",
            "round.heat_train_hours",
            "round.commit_floor_block",
            "round.screen_size",
            "training.target_train_hours",
        ):
            self.assert_key_stated(key)

    def test_netuid_is_current_everywhere_it_is_claimed(self):
        netuid = str(self.keys["subnet.netuid"])
        for doc_name, doc in (
            ("notes/CONTRACT.md", self.contract),
            ("README.md", self.readme),
            ("CLAUDE.md", self.claude),
        ):
            claims = [line for line in doc.splitlines() if "netuid" in line.lower()]
            self.assertTrue(claims, f"{doc_name} never states the netuid")
            self.assertTrue(
                any(netuid in line for line in claims),
                f"{doc_name} states a netuid that is not {netuid}",
            )

    def test_dedup_enforcement_level_is_current(self):
        self.assert_states(
            f"Dedup is `{self.keys['round.dedup_mode']}`", self.contract,
            "notes/CONTRACT.md", "the live dedup enforcement level",
        )

    def test_throughput_reference_is_current(self):
        # The token budget is derived from this number (DEC-CA-0001); quoting a
        # stale one silently mis-sizes every generate-path calculation we do.
        self.assert_key_stated("training.ref_throughput_tokens_per_s")

    def test_generated_notes_are_marked_as_generated(self):
        rendered = (ROOT / "notes/UPSTREAM.md").read_text()
        self.assertIn("GENERATED by scripts/upstream_state.py", rendered)
        self.assert_states(
            "notes/UPSTREAM.md", (ROOT / "notes/CONTRACT.md").read_text(),
            "notes/CONTRACT.md", "where the generated values live",
        )
