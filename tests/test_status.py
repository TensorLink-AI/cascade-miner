import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from miner import experiments, issues, status


def seed_runs(root: Path) -> None:
    runs = root / "runs"
    runs.mkdir(parents=True)
    (runs / "controller-state.json").write_text(json.dumps({
        "initialized": True,
        "updated_at": "2026-08-01T10:00:00+00:00",
        "last_round_id": "round-42",
        "cascade_head": "abc123",
        "eval_pool_repo": "Tensor-Link/cascade-eval-pool",
        "eval_pool_revision": "rev-1234567890ab",
        "eval_pool_snapshot": "2026-07-16",
        "eval_pool_local_snapshots": ["2026-07-16"],
        "last_receipt": {
            "status": "final",
            "summary": {"king_gen_ref": "hippius://king/ref"},
            "miner_heat": {"rank": 3},
        },
    }), encoding="utf-8")
    (runs / "approvals.json").write_text(json.dumps({"requests": [
        {"id": "aaa111bbb222", "action": "gpu_evaluation",
         "reason": "screen arm A", "status": "pending"},
        {"id": "ccc333ddd444", "action": "gpu_evaluation",
         "reason": "done earlier", "status": "completed"},
    ]}), encoding="utf-8")
    (runs / "controller-events.jsonl").write_text(
        "\n".join(json.dumps({"type": "round_result", "round_id": f"r{i}"})
                  for i in range(15)) + "\nnot json\n",
        encoding="utf-8",
    )


class StatusTests(TestCase):
    def test_summary_reads_all_runs_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seed_runs(root)
            experiments.log_entry(root, hypothesis="ledger shows up")
            summary = status.summarize(root)
        self.assertEqual(summary["round"]["id"], "round-42")
        self.assertEqual(summary["round"]["king_gen_ref"], "hippius://king/ref")
        self.assertEqual(summary["round"]["miner_heat"], {"rank": 3})
        self.assertEqual(summary["eval_pool"]["snapshot"], "2026-07-16")
        self.assertEqual(
            [r["id"] for r in summary["approvals"]["pending"]],
            ["aaa111bbb222"],
        )
        self.assertEqual(summary["approvals"]["counts"]["completed"], 1)
        # The corrupt trailing line occupies one tail slot and is skipped.
        self.assertEqual(len(summary["recent_events"]), status.EVENT_TAIL - 1)
        self.assertEqual(summary["recent_events"][-1]["round_id"], "r14")
        self.assertEqual(summary["experiments"][0]["hypothesis"],
                         "ledger shows up")

    def test_summary_renders_before_the_controller_ever_ran(self):
        with TemporaryDirectory() as directory:
            summary = status.summarize(Path(directory))
            text = status.render(summary)
        self.assertFalse(summary["initialized"])
        self.assertIn("(none)", text)
        self.assertIn("none pending", text)

    def test_open_issues_are_summarized_and_rendered(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seed_runs(root)
            triaged = issues.report_entry(root, kind="bug", title="already fixed")
            issues.report_entry(root, kind="bug", title="already fixed",
                                status="resolved", supersedes=triaged["id"])
            live = issues.report_entry(root, kind="feature",
                                       title="agents want a pool-sync tool")
            summary = status.summarize(root)
            text = status.render(summary)
        self.assertEqual([e["id"] for e in summary["open_issues"]], [live["id"]])
        self.assertIn("1 open", text)
        self.assertIn("agents want a pool-sync tool", text)
        self.assertNotIn("already fixed", text)

    def test_render_shows_pending_approvals_with_the_approve_command(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seed_runs(root)
            text = status.render(status.summarize(root))
        self.assertIn("aaa111bbb222", text)
        self.assertIn("--approve <id>", text)
        self.assertIn("hippius://king/ref", text)


class EnvFileTests(TestCase):
    def parse(self, text: str):
        with TemporaryDirectory() as directory:
            path = Path(directory, ".env")
            path.write_text(text, encoding="utf-8")
            return status.parse_env_file(path)

    def test_quoted_values_and_comments_are_safe(self):
        values, problems = self.parse(
            'SAFE="a b"\n'
            "TRAILING=plain # comment\n"
            "# COMMENTED=x y z\n"
            "export EXPORTED=fine\n"
        )
        self.assertEqual(problems, [])
        self.assertEqual(values["SAFE"], "a b")
        self.assertEqual(values["TRAILING"], "plain")
        self.assertEqual(values["EXPORTED"], "fine")
        self.assertNotIn("COMMENTED", values)

    def test_unquoted_space_is_reported_as_the_source_footgun(self):
        values, problems = self.parse("DANGEROUS=a b\nOK=fine\n")
        self.assertEqual(len(problems), 1)
        self.assertIn("DANGEROUS", problems[0])
        self.assertIn("source .env", problems[0])
        self.assertNotIn("DANGEROUS", values)
        self.assertEqual(values["OK"], "fine")

    def test_missing_file_is_empty_not_an_error(self):
        values, problems = status.parse_env_file(Path("/nonexistent/.env"))
        self.assertEqual((values, problems), ({}, []))


def seed_ready_host(root: Path, home: Path) -> dict[str, str]:
    """Build a filesystem where every doctor check passes, returning the env."""
    candidate = root / "generators/candidate"
    candidate.mkdir(parents=True)
    for name in status.CANDIDATE_FILES:
        (candidate / name).write_text("# stub\n", encoding="utf-8")
    snapshot = root / "pools/snapshots/2026-07-16"
    snapshot.mkdir(parents=True)
    (snapshot / "data.json").write_text("{}", encoding="utf-8")
    runs = root / "runs"
    runs.mkdir()
    (runs / "controller-state.json").write_text(
        json.dumps({"initialized": True}), encoding="utf-8")
    (root / ".env").write_text(
        "HF_TOKEN=hf_real\nHIPPIUS_HUB_TOKEN=tok\nLIUM_API_KEY=key\n"
        "CASCADE_MINER_HOTKEY=5Freal\n", encoding="utf-8")
    cascade_dir = home / "cascade"
    cascade_dir.mkdir(parents=True)
    (cascade_dir / "chain.toml").write_text("netuid = 91\n", encoding="utf-8")
    lium = home / ".lium/bin/lium"
    lium.parent.mkdir(parents=True)
    lium.write_text("#!/bin/sh\n", encoding="utf-8")
    ssh_key = home / ".ssh/id_ed25519"
    ssh_key.parent.mkdir()
    ssh_key.write_text("key\n", encoding="utf-8")
    wallet = home / "wallets/test-wallet"
    (wallet / "hotkeys").mkdir(parents=True)
    (wallet / "coldkeypub.txt").write_text("{}", encoding="utf-8")
    (wallet / "hotkeys/default").write_text("{}", encoding="utf-8")
    return {
        "HOME": str(home),
        "CASCADE_DIR": str(cascade_dir),
        "LIUM_SSH_KEY": str(ssh_key),
        "CASCADE_WALLET_NAME": "test-wallet",
        "BT_WALLET_PATH": str(home / "wallets"),
    }


class DoctorTests(TestCase):
    def test_fresh_host_fails_with_a_fix_for_every_stage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory, "repo")
            root.mkdir()
            # CASCADE_DIR must point inside the sandbox: the default is
            # /root/cascade, whose mere existence (or an unreadable /root)
            # on the host would leak into the result.
            env = {"HOME": str(Path(directory, "home")),
                   "CASCADE_DIR": str(Path(directory, "nowhere/cascade"))}
            with patch.dict(os.environ, env, clear=True):
                report = status.doctor(root)
        self.assertFalse(report["ready"])
        by_name = {c["name"]: c for c in report["checks"]}
        for name in ("credentials", "cascade checkout", "lium CLI", "SSH key",
                     "eval pool", "controller state", "candidate", "wallet"):
            self.assertFalse(by_name[name]["ok"], name)
            self.assertTrue(by_name[name]["fix"], name)
        # `next` is the first failing required fix, so the operator always
        # has exactly one command to run.
        first_failing = next(c for c in report["checks"]
                             if not c["ok"] and c["required"])
        self.assertEqual(report["next"], first_failing["fix"])

    def test_ready_host_passes_and_exits_zero(self):
        with TemporaryDirectory() as directory:
            root = Path(directory, "repo")
            home = Path(directory, "home")
            root.mkdir()
            env = seed_ready_host(root, home)
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(status, "_module_available", return_value=True):
                report = status.doctor(root)
                rendered = status.render_doctor(report)
                exit_code = status.main(["--doctor", "--root", str(root)])
        self.assertTrue(report["ready"], json.dumps(report, indent=2))
        self.assertIsNone(report["next"])
        self.assertEqual(exit_code, 0)
        self.assertIn("ready", rendered)
        # Registration cannot be checked offline; the reminder must survive.
        self.assertIn("one-shot per hotkey", rendered)

    def test_a_moved_reference_clone_fails_the_upstream_sync_check(self):
        # The operator pulls the clone but forgets scripts/sync.sh: the venv
        # still scores with the old metric and nothing says so. The doctor is
        # what agents run to orient, so the drift has to surface there.
        with TemporaryDirectory() as directory:
            root = Path(directory, "repo")
            home = Path(directory, "home")
            root.mkdir()
            env = seed_ready_host(root, home)
            cascade_dir = Path(env["CASCADE_DIR"])
            (cascade_dir / "chain.toml").write_text(
                "[subnet]\nnetuid = 91\n\n[round]\nfinalists = 1\n", encoding="utf-8")
            notes = root / "notes"
            notes.mkdir()
            (notes / "upstream-state.json").write_text(
                json.dumps({"keys": {"subnet.netuid": 91, "round.finalists": 3},
                            "untracked_keys": []}), encoding="utf-8")
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(status, "_module_available", return_value=True):
                report = status.doctor(root)
        check = {c["name"]: c for c in report["checks"]}["upstream sync"]
        self.assertFalse(check["ok"])
        self.assertIn("round.finalists", check["detail"])
        self.assertIn("scripts/sync.sh", check["fix"])
        self.assertFalse(report["ready"])

    def test_upstream_sync_check_reports_drift_it_cannot_invent(self):
        # No snapshot to compare against (or an unreadable clone) must read as
        # "nothing to say", never as a phantom failure that blocks a host.
        with TemporaryDirectory() as directory:
            root = Path(directory, "repo")
            root.mkdir()
            self.assertEqual(
                status.upstream_drift(root, root / "notes/upstream-state.json"), "")

    def test_pending_approvals_warn_but_do_not_block_readiness(self):
        with TemporaryDirectory() as directory:
            root = Path(directory, "repo")
            home = Path(directory, "home")
            root.mkdir()
            env = seed_ready_host(root, home)
            (root / "runs/approvals.json").write_text(json.dumps({"requests": [
                {"id": "req1", "action": "gpu_evaluation", "status": "pending"},
            ]}), encoding="utf-8")
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(status, "_module_available", return_value=True):
                report = status.doctor(root)
        self.assertTrue(report["ready"])
        approvals = next(c for c in report["checks"] if c["name"] == "approvals")
        self.assertFalse(approvals["ok"])
        self.assertFalse(approvals["required"])
        self.assertIn("req1", approvals["detail"])

    def test_env_quoting_problem_fails_the_credentials_check(self):
        with TemporaryDirectory() as directory:
            root = Path(directory, "repo")
            home = Path(directory, "home")
            root.mkdir()
            env = seed_ready_host(root, home)
            (root / ".env").write_text(
                "HF_TOKEN=hf_real\nBAD=a b\n", encoding="utf-8")
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(status, "_module_available", return_value=True):
                report = status.doctor(root)
        credentials = next(c for c in report["checks"]
                           if c["name"] == "credentials")
        self.assertFalse(credentials["ok"])
        self.assertIn("BAD", credentials["detail"])

    def test_unreadable_path_reads_as_absent_not_a_crash(self):
        # On a non-root host /root/cascade can raise EACCES on stat, which
        # pathlib does not swallow. The doctor must report the stage as not
        # ready instead of crashing (seen on the CI runner).
        class Denied:
            def is_file(self):
                raise PermissionError(13, "Permission denied")

        self.assertFalse(status._is_file(Denied()))

    def test_doctor_json_is_machine_readable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory, "repo")
            root.mkdir()
            env = {"HOME": str(Path(directory, "home")),
                   "CASCADE_DIR": str(Path(directory, "nowhere/cascade"))}
            with patch.dict(os.environ, env, clear=True):
                report = status.doctor(root)
        parsed = json.loads(json.dumps(report))
        self.assertEqual({c["name"] for c in parsed["checks"]},
                         {c["name"] for c in report["checks"]})
