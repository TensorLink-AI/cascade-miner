import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from miner import controller
from miner.controller import (
    CascadeDirty,
    Controller,
    absolute_path,
    assert_runtime,
    queue_approval,
    read_json,
    refresh_cascade,
    set_approval_status,
    sync_eval_pool,
)
from scripts.improve_candidate import build_prompt


def make_controller(root: Path, **overrides) -> Controller:
    values = {
        "root": root,
        "cascade_dir": root / "cascade",
        "chain_toml": root / "cascade/chain.toml",
        "cascade_python": Path(sys.executable),
        "state_file": root / "runs/state.json",
        "events_file": root / "runs/events.jsonl",
        "approvals_file": root / "runs/approvals.json",
        "eval_pool_dir": root / "pools",
    }
    values.update(overrides)
    return Controller(**values)


class RuntimeTests(TestCase):
    def test_absolute_interpreter_path_does_not_resolve_venv_symlink(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "base-python"
            target.write_text("python")
            link = root / ".venv/bin/python"
            link.parent.mkdir(parents=True)
            link.symlink_to(target)

            self.assertEqual(absolute_path(link), link.absolute())
            self.assertNotEqual(absolute_path(link), link.resolve())

    def test_runtime_check_names_failing_interpreter(self):
        with self.assertRaisesRegex(RuntimeError, sys.executable):
            assert_runtime(
                Path(sys.executable), Path.cwd(),
                modules=("module_that_does_not_exist_cascade_miner",),
            )

    def test_runtime_check_names_missing_interpreter(self):
        missing = Path("/definitely/missing/cascade-python")
        with self.assertRaisesRegex(RuntimeError, str(missing)):
            assert_runtime(missing, Path.cwd())

    def test_parser_defaults_to_repository_root(self):
        args = controller.build_parser().parse_args([])
        self.assertEqual(args.root, controller.REPO_ROOT)


class PoolSyncTests(TestCase):
    def _fake_hub(self, root: Path, *, noisy: bool = False) -> Path:
        package = root / "huggingface_hub"
        package.mkdir()
        noise = "sys.stderr.write('x' * 600000 + 'REAL_ERROR_TAIL\\n')" if noisy else ""
        package.joinpath("__init__.py").write_text(
            "import json, os, pathlib, sys\n"
            "class Info:\n    sha = 'revision-2'\n"
            "class HfApi:\n"
            "    def __init__(self, token=None):\n"
            "        if token != 'hf_test': raise RuntimeError('missing token')\n"
            "    def dataset_info(self, repo_id):\n"
            "        assert os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS') == '1'\n"
            f"        {noise}\n"
            + ("        raise RuntimeError('boom')\n" if noisy else "        return Info()\n")
            + "def snapshot_download(**kwargs):\n"
              "    pathlib.Path(kwargs['local_dir']).mkdir(parents=True, exist_ok=True)\n"
              "    pathlib.Path(kwargs['local_dir'], 'downloaded').write_text(kwargs['revision'])\n"
        )
        return root

    def test_unchanged_revision_skips_snapshot_download_and_uses_token(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fake = self._fake_hub(root)
            dest = root / "pool"
            with patch.dict(os.environ, {
                "PYTHONPATH": str(fake), "HF_TOKEN": "hf_test",
            }, clear=False):
                result = sync_eval_pool(
                    Path(sys.executable), root, "owner/pool", dest, "revision-2"
                )
            self.assertFalse(result["downloaded"])
            self.assertFalse((dest / "downloaded").exists())

    def test_changed_revision_downloads_once(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fake = self._fake_hub(root)
            dest = root / "pool"
            with patch.dict(os.environ, {
                "PYTHONPATH": str(fake), "HF_TOKEN": "hf_test",
            }, clear=False):
                result = sync_eval_pool(
                    Path(sys.executable), root, "owner/pool", dest, "revision-1"
                )
            self.assertTrue(result["downloaded"])
            self.assertEqual((dest / "downloaded").read_text(), "revision-2")

    def test_pool_error_is_quiet_and_truncated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fake = self._fake_hub(root, noisy=True)
            with patch.dict(os.environ, {
                "PYTHONPATH": str(fake), "HF_TOKEN": "hf_test",
            }, clear=False):
                with self.assertRaises(RuntimeError) as caught:
                    sync_eval_pool(
                        Path(sys.executable), root, "owner/pool", root / "pool", ""
                    )
            message = str(caught.exception)
            self.assertIn("REAL_ERROR_TAIL", message)
            self.assertLess(len(message), 2100)


class ApprovalTests(TestCase):
    def test_pending_request_is_deduplicated_and_approvable(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "approvals.json"
            first = queue_approval(
                path, action="gpu_evaluation", reason="candidate ready",
                context={"round_id": "1", "candidate_digest": "abc"},
            )
            second = queue_approval(
                path, action="gpu_evaluation", reason="candidate ready",
                context={"round_id": "1", "candidate_digest": "abc"},
            )
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(len(read_json(path)["requests"]), 1)
            self.assertEqual(
                set_approval_status(path, first["id"], "approved")["status"],
                "approved",
            )

    def test_agent_prompt_describes_both_modes(self):
        root = Path("/tmp/project")
        self.assertIn("Controller mode: human", build_prompt(root, "{}", "human"))
        self.assertIn(
            "Controller mode: autonomous", build_prompt(root, "{}", "autonomous")
        )

    def test_privileged_actions_map_only_to_configured_commands(self):
        instance = make_controller(
            Path("/tmp/project"),
            create_hotkey_command="./ops/create-hotkey",
            register_hotkey_command="./ops/register-hotkey",
            submit_command="./ops/submit",
        )
        self.assertEqual(instance.action_command("create_hotkey"), "./ops/create-hotkey")
        self.assertEqual(instance.action_command("register_hotkey"), "./ops/register-hotkey")
        self.assertEqual(instance.action_command("submit_candidate"), "./ops/submit")
        self.assertEqual(instance.action_command("arbitrary_shell"), "")


class CycleTests(TestCase):
    def _cycle_patches(self, *, round_id: str = "round-1"):
        return (
            patch.object(controller, "refresh_cascade",
                         return_value=("head-1", "head-1")),
            patch.object(controller, "chain_snapshot",
                         return_value={"digest": "chain-1", "values": {}}),
            patch.object(controller, "sync_eval_pool", return_value={
                "repo_id": "owner/pool", "revision": "pool-1", "downloaded": False,
            }),
            patch.object(controller, "audit_latest", return_value={
                "round_id": round_id, "status": "scored", "ok": True,
            }),
        )

    def test_first_cycle_seeds_state_without_hook(self):
        with TemporaryDirectory() as directory:
            instance = make_controller(Path(directory), improve_command="agent")
            patches = self._cycle_patches()
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(controller, "run_hook") as hook:
                events = instance.cycle()
            hook.assert_not_called()
            state = read_json(instance.state_file)
            self.assertTrue(state["initialized"])
            self.assertEqual(state["last_round_id"], "round-1")
            self.assertEqual(len(events), 3)

    def test_state_and_event_are_persisted_before_hook_crash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            instance = make_controller(root, improve_command="agent")
            instance.state_file.parent.mkdir(parents=True)
            instance.state_file.write_text(json.dumps({
                "initialized": True,
                "cascade_head": "head-1",
                "chain": {"digest": "chain-1", "values": {}},
                "eval_pool_revision": "pool-1",
                "last_round_id": "round-1",
            }))
            patches = self._cycle_patches(round_id="round-2")

            def crash(*args, **kwargs):
                state = read_json(instance.state_file)
                self.assertEqual(state["last_round_id"], "round-2")
                self.assertIn("round_result", instance.events_file.read_text())
                raise RuntimeError("hook crashed")

            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(controller, "candidate_dirty", return_value=False), \
                    patch.object(controller, "run_hook", side_effect=crash):
                with self.assertRaisesRegex(RuntimeError, "hook crashed"):
                    instance.cycle()
            self.assertEqual(read_json(instance.state_file)["last_round_id"], "round-2")

    def test_autonomous_mode_invokes_hook_at_most_once_per_cycle(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            instance = make_controller(
                root, improve_command="agent", mode="autonomous",
                max_improvements_per_round=3,
            )
            instance.state_file.parent.mkdir(parents=True)
            instance.state_file.write_text(json.dumps({
                "initialized": True,
                "cascade_head": "head-1",
                "chain": {"digest": "chain-1", "values": {}},
                "eval_pool_revision": "pool-1",
                "last_round_id": "round-1",
            }))
            (instance.eval_pool_dir / "snapshots").mkdir(parents=True)
            patches = self._cycle_patches(round_id="round-2")
            hook_result = {"exit_code": 0, "stdout": "", "stderr": ""}
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(controller, "candidate_dirty", return_value=True), \
                    patch.object(controller, "run_hook", return_value=hook_result) as hook:
                instance.cycle()
            self.assertEqual(hook.call_count, 1)

class DirtyCascadeTests(TestCase):
    def test_dirty_checkout_blocks_before_any_fetch(self):
        with TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo,
                           check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            tracked = repo / "chain.toml"
            tracked.write_text("value = 1\n")
            subprocess.run(["git", "add", "chain.toml"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
            tracked.write_text("value = 2\n")

            with self.assertRaises(CascadeDirty) as caught:
                refresh_cascade(repo)
            self.assertIn("chain.toml", caught.exception.paths)
            self.assertIn("never commit", str(caught.exception))

    def test_main_emits_one_dirty_event_and_exits_nonzero(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.jsonl"
            dirty = CascadeDirty(root / "cascade", ["chain.toml"])
            with patch.object(controller, "assert_runtime"), \
                    patch.object(controller.Controller, "cycle", side_effect=dirty):
                result = controller.main([
                    "--root", str(root),
                    "--events-file", str(events),
                    "--once",
                ])
            self.assertEqual(result, 1)
            records = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([record["type"] for record in records], ["cascade_dirty"])

    def test_hook_timeout_defaults_to_two_hours(self):
        args = controller.build_parser().parse_args([])
        self.assertEqual(args.hook_timeout, 2 * 3600)
