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
    local_snapshots,
    queue_approval,
    read_json,
    refresh_cascade,
    resolve_snapshot,
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
    """The eval pool is 10k+ files, so every download decision is a cost."""

    SNAPSHOTS = ("2026-06-16", "2026-07-01", "2026-07-16")

    def _fake_hub(self, root: Path, *, noisy: bool = False) -> Path:
        package = root / "huggingface_hub"
        package.mkdir()
        noise = "sys.stderr.write('x' * 600000 + 'REAL_ERROR_TAIL\\n')" if noisy else ""
        package.joinpath("__init__.py").write_text(
            "import json, os, pathlib, sys\n"
            f"SNAPSHOTS = {list(self.SNAPSHOTS)!r}\n"
            "class Sibling:\n"
            "    def __init__(self, rfilename, size):\n"
            "        self.rfilename, self.size = rfilename, size\n"
            "class Info:\n"
            "    sha = 'revision-2'\n"
            "    siblings = ([Sibling('README.md', 10)]\n"
            "        + [Sibling(f'snapshots/{name}/windows.npy', 2_000_000)\n"
            "           for name in SNAPSHOTS]\n"
            "        + [Sibling(f'snapshots/{name}/meta.json', 20)\n"
            "           for name in SNAPSHOTS])\n"
            "class HfApi:\n"
            "    def __init__(self, token=None):\n"
            "        if token != 'hf_test': raise RuntimeError('missing token')\n"
            "    def dataset_info(self, repo_id, files_metadata=False):\n"
            "        assert os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS') == '1'\n"
            "        assert files_metadata, 'sizes are needed for the progress line'\n"
            f"        {noise}\n"
            + ("        raise RuntimeError('boom')\n" if noisy else "        return Info()\n")
            + "def snapshot_download(**kwargs):\n"
              "    dest = pathlib.Path(kwargs['local_dir'])\n"
              "    patterns = kwargs.get('allow_patterns')\n"
              "    dest.mkdir(parents=True, exist_ok=True)\n"
              "    dest.joinpath('downloaded').write_text(json.dumps(\n"
              "        {'revision': kwargs['revision'], 'allow_patterns': patterns}))\n"
              "    for name in SNAPSHOTS:\n"
              "        if patterns and not any(\n"
              "                p.startswith(f'snapshots/{name}/') for p in patterns):\n"
              "            continue\n"
              "        directory = dest / 'snapshots' / name\n"
              "        directory.mkdir(parents=True, exist_ok=True)\n"
              "        directory.joinpath('windows.npy').write_text('x')\n"
              "    return str(dest)\n"
        )
        return root

    def _sync(self, root: Path, dest: Path, known: str = "", **kwargs):
        messages: list[str] = []
        with patch.dict(os.environ, {
            "PYTHONPATH": str(root), "HF_TOKEN": "hf_test",
        }, clear=False):
            result = sync_eval_pool(
                Path(sys.executable), root, "owner/pool", dest, known,
                log=messages.append, **kwargs,
            )
        return result, messages

    def _seed_local(self, dest: Path, *names: str) -> None:
        for name in names:
            directory = dest / "snapshots" / name
            directory.mkdir(parents=True)
            directory.joinpath("windows.npy").write_text("x")

    def _download_call(self, dest: Path) -> dict:
        return json.loads((dest / "downloaded").read_text())

    def test_default_selector_downloads_only_the_latest_snapshot(self):
        with TemporaryDirectory() as directory:
            root = self._fake_hub(Path(directory))
            dest = root / "pool"
            result, messages = self._sync(root, dest, "revision-1")
            self.assertTrue(result["downloaded"])
            self.assertEqual(result["snapshot"], "2026-07-16")
            self.assertEqual(result["available_snapshots"], list(self.SNAPSHOTS))
            self.assertEqual(
                self._download_call(dest)["allow_patterns"],
                ["snapshots/2026-07-16/*", "*.json", "*.md"],
            )
            self.assertEqual(local_snapshots(dest), ["2026-07-16"])
            self.assertTrue(
                any("downloading snapshot 2026-07-16" in line and "files" in line
                    for line in messages),
                messages,
            )

    def test_explicit_snapshot_is_fetched_and_unknown_names_list_options(self):
        with TemporaryDirectory() as directory:
            root = self._fake_hub(Path(directory))
            dest = root / "pool"
            result, _ = self._sync(root, dest, snapshot="2026-07-01")
            self.assertEqual(result["snapshot"], "2026-07-01")
            self.assertEqual(local_snapshots(dest), ["2026-07-01"])
            with self.assertRaisesRegex(RuntimeError, "2026-06-16, 2026-07-01"):
                self._sync(root, root / "other", snapshot="2026-12-31")

    def test_all_selector_mirrors_every_snapshot(self):
        with TemporaryDirectory() as directory:
            root = self._fake_hub(Path(directory))
            dest = root / "pool"
            result, _ = self._sync(root, dest, snapshot="all")
            self.assertEqual(result["snapshot"], "all")
            self.assertIsNone(self._download_call(dest)["allow_patterns"])
            self.assertEqual(local_snapshots(dest), list(self.SNAPSHOTS))

    def test_unchanged_revision_skips_snapshot_download_and_uses_token(self):
        with TemporaryDirectory() as directory:
            root = self._fake_hub(Path(directory))
            dest = root / "pool"
            self._seed_local(dest, "2026-07-16")
            result, _ = self._sync(root, dest, "revision-2")
            self.assertFalse(result["downloaded"])
            self.assertEqual(result["reason"], "revision unchanged")
            self.assertFalse((dest / "downloaded").exists())

    def test_first_run_with_a_populated_pool_skips_the_download(self):
        with TemporaryDirectory() as directory:
            root = self._fake_hub(Path(directory))
            dest = root / "pool"
            self._seed_local(dest, "2026-07-16")
            result, _ = self._sync(root, dest, "")
            self.assertFalse(result["downloaded"])
            self.assertIn("already present", result["reason"])
            self.assertEqual(result["revision"], "revision-2")

    def test_a_partial_mirror_is_not_treated_as_a_complete_one(self):
        with TemporaryDirectory() as directory:
            root = self._fake_hub(Path(directory))
            dest = root / "pool"
            self._seed_local(dest, "2026-07-16")
            result, _ = self._sync(root, dest, "", snapshot="all")
            self.assertTrue(result["downloaded"])
            self.assertEqual(local_snapshots(dest), list(self.SNAPSHOTS))

    def test_missing_snapshot_downloads_even_when_the_revision_matches(self):
        with TemporaryDirectory() as directory:
            root = self._fake_hub(Path(directory))
            dest = root / "pool"
            self._seed_local(dest, "2026-06-16")
            result, _ = self._sync(root, dest, "revision-2")
            self.assertTrue(result["downloaded"])
            self.assertEqual(local_snapshots(dest), ["2026-06-16", "2026-07-16"])

    def test_download_failure_is_retried_before_giving_up(self):
        with TemporaryDirectory() as directory:
            root = self._fake_hub(Path(directory))
            calls: list[int] = []
            real = controller.pool_child

            def flaky(python, cwd, script, args, timeout):
                if script is controller.POOL_DOWNLOAD_SCRIPT:
                    calls.append(1)
                    if len(calls) == 1:
                        raise RuntimeError("eval-pool sync failed: 429 rate limited")
                return real(python, cwd, script, args, timeout)

            with patch.object(controller, "pool_child", side_effect=flaky), \
                    patch.object(controller.time, "sleep"):
                result, messages = self._sync(root, root / "pool", "revision-1")
            self.assertTrue(result["downloaded"])
            self.assertEqual(len(calls), 2)
            self.assertTrue(any("retrying" in line for line in messages), messages)

    def test_pool_error_is_quiet_and_truncated(self):
        with TemporaryDirectory() as directory:
            root = self._fake_hub(Path(directory), noisy=True)
            with self.assertRaises(RuntimeError) as caught:
                self._sync(root, root / "pool", "")
            message = str(caught.exception)
            self.assertIn("REAL_ERROR_TAIL", message)
            self.assertLess(len(message), 2100)

    def test_latest_falls_back_to_everything_when_nothing_is_dated(self):
        self.assertEqual(resolve_snapshot("latest", []), "all")
        self.assertEqual(resolve_snapshot("", ["2026-07-16"]), "2026-07-16")


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
    POOL = {
        "repo_id": "owner/pool", "revision": "pool-1", "downloaded": False,
        "snapshot": "2026-07-16", "reason": "revision unchanged",
        "available_snapshots": ["2026-07-16"], "local_snapshots": ["2026-07-16"],
    }

    def _cycle_patches(self, *, round_id: str = "round-1", audit=None):
        return (
            patch.object(controller, "refresh_cascade",
                         return_value=("head-1", "head-1")),
            patch.object(controller, "chain_snapshot",
                         return_value={"digest": "chain-1", "values": {}}),
            patch.object(controller, "sync_eval_pool", return_value=dict(self.POOL)),
            patch.object(controller, "audit_latest", **(
                {"side_effect": audit} if audit is not None else
                {"return_value": {"round_id": round_id, "status": "scored", "ok": True}}
            )),
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

    def test_pool_revision_survives_a_later_failure_in_the_same_cycle(self):
        """A failed audit must not cost another full pool download next poll."""
        with TemporaryDirectory() as directory:
            instance = make_controller(Path(directory))
            patches = self._cycle_patches(audit=RuntimeError("chain unreachable"))
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(RuntimeError, "chain unreachable"):
                    instance.cycle()
            state = read_json(instance.state_file)
            self.assertEqual(state["eval_pool_revision"], "pool-1")
            self.assertEqual(state["eval_pool_snapshot"], "2026-07-16")
            self.assertNotIn("initialized", state)
            self.assertIn("eval_pool_update", instance.events_file.read_text())

    def test_pool_selector_reaches_the_sync_and_reuses_the_known_revision(self):
        with TemporaryDirectory() as directory:
            instance = make_controller(Path(directory), eval_pool_snapshot="all")
            instance.state_file.parent.mkdir(parents=True)
            instance.state_file.write_text(json.dumps({"eval_pool_revision": "pool-0"}))
            patches = self._cycle_patches()
            with patches[0], patches[1], patches[2] as sync, patches[3]:
                instance.cycle()
            self.assertEqual(sync.call_args.args[4], "pool-0")
            self.assertEqual(sync.call_args.kwargs["snapshot"], "all")

    def _seeded_autonomous(self, root: Path, **overrides) -> Controller:
        values = {"improve_command": "agent", "mode": "autonomous",
                  "max_improvements_per_round": 3,
                  "approved_eval_command": "evaluate"}
        values.update(overrides)
        instance = make_controller(root, **values)
        instance.state_file.parent.mkdir(parents=True)
        instance.state_file.write_text(json.dumps({
            "initialized": True,
            "cascade_head": "head-1",
            "chain": {"digest": "chain-1", "values": {}},
            "eval_pool_revision": "pool-1",
            "last_round_id": "round-1",
        }))
        (instance.eval_pool_dir / "snapshots").mkdir(parents=True)
        return instance

    def test_autonomous_mode_chains_passes_up_to_the_round_cap(self):
        """--max-improvements-per-round is a real contract: a successful
        evaluation starts the next improvement pass in the same poll."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            instance = self._seeded_autonomous(root)
            patches = self._cycle_patches(round_id="round-2")
            hook_result = {"exit_code": 0, "stdout": "", "stderr": ""}
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(controller, "candidate_dirty", return_value=True), \
                    patch.object(controller, "run_hook", return_value=hook_result) as hook:
                instance.cycle()
            # Three improve passes, each followed by its evaluation.
            commands = [call.args[0] for call in hook.call_args_list]
            self.assertEqual(commands,
                             ["agent", "evaluate"] * 3)
            state = read_json(instance.state_file)
            self.assertEqual(state["improvement_passes"]["round-2"], 3)

    def test_a_failed_evaluation_stops_the_chain(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            instance = self._seeded_autonomous(root)
            patches = self._cycle_patches(round_id="round-2")
            results = iter([
                {"exit_code": 0, "stdout": "", "stderr": ""},   # improve 1
                {"exit_code": 1, "stdout": "", "stderr": "oom"},  # eval 1 fails
            ])
            with patches[0], patches[1], patches[2], patches[3], \
                    patch.object(controller, "candidate_dirty", return_value=True), \
                    patch.object(controller, "run_hook",
                                 side_effect=lambda *a, **k: next(results)) as hook:
                instance.cycle()
            self.assertEqual(hook.call_count, 2)
            state = read_json(instance.state_file)
            self.assertEqual(state["improvement_passes"]["round-2"], 1)

    def test_human_mode_still_stops_at_the_first_approval(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            instance = self._seeded_autonomous(root, mode="human")
            patches = self._cycle_patches(round_id="round-2")
            hook_result = {"exit_code": 0, "stdout": "", "stderr": ""}
            # Checked three times: the pending-review guard, the loop entry
            # condition, and the post-improvement "did the pass change
            # anything" probe. Only the last sees the new candidate.
            dirty = patch.object(controller, "candidate_dirty",
                                 side_effect=[False, False, True])
            with patches[0], patches[1], patches[2], patches[3], dirty, \
                    patch.object(controller, "run_hook", return_value=hook_result) as hook:
                instance.cycle()
            # One improvement pass, then the evaluation queues for approval.
            self.assertEqual(hook.call_count, 1)
            approvals = read_json(instance.approvals_file)["requests"]
            self.assertEqual([r["action"] for r in approvals], ["gpu_evaluation"])

class SyncPoolCommandTests(TestCase):
    def test_sync_pool_records_the_revision_without_needing_cascade(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pool = {
                "repo_id": "owner/pool", "revision": "pool-9", "downloaded": True,
                "snapshot": "2026-07-16", "reason": "downloaded",
                "available_snapshots": ["2026-07-16"],
                "local_snapshots": ["2026-07-16"],
            }
            with patch.object(controller, "assert_runtime") as runtime, \
                    patch.object(controller, "sync_eval_pool",
                                 return_value=pool) as sync:
                code = controller.main([
                    "--root", str(root), "--sync-pool",
                    "--eval-pool-snapshot", "2026-07-16",
                ])
            self.assertEqual(code, 0)
            self.assertEqual(runtime.call_args.kwargs["modules"], ("huggingface_hub",))
            self.assertEqual(sync.call_args.kwargs["snapshot"], "2026-07-16")
            state = read_json(root / "runs/controller-state.json")
            self.assertEqual(state["eval_pool_revision"], "pool-9")
            self.assertEqual(state["eval_pool_snapshot"], "2026-07-16")
            # Seeding only the pool must not look like a completed first cycle.
            self.assertNotIn("initialized", state)

    def test_snapshot_selector_defaults_to_latest(self):
        args = controller.build_parser().parse_args([])
        self.assertEqual(args.eval_pool_snapshot, "latest")
        self.assertFalse(args.sync_pool)


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
