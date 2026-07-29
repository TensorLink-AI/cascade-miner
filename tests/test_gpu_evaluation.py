"""Offline tests for the paid GPU evaluation boundary (no network or rental)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from miner import pods


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run-gpu-evaluation"


def load_script():
    loader = importlib.machinery.SourceFileLoader("run_gpu_evaluation", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


gpu_eval = load_script()


class FakePuller:
    def __init__(self, events):
        self.events = events

    def terminate(self):
        self.events.append("puller_terminate")

    def wait(self, timeout):
        self.events.append("puller_wait")


class FakePod:
    def __init__(self, events, *, fail_remote=False):
        self.name = "paid-pod"
        self.ip = "127.0.0.1"
        self.port = "22"
        self.price_per_hour = 1.0
        self.ttl_iso = ""
        self.events = events
        self.fail_remote = fail_remote
        self.commands = []

    def ssh(self, command, timeout=600):
        self.events.append("remote")
        self.commands.append(command)
        code = 7 if self.fail_remote else 0
        return subprocess.CompletedProcess([], code, "", "remote boom" if code else "")

    def push(self, local, remote, timeout=900):
        self.events.append("push")

    def pull(self, remote, local, timeout=900):
        self.events.append("pull")


class GpuEvaluationTests(TestCase):
    def make_root(self, directory: str) -> Path:
        root = Path(directory)
        candidate = root / "generators/candidate"
        candidate.mkdir(parents=True)
        for name in ("generator.py", "config.json", "requirements.txt"):
            (candidate / name).write_text("{}" if name == "config.json" else "")
        (root / "pools/snapshots/2026-07-16").mkdir(parents=True)
        (root / "runs").mkdir()
        (root / "runs/controller-state.json").write_text(json.dumps({
            "last_receipt": {"summary": {"king_gen_ref": "owner/king@sha256:abc"}}
        }))
        return root

    def run_with_fakes(self, root: Path, *, fail_remote=False):
        events = []
        pod = FakePod(events, fail_remote=fail_remote)
        fetch_ok = subprocess.CompletedProcess([], 0, "", "")
        event = {"context": {
            "candidate_path": "generators/candidate", "estimated_hours": 3,
        }}

        def start_puller(*args, **kwargs):
            events.append("puller_start")
            return FakePuller(events)

        with patch.object(gpu_eval.subprocess, "run", return_value=fetch_ok), \
                patch.object(gpu_eval, "wait_for_pod", return_value=pod), \
                patch.object(pods, "rent", side_effect=lambda *a, **k: events.append("rent")), \
                patch.object(pods, "assert_single_gpu"), \
                patch.object(pods, "check_ttl_covers") as ttl, \
                patch.object(pods, "provision", side_effect=lambda *a, **k: events.append("provision")), \
                patch.object(pods, "assert_deps"), \
                patch.object(pods, "start_puller", side_effect=start_puller), \
                patch.object(pods, "stop", side_effect=lambda name: events.append(("stop", name))), \
                patch.dict(gpu_eval.os.environ, {
                    "CASCADE_MINER_STATE": str(root / "runs/controller-state.json"),
                    "CASCADE_EVAL_SEEDS": "4,9",
                }, clear=False):
            if fail_remote:
                with self.assertRaisesRegex(RuntimeError, "remote evaluation failed"):
                    gpu_eval.run(event, root=root)
            else:
                self.assertEqual(gpu_eval.run(event, root=root), 0)
        ttl.assert_called_once_with(pod, 3.0)
        return events, pod

    def test_paired_seeds_pull_before_training_and_stop_after_success(self):
        with TemporaryDirectory() as directory:
            events, pod = self.run_with_fakes(self.make_root(directory))
        self.assertLess(events.index("puller_start"), events.index("remote"))
        self.assertEqual(len(pod.commands), 4)
        self.assertIn("generators/king-control", pod.commands[0])
        self.assertIn("generators/candidate", pod.commands[1])
        self.assertIn("--seed 4", pod.commands[0])
        self.assertIn("--seed 4", pod.commands[1])
        self.assertIn("--seed 9", pod.commands[2])
        self.assertIn("--seed 9", pod.commands[3])
        self.assertEqual(events[-1], ("stop", "paid-pod"))

    def test_remote_failure_still_stops_named_pod(self):
        with TemporaryDirectory() as directory:
            events, _ = self.run_with_fakes(self.make_root(directory), fail_remote=True)
        self.assertIn("puller_terminate", events)
        self.assertEqual(events[-1], ("stop", "paid-pod"))

    def test_approved_event_shape_and_candidate_containment(self):
        context = gpu_eval.approval_context({
            "approval": {"context": {"candidate_path": "generators/candidate"}}
        })
        self.assertEqual(context["candidate_path"], "generators/candidate")
        with TemporaryDirectory() as directory:
            root = self.make_root(directory)
            with self.assertRaisesRegex(ValueError, "direct child"):
                gpu_eval.contained_candidate(root, "../outside")

    def test_wallet_ops_are_executable_refusing_stubs(self):
        for name in ("create-next-hotkey", "register-next-hotkey", "submit-candidate"):
            path = ROOT / "ops" / name
            result = subprocess.run([str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("REFUSED", result.stderr)

    def test_bootstrap_candidate_has_submission_layout(self):
        candidate = ROOT / "generators/candidate"
        self.assertEqual(
            {path.name for path in candidate.iterdir() if path.is_file()},
            {"generator.py", "config.json", "requirements.txt"},
        )

    def test_example_env_wires_the_shipped_approved_command(self):
        env = (ROOT / "example.env").read_text()
        self.assertIn(
            "CASCADE_APPROVED_EVAL_COMMAND=.venv/bin/python scripts/run-gpu-evaluation",
            env,
        )
