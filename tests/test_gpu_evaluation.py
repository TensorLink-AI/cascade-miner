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
    """Emulates the launch/poll protocol: nohup, then a status marker file."""

    def __init__(self, events, *, fail_remote=False, probe_failures=0):
        self.name = "paid-pod"
        self.ip = "127.0.0.1"
        self.port = "22"
        self.price_per_hour = 1.0
        self.ttl_iso = ""
        self.events = events
        self.fail_remote = fail_remote
        self.probe_failures = probe_failures
        self.commands = []
        self.launches = []
        self.probes = 0

    def ssh(self, command, timeout=600):
        self.commands.append(command)
        if "CASCADE_LAUNCHED" in command:
            self.events.append("launch")
            self.launches.append(command)
            return subprocess.CompletedProcess([], 0, "CASCADE_LAUNCHED\n", "")
        if "CASCADE_STATUS" in command:
            self.events.append("probe")
            self.probes += 1
            if self.probes <= self.probe_failures:
                return subprocess.CompletedProcess([], 255, "", "connection reset")
            code = 7 if self.fail_remote else 0
            return subprocess.CompletedProcess(
                [], 0,
                f"CASCADE_STATUS={code}\nCASCADE_LOG=step 100\nCASCADE_GPU=98 %, 12 MiB\n",
                "",
            )
        self.events.append("remote")
        return subprocess.CompletedProcess([], 0, "remote boom", "")

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

    def run_with_fakes(self, root: Path, *, fail_remote=False, probe_failures=0):
        events = []
        pod = FakePod(events, fail_remote=fail_remote, probe_failures=probe_failures)
        fetch_ok = subprocess.CompletedProcess([], 0, "", "")
        event = {"context": {
            "candidate_path": "generators/candidate", "estimated_hours": 3,
        }}

        def start_puller(*args, **kwargs):
            events.append("puller_start")
            return FakePuller(events)

        with patch.object(gpu_eval.subprocess, "run", return_value=fetch_ok), \
                patch.object(gpu_eval.time, "sleep"), \
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
                with self.assertRaisesRegex(RuntimeError, "remote evaluation .* failed"):
                    gpu_eval.run(event, root=root)
            else:
                self.assertEqual(gpu_eval.run(event, root=root), 0)
        ttl.assert_called_once_with(pod, 3.0)
        return events, pod

    def test_paired_seeds_pull_before_training_and_stop_after_success(self):
        with TemporaryDirectory() as directory:
            events, pod = self.run_with_fakes(self.make_root(directory))
        self.assertLess(events.index("puller_start"), events.index("launch"))
        self.assertEqual(len(pod.launches), 4)
        self.assertIn("generators/king-control", pod.launches[0])
        self.assertIn("generators/candidate", pod.launches[1])
        self.assertIn("--seed 4", pod.launches[0])
        self.assertIn("--seed 4", pod.launches[1])
        self.assertIn("--seed 9", pod.launches[2])
        self.assertIn("--seed 9", pod.launches[3])
        self.assertEqual(events[-1], ("stop", "paid-pod"))

    def test_training_is_detached_so_a_dropped_session_cannot_abandon_it(self):
        with TemporaryDirectory() as directory:
            _, pod = self.run_with_fakes(self.make_root(directory))
        for launch in pod.launches:
            self.assertIn("nohup ", launch)
            self.assertIn("</dev/null", launch)
            self.assertRegex(launch, r"echo \$\? > /tmp/cascade-eval-[\w.-]+\.status")
        # Each generator/seed pair gets its own marker, never a shared one.
        markers = {gpu_eval.remote_paths(f"{name}-seed{seed}")
                   for seed in (4, 9) for name in ("king-control", "candidate")}
        self.assertEqual(len(markers), 4)

    def test_transient_probe_failures_do_not_abandon_a_running_evaluation(self):
        with TemporaryDirectory() as directory:
            events, pod = self.run_with_fakes(self.make_root(directory), probe_failures=3)
        self.assertEqual(len(pod.launches), 4)
        self.assertEqual(events[-1], ("stop", "paid-pod"))

    def test_lost_contact_is_reported_rather_than_silently_retried_forever(self):
        pod = FakePod([], probe_failures=999)
        with self.assertRaisesRegex(RuntimeError, "lost contact"):
            with patch.object(gpu_eval.time, "sleep"):
                gpu_eval.checked_remote(pod, ["/bin/true"], 3600,
                                        label="king-control-seed0", say=lambda *a, **k: None)
        self.assertEqual(pod.probes, gpu_eval.MAX_CONSECUTIVE_PROBE_FAILURES)

    def test_nonzero_marker_is_a_failure_and_still_stops_the_named_pod(self):
        with TemporaryDirectory() as directory:
            events, _ = self.run_with_fakes(self.make_root(directory), fail_remote=True)
        self.assertIn("puller_terminate", events)
        self.assertEqual(events[-1], ("stop", "paid-pod"))

    def test_probe_reply_is_parsed_into_status_log_and_gpu(self):
        self.assertEqual(
            gpu_eval.parse_probe("CASCADE_STATUS=0\nCASCADE_LOG=a=b\nCASCADE_GPU=98 %, 1 MiB"),
            ("0", "a=b", "98 %, 1 MiB"),
        )
        # An unfinished run reports no status, which must not read as success.
        self.assertEqual(
            gpu_eval.parse_probe("CASCADE_STATUS=\nCASCADE_LOG=training\nCASCADE_GPU="),
            ("", "training", ""),
        )

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
            'CASCADE_APPROVED_EVAL_COMMAND=".venv/bin/python scripts/run-gpu-evaluation"',
            env,
        )

    def test_example_env_quotes_every_value_containing_a_space(self):
        """`set -a; source .env` executes the second word of an unquoted value."""
        for line in (ROOT / "example.env").read_text().splitlines():
            if line.startswith("#") or "=" not in line:
                continue
            value = line.split("=", 1)[1]
            if " " in value:
                self.assertTrue(
                    value.startswith(('"', "'")) and value.endswith(('"', "'")),
                    f"unquoted value with a space would be executed on source: {line}",
                )
