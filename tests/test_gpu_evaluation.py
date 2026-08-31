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

    def __init__(self, events, *, fail_remote=False, probe_failures=0,
                 launch_failures=0, log_exists=False):
        self.name = "paid-pod"
        self.ip = "127.0.0.1"
        self.port = "22"
        self.price_per_hour = 1.0
        self.ttl_iso = ""
        self.events = events
        self.fail_remote = fail_remote
        self.probe_failures = probe_failures
        self.launch_failures = launch_failures
        self.log_exists = log_exists
        self.commands = []
        self.launches = []
        self.launch_attempts = 0
        self.probes = 0
        self.pushes = []

    def ssh(self, command, timeout=600):
        self.commands.append(command)
        if "CASCADE_PRESENT" in command:
            self.events.append("present-probe")
            if self.log_exists:
                return subprocess.CompletedProcess([], 0, "CASCADE_PRESENT\n", "")
            return subprocess.CompletedProcess([], 1, "", "")
        if "CASCADE_LAUNCHED" in command:
            self.launch_attempts += 1
            if self.launch_attempts <= self.launch_failures:
                self.events.append("launch-fail")
                return subprocess.CompletedProcess([], 255, "", "connection reset")
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
        self.pushes.append((local, remote))

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

    def run_with_fakes(self, root: Path, *, fail_remote=False, probe_failures=0,
                       expected_hours=4.0, **run_kwargs):
        events = []
        pod = FakePod(events, fail_remote=fail_remote, probe_failures=probe_failures)
        fetch_ok = subprocess.CompletedProcess([], 0, "", "")
        event = {"context": {
            "candidate_path": "generators/candidate", "estimated_hours": 3,
        }}

        def start_puller(*args, **kwargs):
            events.append("puller_start")
            return FakePuller(events)

        def fake_rent(*args, **kwargs):
            events.append("rent")
            return pod

        with patch.object(gpu_eval.subprocess, "run", return_value=fetch_ok), \
                patch.object(gpu_eval.time, "sleep"), \
                patch.object(pods, "rent", side_effect=fake_rent), \
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
                    gpu_eval.run(event, root=root, **run_kwargs)
            else:
                self.assertEqual(gpu_eval.run(event, root=root, **run_kwargs), 0)
        # TTL must cover what is actually queued (seeds x arms x train hours),
        # not just the approval's single-candidate estimate.
        ttl.assert_called_once_with(pod, expected_hours)
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

    def add_candidate(self, root: Path, name: str) -> None:
        variant = root / "generators" / name
        variant.mkdir(parents=True)
        for filename in ("generator.py", "config.json", "requirements.txt"):
            (variant / filename).write_text("{}" if filename == "config.json" else "")

    def add_warm_dir(self, root: Path, name: str) -> Path:
        warm = root / "warm" / name
        warm.mkdir(parents=True)
        (warm / "model.safetensors").write_bytes(b"offline fixture")
        return warm

    def test_multiple_candidates_share_the_pod_and_one_king_control(self):
        with TemporaryDirectory() as directory:
            root = self.make_root(directory)
            self.add_candidate(root, "variant-b")
            events, pod = self.run_with_fakes(
                root, candidates_arg="candidate,variant-b", parallel=2,
                expected_hours=6.0)
        # One king arm per seed, however many variants ride along.
        self.assertEqual(len(pod.launches), 6)
        for seed_launches in (pod.launches[:3], pod.launches[3:]):
            names = [launch.split("generators/")[1].split(" ")[0].split("'")[0]
                     for launch in seed_launches]
            self.assertEqual(names[0], "king-control")
            self.assertEqual(sorted(names[1:]), ["candidate", "variant-b"])
        self.assertEqual(events.count("rent"), 1)

    def test_bare_candidate_names_resolve_under_generators(self):
        with TemporaryDirectory() as directory:
            root = self.make_root(directory)
            candidates = gpu_eval.parse_candidates(root, {}, "candidate")
        self.assertEqual([c.name for c in candidates], ["candidate"])

    def test_warm_map_pins_each_seed_to_one_init_for_both_arms(self):
        with TemporaryDirectory() as directory:
            root = self.make_root(directory)
            self.add_warm_dir(root, "r27")
            self.add_warm_dir(root, "r29")
            _, pod = self.run_with_fakes(
                root, warm_map_arg="4=warm/r27,9=warm/r29")
        self.assertEqual(len(pod.launches), 4)
        for launch in pod.launches[:2]:      # seed 4: king and candidate alike
            self.assertIn("--warm-init", launch)
            self.assertIn("warm/r27", launch)
        for launch in pod.launches[2:]:      # seed 9
            self.assertIn("warm/r29", launch)
        # The warm checkpoints themselves were pushed to the pod.
        pushed = [remote for _, remote in pod.pushes]
        self.assertTrue(any(remote.endswith("/warm/r27/") for remote in pushed))
        self.assertTrue(any(remote.endswith("/warm/r29/") for remote in pushed))

    def test_warm_map_must_cover_the_eval_seeds_exactly(self):
        with TemporaryDirectory() as directory:
            root = self.make_root(directory)
            self.add_warm_dir(root, "r27")
            with self.assertRaisesRegex(ValueError, "unmapped seeds: \\[9\\]"):
                gpu_eval.parse_warm_map("4=warm/r27", [4, 9], root)
            with self.assertRaisesRegex(ValueError, "mapped but not evaluated"):
                gpu_eval.parse_warm_map("4=warm/r27,5=warm/r27", [4], root)
            with self.assertRaisesRegex(ValueError, "not of the form seed=path"):
                gpu_eval.parse_warm_map("r27", [4], root)
            with self.assertRaisesRegex(ValueError, "not a non-empty directory"):
                gpu_eval.parse_warm_map("4=warm/missing", [4], root)

    def test_warm_map_rejects_colliding_basenames(self):
        with TemporaryDirectory() as directory:
            root = self.make_root(directory)
            (root / "a/r27").mkdir(parents=True)
            (root / "a/r27/w").write_bytes(b"x")
            (root / "b/r27").mkdir(parents=True)
            (root / "b/r27/w").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "share the basename"):
                gpu_eval.parse_warm_map("4=a/r27,9=b/r27", [4, 9], root)

    def test_launch_retries_transient_ssh_failures(self):
        pod = FakePod([], launch_failures=2)
        with patch.object(gpu_eval.time, "sleep"):
            gpu_eval.checked_remote(pod, ["/bin/true"], 3600,
                                    label="king-control-seed0",
                                    say=lambda *a, **k: None)
        self.assertEqual(pod.launch_attempts, 3)
        self.assertEqual(len(pod.launches), 1)

    def test_a_timed_out_launch_that_actually_started_is_not_relaunched(self):
        # The failed ssh may have executed the command anyway; a blind retry
        # would run the training twice. The log's existence says it started.
        pod = FakePod([], launch_failures=1, log_exists=True)
        with patch.object(gpu_eval.time, "sleep"):
            gpu_eval.checked_remote(pod, ["/bin/true"], 3600,
                                    label="king-control-seed0",
                                    say=lambda *a, **k: None)
        self.assertEqual(pod.launch_attempts, 1)
        self.assertEqual(len(pod.launches), 0)
        self.assertIn("present-probe", pod.events)

    def test_resume_skips_scored_arms_and_pushes_completed_checkpoints(self):
        with TemporaryDirectory() as directory:
            root = self.make_root(directory)
            scored = root / "scores/king-control__seed4"
            scored.mkdir(parents=True)
            (scored / "summary.json").write_text("{}")
            trained = root / "ckpts/candidate__seed4"
            trained.mkdir(parents=True)
            (trained / "TRAINED.json").write_text("{}")
            _, pod = self.run_with_fakes(root, resume=True)
        labels = [launch for launch in pod.launches]
        self.assertEqual(len(labels), 3)     # king seed4 already scored
        self.assertNotIn("king-control-seed4", " ".join(
            launch.split(".status")[0] for launch in labels))
        pushed = [remote for _, remote in pod.pushes]
        self.assertTrue(any(remote.endswith("/ckpts/candidate__seed4/")
                            for remote in pushed))

    def test_without_resume_nothing_is_skipped_or_pushed_back(self):
        with TemporaryDirectory() as directory:
            root = self.make_root(directory)
            scored = root / "scores/king-control__seed4"
            scored.mkdir(parents=True)
            (scored / "summary.json").write_text("{}")
            _, pod = self.run_with_fakes(root)
        self.assertEqual(len(pod.launches), 4)

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

    def test_rent_returns_the_pod_it_created(self):
        listing = json.dumps([{"name": "eval-1", "ssh_cmd": "ssh root@1.2.3.4 -p 42",
                               "price_per_hour": 1.5, "removal_scheduled_at": "soon"}])

        def fake_lium(*args, timeout=900):
            command = args[0]
            if command == "up":
                return subprocess.CompletedProcess([], 0, "rented", "")
            if command == "ps":
                return subprocess.CompletedProcess([], 0, listing, "")
            raise AssertionError(f"unexpected lium call: {args}")

        with patch.object(pods, "_lium", side_effect=fake_lium):
            pod = pods.rent("eval-1", ttl_hours=1)
        self.assertEqual((pod.name, pod.ip, pod.port), ("eval-1", "1.2.3.4", "42"))

    def test_rent_refuses_an_ambiguous_name_match(self):
        row = {"name": "eval-1", "ssh_cmd": "ssh root@1.2.3.4 -p 42",
               "price_per_hour": 1.5, "removal_scheduled_at": ""}
        listing = json.dumps([row, dict(row, ssh_cmd="ssh root@5.6.7.8 -p 43")])

        def fake_lium(*args, timeout=900):
            payload = listing if args[0] == "ps" else "rented"
            return subprocess.CompletedProcess([], 0, payload, "")

        with patch.object(pods, "_lium", side_effect=fake_lium):
            with self.assertRaisesRegex(RuntimeError, "refusing to guess"):
                pods.rent("eval-1", ttl_hours=1)

    def test_rent_stops_a_pod_that_never_becomes_reachable(self):
        calls = []

        def fake_lium(*args, timeout=900):
            calls.append(args[0])
            if args[0] == "ps":
                return subprocess.CompletedProcess([], 0, "[]", "")
            return subprocess.CompletedProcess([], 0, "ok", "")

        with patch.object(pods, "_lium", side_effect=fake_lium):
            with self.assertRaises(TimeoutError):
                pods.rent("eval-1", ttl_hours=1, ready_timeout=0)
        # The rental went through; discovery failing must not leave it billing.
        self.assertIn("rm", calls)

    def test_assert_single_gpu_can_be_scoped_to_one_pod(self):
        listing = json.dumps([
            {"name": "mine", "gpu_count": 1},
            {"name": "someone-elses", "gpu_count": 8, "config": "8xRTX4090",
             "price_per_hour": 2.56},
        ])
        mine = pods.Pod(name="mine", ip="1.2.3.4", port="42",
                        price_per_hour=1.0, ttl_iso="")
        ps_ok = subprocess.CompletedProcess([], 0, listing, "")
        with patch.object(pods, "_lium", return_value=ps_ok):
            pods.assert_single_gpu(mine)             # scoped: fine
            with self.assertRaisesRegex(RuntimeError, "multi-GPU"):
                pods.assert_single_gpu()             # account-wide: caught

    def test_puller_can_also_mirror_checkpoints(self):
        with TemporaryDirectory() as directory:
            pod = pods.Pod(name="p", ip="1.2.3.4", port="42",
                           price_per_hour=1.0, ttl_iso="")
            with patch.object(pods.subprocess, "Popen") as popen, \
                    patch.object(pods.subprocess, "run"):
                pods.start_puller([pod], f"{directory}/scores",
                                  local_ckpts=f"{directory}/ckpts")
                script = Path(popen.call_args.args[0][0]).read_text()
            self.assertTrue(Path(directory, "ckpts").is_dir())
        self.assertIn("/scores/ .", script)
        self.assertIn("/ckpts/ .", script)

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
