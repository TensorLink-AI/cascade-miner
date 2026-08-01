"""Offline tests for the improvement-hook backends, including hermes-native."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.improve_candidate import (
    REQUEST_NAME,
    RESPONSE_NAME,
    agent_command,
    choose_agent,
    request_improvement,
)


def answer(root: Path, status: str, *, request_id=None, delay=0.0) -> threading.Thread:
    """Play the in-session agent: wait for the request, then reply to it."""
    def worker():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            request = root / REQUEST_NAME
            if request.is_file():
                payload = json.loads(request.read_text(encoding="utf-8"))
                time.sleep(delay)
                (root / RESPONSE_NAME).write_text(json.dumps({
                    "id": request_id or payload["id"],
                    "status": status,
                    "detail": "swapped one family",
                }), encoding="utf-8")
                return
            time.sleep(0.01)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


class HermesNativeTests(TestCase):
    def call(self, root: Path, *, timeout=5, poll=1):
        return request_improvement(
            root, "prompt body", '{"type": "round_result"}', "human",
            timeout=timeout, poll=poll, say=lambda *a, **k: None,
        )

    def test_completed_response_ends_the_pass_successfully(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            thread = answer(root, "completed")
            with patch("scripts.improve_candidate.time.sleep", lambda _: None):
                self.assertEqual(self.call(root), 0)
            thread.join(5)
            # A satisfied request is cleared so the next cycle starts clean.
            self.assertFalse((root / REQUEST_NAME).exists())

    def test_failure_statuses_are_reported_as_a_failed_pass(self):
        for status in ("failed", "rejected", "skipped"):
            with self.subTest(status=status), TemporaryDirectory() as directory:
                root = Path(directory)
                thread = answer(root, status)
                with patch("scripts.improve_candidate.time.sleep", lambda _: None):
                    self.assertEqual(self.call(root), 1)
                thread.join(5)

    def test_request_records_the_prompt_and_event_for_the_session(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            captured = {}

            def capture():
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    request = root / REQUEST_NAME
                    if request.is_file():
                        payload = json.loads(request.read_text(encoding="utf-8"))
                        captured.update(payload)
                        (root / RESPONSE_NAME).write_text(json.dumps(
                            {"id": payload["id"], "status": "completed"}
                        ), encoding="utf-8")
                        return
                    time.sleep(0.01)

            thread = threading.Thread(target=capture, daemon=True)
            thread.start()
            with patch("scripts.improve_candidate.time.sleep", lambda _: None):
                self.assertEqual(self.call(root), 0)
            thread.join(5)
        self.assertEqual(captured["prompt"], "prompt body")
        self.assertEqual(captured["event"], {"type": "round_result"})
        self.assertEqual(captured["mode"], "human")
        self.assertEqual(captured["status"], "pending")

    def test_no_response_times_out_and_leaves_the_request_for_inspection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.improve_candidate.time.sleep", lambda _: None):
                self.assertEqual(self.call(root, timeout=0), 124)
            self.assertTrue((root / REQUEST_NAME).exists())

    def test_a_stale_response_cannot_satisfy_a_new_request(self):
        """A leftover reply from the previous cycle must not end this pass."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            (root / RESPONSE_NAME).write_text(json.dumps(
                {"id": "an-older-request", "status": "completed"}
            ), encoding="utf-8")
            with patch("scripts.improve_candidate.time.sleep", lambda _: None):
                self.assertEqual(self.call(root, timeout=0), 124)

    def test_a_mismatched_id_is_ignored_rather_than_accepted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # Answers this pass's request, but stamps someone else's id on it.
            thread = answer(root, "completed", request_id="not-the-pending-one")
            with patch("scripts.improve_candidate.time.sleep", lambda _: None):
                self.assertEqual(self.call(root, timeout=1), 124)
            thread.join(5)
            self.assertTrue((root / RESPONSE_NAME).exists())


class AgentSelectionTests(TestCase):
    def test_hermes_native_is_never_chosen_automatically(self):
        """A missing CLI is usually a broken install, not an agent session."""
        with patch("scripts.improve_candidate.shutil.which", return_value=None), \
                patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "hermes-native"):
                choose_agent("auto")

    def test_explicit_selection_is_honoured(self):
        self.assertEqual(choose_agent("hermes-native"), "hermes-native")

    def test_hermes_native_has_no_subprocess_command(self):
        """It must go through the file handshake, never a `hermes` binary."""
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            agent_command("hermes-native", Path("/tmp"), "prompt")
