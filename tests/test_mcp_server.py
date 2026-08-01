import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from miner import mcp_server
from miner.controller import read_json
from scripts.improve_candidate import TERMINAL_STATUSES


class ServerHarness(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "generators/candidate").mkdir(parents=True)
        self.server = mcp_server.Server(self.root)

    def call(self, name: str, arguments: dict | None = None) -> dict:
        response = self.server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        })
        self.assertNotIn("error", response)
        return response["result"]

    def payload(self, result: dict) -> dict | list:
        self.assertFalse(result["isError"], result["content"][0]["text"])
        return json.loads(result["content"][0]["text"])


class ProtocolTests(ServerHarness):
    def test_initialize_echoes_a_known_protocol_version(self):
        response = self.server.handle({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        })
        result = response["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertEqual(result["serverInfo"]["name"], "cascade-miner")

    def test_initialize_falls_back_on_an_unknown_version(self):
        response = self.server.handle({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2099-01-01"},
        })
        self.assertEqual(response["result"]["protocolVersion"], "2024-11-05")

    def test_notifications_get_no_response(self):
        self.assertIsNone(self.server.handle(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_unknown_method_is_a_jsonrpc_error(self):
        response = self.server.handle(
            {"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
        self.assertEqual(response["error"]["code"], -32601)

    def test_tools_list_names_the_full_surface(self):
        response = self.server.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertEqual(names, {
            "miner_status", "get_state", "get_latest_receipt",
            "list_approvals", "request_action", "get_policy",
            "log_experiment", "list_experiments", "run_quick_verify",
            "get_improve_request", "respond_improve_request", "get_brief",
        })

    def test_unknown_tool_is_an_error(self):
        response = self.server.handle({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "rent_gpu_pod", "arguments": {}},
        })
        self.assertEqual(response["error"]["code"], -32603)

    def test_serve_speaks_newline_delimited_jsonrpc(self):
        stdin = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {}}) + "\n"
            + json.dumps({"jsonrpc": "2.0",
                          "method": "notifications/initialized"}) + "\n"
            + "garbage\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2,
                          "method": "tools/list"}) + "\n"
        )
        stdout = io.StringIO()
        code = mcp_server.serve(self.root, stdin=stdin, stdout=stdout)
        self.assertEqual(code, 0)
        responses = [json.loads(line) for line in
                     stdout.getvalue().strip().splitlines()]
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0]["id"], 1)
        self.assertEqual(responses[1]["error"]["code"], -32700)
        self.assertIn("tools", responses[2]["result"])


class RequestActionTests(ServerHarness):
    def test_request_queues_a_pending_approval_and_executes_nothing(self):
        payload = self.payload(self.call("request_action", {
            "action": "gpu_evaluation",
            "reason": "screen the burst-family arm",
            "estimated_hours": 1.5,
        }))
        request = payload["request"]
        self.assertEqual(request["status"], "pending")
        self.assertIn(request["id"], payload["note"])
        stored = read_json(self.root / "runs/approvals.json")["requests"]
        self.assertEqual([r["id"] for r in stored], [request["id"]])
        self.assertEqual(stored[0]["context"]["candidate_path"],
                         "generators/candidate")
        self.assertEqual(stored[0]["context"]["estimated_hours"], 1.5)
        self.assertEqual(stored[0]["context"]["requested_via"], "mcp")

    def test_duplicate_request_returns_the_same_pending_entry(self):
        first = self.payload(self.call("request_action", {
            "action": "gpu_evaluation", "reason": "same ask"}))
        second = self.payload(self.call("request_action", {
            "action": "gpu_evaluation", "reason": "same ask"}))
        self.assertEqual(first["request"]["id"], second["request"]["id"])
        stored = read_json(self.root / "runs/approvals.json")["requests"]
        self.assertEqual(len(stored), 1)

    def test_missing_reason_is_a_tool_error(self):
        result = self.call("request_action",
                           {"action": "gpu_evaluation", "reason": "  "})
        self.assertTrue(result["isError"])
        self.assertIn("reason", result["content"][0]["text"])

    def test_unknown_action_is_a_tool_error(self):
        result = self.call("request_action",
                           {"action": "rent_pod", "reason": "x"})
        self.assertTrue(result["isError"])

    def test_candidate_path_may_not_escape_generators(self):
        result = self.call("request_action", {
            "action": "gpu_evaluation", "reason": "x",
            "candidate_path": "../ops",
        })
        self.assertTrue(result["isError"])
        self.assertIn("generators/", result["content"][0]["text"])


class LedgerToolTests(ServerHarness):
    def test_log_then_list_experiments(self):
        logged = self.payload(self.call("log_experiment", {
            "hypothesis": "swap family 7", "seeds": [0, 1, 2],
            "snapshots": ["2026-07-16"], "status": "planned",
        }))
        listed = self.payload(self.call("list_experiments", {}))
        self.assertEqual(listed, [logged])

    def test_bad_status_is_a_tool_error(self):
        result = self.call("log_experiment",
                           {"hypothesis": "x", "status": "amazing"})
        self.assertTrue(result["isError"])


class ReadToolTests(ServerHarness):
    def test_status_and_state_work_before_the_controller_ran(self):
        summary = self.payload(self.call("miner_status"))
        self.assertFalse(summary["initialized"])
        self.assertEqual(self.payload(self.call("get_state")), {})
        self.assertEqual(self.payload(self.call("get_latest_receipt")), {})
        self.assertEqual(self.payload(self.call("list_approvals")), [])

    def test_get_policy_without_a_policy_file(self):
        payload = self.payload(self.call("get_policy"))
        self.assertIsNone(payload["policy"])
        self.assertIn("--autonomous-actions", payload["note"])

    def test_get_policy_with_a_policy_file_reports_caps_and_usage(self):
        (self.root / "policy.toml").write_text(
            "[actions.gpu_evaluation]\nautonomous = true\n"
            "max_runs_per_day = 4\n", encoding="utf-8")
        payload = self.payload(self.call("get_policy"))
        rule = payload["policy"]["actions"]["gpu_evaluation"]
        self.assertTrue(rule["autonomous"])
        self.assertEqual(rule["max_runs_per_day"], 4)
        self.assertEqual(payload["usage_last_24h"]["gpu_evaluation"],
                         {"runs": 0, "hours": 0.0})

    def test_get_brief_carries_the_hard_limits(self):
        brief = self.payload(self.call("get_brief"))
        joined = " ".join(brief["hard_limits"])
        self.assertIn("Never rent a pod", joined)
        self.assertIn("noise floor", joined)
        self.assertIn("notes/METHOD.md", brief["canonical_docs"])
        self.assertIn("status", brief)


class ImproveHandshakeTests(ServerHarness):
    def test_no_pending_request(self):
        payload = self.payload(self.call("get_improve_request"))
        self.assertFalse(payload["pending"])
        result = self.call("respond_improve_request", {"status": "completed"})
        self.assertTrue(result["isError"])

    def test_respond_matches_the_pending_request_id(self):
        request_path = self.root / "runs/improve-request.json"
        request_path.parent.mkdir(parents=True)
        request_path.write_text(json.dumps(
            {"id": "1722500000-42", "status": "pending", "prompt": "improve"}),
            encoding="utf-8")
        pending = self.payload(self.call("get_improve_request"))
        self.assertTrue(pending["pending"])
        payload = self.payload(self.call("respond_improve_request", {
            "status": "completed", "detail": "swapped family 7"}))
        response = read_json(self.root / "runs/improve-response.json")
        self.assertEqual(response, payload["response"])
        self.assertEqual(response["id"], "1722500000-42")
        # The blocked hook accepts exactly these statuses; stay in sync.
        self.assertIn(response["status"], TERMINAL_STATUSES)

    def test_respond_rejects_unknown_status(self):
        request_path = self.root / "runs/improve-request.json"
        request_path.parent.mkdir(parents=True)
        request_path.write_text(json.dumps({"id": "1-2"}), encoding="utf-8")
        result = self.call("respond_improve_request", {"status": "done"})
        self.assertTrue(result["isError"])
