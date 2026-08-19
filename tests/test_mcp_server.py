import io
import json
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from miner import issues, mcp_server
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
            "list_heat_entrants", "fetch_generator",
            "list_approvals", "request_action", "get_policy",
            "log_experiment", "list_experiments",
            "report_issue", "list_issues", "run_quick_verify", "screen_candidate",
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


class IssueToolTests(ServerHarness):
    def test_report_then_list_issues(self):
        filed = self.payload(self.call("report_issue", {
            "kind": "bug",
            "title": "status crashes without a candidate dir",
            "detail": "rm -r generators/candidate; python -m miner.status",
            "area": "miner/status.py",
        }))
        self.assertEqual(filed["entry"]["status"], "open")
        self.assertIn(filed["entry"]["id"], filed["note"])
        listed = self.payload(self.call("list_issues", {}))
        self.assertEqual(listed, [filed["entry"]])

    def test_open_only_applies_triage(self):
        filed = self.payload(self.call("report_issue", {
            "kind": "feature", "title": "expose pool sync as a tool"}))
        issues.report_entry(
            self.root, kind="feature", title="expose pool sync as a tool",
            status="resolved", supersedes=filed["entry"]["id"])
        self.assertEqual(
            self.payload(self.call("list_issues", {"open_only": True})), [])
        self.assertEqual(
            len(self.payload(self.call("list_issues", {}))), 2)

    def test_open_issues_reach_miner_status(self):
        self.payload(self.call("report_issue", {
            "kind": "bug", "title": "visible to the operator"}))
        summary = self.payload(self.call("miner_status"))
        self.assertEqual(summary["open_issues"][0]["title"],
                         "visible to the operator")

    def test_missing_title_is_a_tool_error(self):
        result = self.call("report_issue", {"kind": "bug", "title": "  "})
        self.assertTrue(result["isError"])
        self.assertIn("title", result["content"][0]["text"])

    def test_unknown_kind_is_a_tool_error(self):
        result = self.call("report_issue", {"kind": "rant", "title": "x"})
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


class HeatEntrantTests(ServerHarness):
    def _seed_receipt(self, receipt: dict) -> None:
        state = self.root / "runs/controller-state.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"last_receipt": receipt}),
                         encoding="utf-8")

    def test_no_receipt_yet(self):
        payload = self.payload(self.call("list_heat_entrants"))
        self.assertIsNone(payload["round_id"])
        self.assertIn("run the controller once", payload["note"])

    def test_full_entrant_lists_are_returned(self):
        self._seed_receipt({
            "round_id": "round-9",
            "summary": {"king_gen_ref": "hippius://king"},
            "participants": [{"hotkey": "us"}, {"hotkey": "them"}],
            "heat": {"entrants": [
                {"hotkey": "them", "rank": 1, "gen_ref": "hippius://them"},
                {"hotkey": "us", "rank": 2},
            ]},
        })
        payload = self.payload(self.call("list_heat_entrants"))
        self.assertEqual(payload["round_id"], "round-9")
        self.assertEqual(payload["king_gen_ref"], "hippius://king")
        self.assertEqual(len(payload["entrants"]), 2)
        self.assertEqual(len(payload["participants"]), 2)
        self.assertNotIn("note", payload)

    def test_pre_feature_receipt_gets_an_explanatory_note(self):
        self._seed_receipt({
            "round_id": "round-8",
            "summary": {"king_gen_ref": "hippius://king"},
            "miner_heat": {"rank": 3},
        })
        payload = self.payload(self.call("list_heat_entrants"))
        self.assertEqual(payload["entrants"], [])
        self.assertIn("before entrant lists were kept", payload["note"])


class FetchGeneratorTests(ServerHarness):
    def test_fetch_invokes_cascade_with_a_guarded_destination(self):
        with patch.object(mcp_server.subprocess, "run",
                          return_value=CompletedProcess(
                              [], 0, stdout="fetched", stderr="")) as run:
            payload = self.payload(self.call("fetch_generator", {
                "ref": "hippius://heats/round-9/entrant-a",
            }))
        argv = run.call_args.args[0]
        self.assertEqual(argv[1], "fetch")
        self.assertEqual(argv[2], "hippius://heats/round-9/entrant-a")
        self.assertIn("--chain-toml", argv)
        self.assertIn("--network", argv)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["path"],
                         str(self.root / "generators/entrant-a"))

    def test_out_name_may_not_traverse(self):
        result = self.call("fetch_generator",
                           {"ref": "hippius://x", "out_name": "../ops"})
        self.assertTrue(result["isError"])
        self.assertIn("out_name", result["content"][0]["text"])

    def test_existing_destination_needs_overwrite(self):
        occupied = self.root / "generators/king-control"
        occupied.mkdir(parents=True)
        (occupied / "generator.py").write_text("x", encoding="utf-8")
        result = self.call("fetch_generator", {
            "ref": "hippius://king", "out_name": "king-control"})
        self.assertTrue(result["isError"])
        self.assertIn("overwrite", result["content"][0]["text"])
        with patch.object(mcp_server.subprocess, "run",
                          return_value=CompletedProcess([], 0, "", "")):
            payload = self.payload(self.call("fetch_generator", {
                "ref": "hippius://king", "out_name": "king-control",
                "overwrite": True}))
        self.assertTrue(payload["ok"])

    def test_empty_ref_is_a_tool_error(self):
        result = self.call("fetch_generator", {"ref": "  "})
        self.assertTrue(result["isError"])

    def test_failed_fetch_reports_stderr(self):
        with patch.object(mcp_server.subprocess, "run",
                          return_value=CompletedProcess(
                              [], 1, stdout="", stderr="permission denied")):
            payload = self.payload(self.call("fetch_generator", {
                "ref": "hippius://king", "out_name": "study"}))
        self.assertFalse(payload["ok"])
        self.assertIn("permission denied", payload["stderr"])


class ScreenCandidateTests(ServerHarness):
    def setUp(self):
        super().setUp()
        (self.root / "generators/king-control").mkdir(parents=True)

    def run_screen(self, arguments: dict, *, verdict: str = "measurable",
                   code: int = 0):
        """Patch out the subprocess and drop the report it would have written."""
        def fake_run(argv, **kwargs):
            destination = Path(argv[argv.index("--json") + 1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"verdict": verdict}), encoding="utf-8")
            self.argv = argv
            return CompletedProcess(argv, code, stdout="ran", stderr="")

        with patch.object(mcp_server.subprocess, "run", side_effect=fake_run):
            return self.call("screen_candidate", arguments)

    def test_screen_runs_the_module_and_returns_the_report(self):
        payload = self.payload(self.run_screen(
            {"n_series": 64, "claims": ["regime+"], "seed": 3}))
        self.assertEqual(payload["verdict"], "measurable")
        self.assertEqual(payload["report"]["verdict"], "measurable")
        self.assertEqual(payload["report_path"],
                         "runs/screen/candidate--vs--king-control.json")
        self.assertIn("-m", self.argv)
        self.assertIn("miner.screen", self.argv)
        self.assertEqual(self.argv[self.argv.index("--n-series") + 1], "64")
        self.assertEqual(self.argv[self.argv.index("--seed") + 1], "3")
        self.assertEqual(self.argv[self.argv.index("--claim") + 1], "regime+")

    def test_undosed_is_reported_not_raised(self):
        payload = self.payload(self.run_screen({}, verdict="undosed", code=3))
        self.assertEqual(payload["verdict"], "undosed")
        self.assertEqual(payload["exit_code"], 3)

    def test_missing_king_names_the_fetch_that_fixes_it(self):
        result = self.call("screen_candidate", {"king_path": "generators/absent"})
        self.assertTrue(result["isError"])
        self.assertIn("fetch_generator", result["content"][0]["text"])

    def test_paths_may_not_escape_generators(self):
        result = self.call("screen_candidate", {"candidate_path": "../ops"})
        self.assertTrue(result["isError"])
        self.assertIn("generators/", result["content"][0]["text"])


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
