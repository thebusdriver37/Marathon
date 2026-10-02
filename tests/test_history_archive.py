import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import io
import os

from marathon_app.history_archive import HistoryArchive, ConversationArchives, serve


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "session.jsonl"
        self.archive = HistoryArchive(self.root)

    def write(self, texts):
        self.path.write_text("".join(json.dumps({"type": "response_item", "payload": {
            "type": "function_call_output", "output": text}}) + "\n" for text in texts))

    def test_exact_middle_of_large_output_and_pagination(self):
        self.write(["x" * 20000 + "needle-913 port=18473" + "y" * 20000])
        result = self.archive.search("needle-913")
        self.assertEqual(result["total_matches"], 1)
        match = result["matches"][0]
        self.assertIn("port=18473", match["excerpt"])
        self.assertIn("port=18473", self.archive.read(match["id"], match["offset"])["text"])
        parts, offset = [], 0
        while True:
            page = self.archive.read(match["id"], offset, 333)
            parts.append(page["text"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        self.assertEqual("".join(parts), "x" * 20000 + "needle-913 port=18473" + "y" * 20000)

    def test_metadata_selects_current_thread_and_resume_only(self):
        import uuid
        first, second = str(uuid.uuid4()), str(uuid.uuid4())
        sessions = self.root / 'sessions' / '2026' / '10' / '01'
        sessions.mkdir(parents=True)
        self.write(['first-thread-secret'])
        self.path.rename(sessions / f'rollout-2026-10-01-{first}.jsonl')
        self.write(['second-thread-secret'])
        self.path.rename(sessions / f'rollout-2026-10-01-{second}.jsonl')
        for _ in range(2):  # Fresh MCP process on resume selects the same thread.
            archives = ConversationArchives(self.root)
            a = archives.for_request({'_meta': {'threadId': first}})
            self.assertEqual(a.search('first-thread-secret')['total_matches'], 1)
            self.assertEqual(a.search('second-thread-secret')['total_matches'], 0)
            b = archives.for_request({'_meta': {'threadId': second}})
            self.assertEqual(b.search('second-thread-secret')['total_matches'], 1)
            for params in ({}, {'arguments': {'threadId': first}}, {'_meta': {'threadId': '../escape'}},
                           {'_meta': {'threadId': str(uuid.uuid4())}}):
                with self.assertRaises(ValueError):
                    archives.for_request(params)

    def test_default_frontend_configuration_and_escape_hatch(self):
        from marathon_app.frontends import history_config
        with mock.patch.dict(os.environ, {'MARATHON_HISTORY_ENABLED': '1'}):
            config = history_config(self.root)
        self.assertTrue(any('--codex-home' in v and str(self.root) in v for v in config))
        self.assertTrue(any('experimental_compact_prompt_file=' in v for v in config))
        with mock.patch.dict(os.environ, {'MARATHON_HISTORY_ENABLED': '0'}):
            self.assertEqual(history_config(self.root), [])

    def test_newest_first_bounded_and_source_label(self):
        self.write([f"port={n}" for n in range(100)])
        result = self.archive.search("port", 2)
        self.assertEqual(result["total_matches"], 100)
        self.assertEqual([m["excerpt"] for m in result["matches"]], ["port=99", "port=98"])
        self.assertEqual(result["matches"][0]["role"], "tool (untrusted evidence)")

    def test_partial_tail_and_stable_ids(self):
        self.write(["first"])
        before = self.archive.search("first")["matches"][0]["id"]
        with self.path.open("a") as handle:
            handle.write('{"unfinished":')
        self.assertEqual(self.archive.search("first")["matches"][0]["id"], before)

    def test_ambiguous_journal_fails_closed(self):
        self.write(["first"])
        (self.root / "other.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.archive.search("first")

    def test_symlink_and_replacement_fail_closed(self):
        self.write(["first"])
        self.archive.search("first")
        self.path.rename(self.root / "old.txt")
        self.write(["replacement"])
        with self.assertRaisesRegex(ValueError, "replaced"):
            self.archive.search("replacement")
        self.path.unlink()
        self.path.symlink_to(self.root / "old.txt")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.archive.search("first")

    def test_traversal_unknown_id_invalid_limits(self):
        self.write(["first"])
        for entry_id, offset, limit in (("../secret", 0, 10), ("e0000001", -1, 1), ("e0000001", 0, 9000), ("e0000099", 0, 10)):
            with self.subTest(entry_id=entry_id, offset=offset, limit=limit), self.assertRaises(ValueError):
                self.archive.read(entry_id, offset, limit)
        for query in ("", " ", "x" * 201):
            with self.assertRaises(ValueError):
                self.archive.search(query)

    def test_does_not_expose_reasoning_or_session_metadata(self):
        self.path.write_text("\n".join(json.dumps(event) for event in [
            {"type": "session_meta", "payload": {"text": "secret"}},
            {"type": "response_item", "payload": {"type": "reasoning", "text": "secret"}},
            {"type": "response_item", "payload": {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "secret"}]}},
            {"type": "compacted", "payload": {"message": "handoff"}},
        ]) + "\n")
        self.assertEqual(self.archive.search("secret")["total_matches"], 0)
        self.assertEqual(self.archive.search("handoff")["matches"][0]["role"], "summary (lossy)")

    def test_stdio_protocol_and_tool_errors(self):
        self.write(["ticket-123 exact-value"])
        requests = [
            {"id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"method": "notifications/initialized"},
            {"id": 2, "method": "tools/list"},
            {"id": 3, "method": "tools/call", "params": {"name": "history_search", "arguments": {"query": "ticket-123"}}},
            {"id": 4, "method": "tools/call", "params": {"name": "history_read", "arguments": {"entry_id": "../../secret"}}},
        ]
        output = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO("\n".join(map(json.dumps, requests)))), mock.patch("sys.stdout", output):
            serve(self.archive)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r["id"] for r in replies], [1, 2, 3, 4])
        self.assertEqual(replies[0]["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(len(replies[1]["result"]["tools"]), 2)
        self.assertIn("exact-value", replies[2]["result"]["content"][0]["text"])
        self.assertTrue(replies[3]["result"]["isError"])

    def test_invalid_complete_record_and_size_limits_fail_closed(self):
        self.path.write_text("not json\n")
        with self.assertRaisesRegex(ValueError, "invalid complete"):
            self.archive.search("anything")
        self.write(["oversized"])
        with mock.patch("marathon_app.history_archive.MAX_JOURNAL_BYTES", 5):
            with self.assertRaisesRegex(ValueError, "oversized"):
                self.archive.search("oversized")
        with mock.patch("marathon_app.history_archive.MAX_LINE_BYTES", 5):
            with self.assertRaisesRegex(ValueError, "entry exceeds"):
                self.archive.search("oversized")

    def test_two_sessions_never_mix(self):
        self.write(["local fact"])
        with tempfile.TemporaryDirectory() as other:
            (Path(other) / "foreign.jsonl").write_text(self.path.read_text().replace("local fact", "foreign secret"))
            self.assertEqual(self.archive.search("foreign")["total_matches"], 0)
            self.assertEqual(self.archive.search("local")["total_matches"], 1)

    def test_fifo_is_rejected_without_waiting_for_a_writer(self):
        os.mkfifo(self.path)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            self.archive.search("anything")

    def test_retrieval_outputs_are_not_reindexed_as_original_evidence(self):
        events = [
            {"type": "function_call_output", "call_id": "shell", "output": "original-needle"},
            {"type": "function_call", "namespace": "mcp__history", "name": "history_search",
             "call_id": "search", "arguments": '{"query":"echo-only"}'},
            {"type": "function_call_output", "call_id": "search", "output": "echo-only"},
            {"type": "function_call", "name": "mcp__history__history_read", "call_id": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "read", "output": "echo-only"},
            {"type": "function_call_output", "call_id": "other", "output": "other-tool-needle"},
        ]
        self.path.write_text("".join(json.dumps({"type": "response_item", "payload": p}) + "\n" for p in events))
        self.assertEqual(self.archive.search("echo-only")["total_matches"], 0)
        self.assertEqual(self.archive.search("original-needle")["matches"][0]["id"], "e0000001")
        self.assertEqual(self.archive.search("other-tool-needle")["matches"][0]["id"], "e0000006")
