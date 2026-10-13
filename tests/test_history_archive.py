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

    def append_events(self, *events):
        with self.path.open("a") as handle:
            for event in events:
                handle.write(json.dumps(event) + "\n")

    @staticmethod
    def message(text, role="user", **extra):
        return {"type": "response_item", "payload": {
            "type": "message", "role": role,
            "content": [{"type": "input_text", "text": text}]}, **extra}

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

    def test_invalid_complete_record_fails_closed(self):
        self.path.write_text("not json\n")
        with self.assertRaisesRegex(ValueError, "invalid complete"):
            self.archive.search("anything")

    def test_large_journal_remains_searchable_without_caching_transcript(self):
        self.write(["early-needle"])
        # Real file beyond the old cap; ignored reasoning records must not be
        # retained in memory or parsed again during later paginated reads.
        row = json.dumps({"type": "response_item", "payload": {
            "type": "reasoning", "text": "x" * (1024 * 1024)}}) + "\n"
        with self.path.open("a") as handle:
            for _ in range(65):
                handle.write(row)
        self.append_events(self.message("late-needle"))
        self.assertGreater(self.path.stat().st_size, 64 * 1024 * 1024)
        self.assertEqual(self.archive.search("needle")["total_matches"], 2)
        with mock.patch.object(self.archive, "_decode", wraps=self.archive._decode) as decode:
            self.assertEqual(self.archive.read("e0000001")["text"], "early-needle")
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(len(self.archive._index), 2)

    def test_oversized_record_is_skipped_with_notice_and_stable_ids(self):
        self.write(["early-needle"])
        self.append_events({"type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": "A" * (4 * 1024 * 1024)}]}},
            self.message("late-needle"))
        result = self.archive.search("needle")
        self.assertEqual([m["id"] for m in result["matches"]], ["e0000003", "e0000001"])
        self.assertEqual(result["coverage"]["skipped_oversized_records"], 1)
        self.assertEqual(self.archive.read("e0000003")["text"], "late-needle")
        with self.assertRaisesRegex(ValueError, "not found"):
            self.archive.read("e0000002")

    def test_partial_oversized_append_is_counted_once_when_completed(self):
        self.write(["early-needle"])
        with self.path.open("a") as handle:
            handle.write('{"type":"response_item","payload":{"type":"reasoning","text":"' + "x" * 600)
        with mock.patch("marathon_app.history_archive.MAX_LINE_BYTES", 512):
            self.assertEqual(self.archive.search("needle")["coverage"]["skipped_oversized_records"], 0)
            with self.path.open("a") as handle:
                handle.write('"}}\n')
            self.append_events(self.message("late-needle"))
            for _ in range(2):
                result = self.archive.search("needle")
                self.assertEqual(result["total_matches"], 2)
                self.assertEqual(result["coverage"]["skipped_oversized_records"], 1)

    def test_rollback_invalidates_search_and_cached_read_ids_across_compaction(self):
        self.append_events(self.message("keep disabled"), self.message("enable deployment"),
                           self.message("deployment approved", "assistant"),
                           {"type": "compacted", "payload": {"message": "deployment approved"}})
        removed = self.archive.search("enable")["matches"][0]["id"]
        self.append_events({"type": "event_msg", "payload": {
            "type": "thread_rolled_back", "num_turns": 1}})
        with self.assertRaisesRegex(ValueError, "not found"):
            self.archive.read(removed)
        self.assertEqual(self.archive.search("deployment")["total_matches"], 0)
        self.append_events(self.message("test only"))
        for archive in (self.archive, HistoryArchive(self.root)):
            self.assertEqual(archive.search("keep disabled")["total_matches"], 1)
            self.assertEqual(archive.search("test only")["total_matches"], 1)
            self.assertEqual(archive.search("deployment")["total_matches"], 0)
        self.append_events({"type": "event_msg", "payload": {
            "type": "thread_rolled_back", "num_turns": 99}})
        self.assertEqual(list(self.archive.entries()), [])

    def test_native_turn_boundaries_do_not_double_count_user_events_or_steering(self):
        self.append_events(
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "one"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "keep"}},
            self.message("keep"),
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "two"}},
            self.message("remove first"), self.message("remove steering"),
            {"type": "compacted", "payload": {"message": "remove summary"}},
            {"type": "event_msg", "payload": {"type": "thread_rolled_back", "num_turns": 1}})
        self.assertEqual(self.archive.search("keep")["total_matches"], 1)
        self.assertEqual(self.archive.search("remove")["total_matches"], 0)

    def test_legacy_user_events_after_response_records_do_not_double_count(self):
        self.append_events(self.message("keep"),
                           {"type": "event_msg", "payload": {"type": "user_message", "message": "keep"}},
                           self.message("remove"),
                           {"type": "event_msg", "payload": {"type": "user_message", "message": "remove"}},
                           {"type": "event_msg", "payload": {"type": "thread_rolled_back", "num_turns": 1}})
        self.assertEqual(self.archive.search("keep")["total_matches"], 1)
        self.assertEqual(self.archive.search("remove")["total_matches"], 0)

    def test_legacy_scaffolding_and_synthetic_summaries_are_not_user_turns(self):
        self.append_events(self.message("keep"), self.message("remove"),
                           self.message("Another language model started to solve this problem. remove"),
                           self.message("<environment_context>remove</environment_context>"),
                           {"type": "event_msg", "payload": {"type": "thread_rolled_back", "num_turns": 1}})
        self.assertEqual(self.archive.search("keep")["total_matches"], 1)
        self.assertEqual(self.archive.search("remove")["total_matches"], 0)

    def test_opaque_record_before_rollback_withholds_uncertain_history(self):
        self.append_events(self.message("possibly withdrawn"), self.message("x" * 700),
                           {"type": "event_msg", "payload": {"type": "thread_rolled_back", "num_turns": 1}},
                           self.message("new active request"))
        with mock.patch("marathon_app.history_archive.MAX_LINE_BYTES", 512):
            result = self.archive.search("withdrawn")
            self.assertEqual(result["total_matches"], 0)
            self.assertTrue(result["coverage"]["pre_rollback_history_unavailable"])
            self.assertEqual(self.archive.search("active")["total_matches"], 1)

    def test_summary_output_is_not_original_evidence_or_duplicate_summary(self):
        self.append_events(self.message("canonical-needle", "assistant", metadata={"compaction_output": True}),
                           {"type": "compacted", "payload": {"message": "canonical-needle"}},
                           self.message("uncommitted-needle", "assistant", metadata={"compaction_output": True}))
        result = self.archive.search("needle")
        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["matches"][0]["role"], "summary (lossy)")
        self.assertEqual(self.archive.read("e0000002")["role"], "summary (lossy)")
        for entry_id in ("e0000001", "e0000003"):
            with self.assertRaisesRegex(ValueError, "not found"):
                self.archive.read(entry_id)

    def test_truncation_of_indexed_journal_is_rejected(self):
        self.write(["original evidence"])
        self.archive.search("original")
        self.path.write_text("")
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.archive.read("e0000001")

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
