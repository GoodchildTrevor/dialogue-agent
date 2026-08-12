"""Tests for pure helper functions in app.graph.nodes.orchestrator.

Covers:
    _uploaded_files_reminder, _safe_json_loads, _normalize_tool_calls,
    _sanitize_llm_output, _sanitize_tool_content,
    _format_intermediate_steps, _build_unknown_tool_error_step
"""

import json
from app.graph.nodes.orchestrator import (
    _uploaded_files_reminder,
    _safe_json_loads,
    _normalize_tool_calls,
    _sanitize_llm_output,
    _sanitize_tool_content,
    _format_intermediate_steps,
    _build_unknown_tool_error_step,
)


# ── _uploaded_files_reminder ────────────────────────────────────────────────

class TestUploadedFilesReminder:
    def test_empty_list_returns_empty_string(self):
        assert _uploaded_files_reminder([]) == ""

    def test_falsy_input_returns_empty_string(self):
        # Function checks `if not uploaded_files` so any falsy value works
        assert _uploaded_files_reminder([]) == ""

    def test_inline_only_file_embeds_text(self):
        files = [{"filename": "notes.txt", "inline_text": "Hello world"}]
        result = _uploaded_files_reminder(files)
        assert "[UPLOADED FILES — ACTION REQUIRED]" in result
        assert "--- FILE: \"notes.txt\" ---" in result
        assert "Hello world" in result
        assert "--- END OF FILE: \"notes.txt\" ---" in result
        assert "Do NOT call any tool" in result
        assert "[END UPLOADED FILES]" in result

    def test_qdrant_only_file_includes_search_instructions(self):
        files = [{"filename": "doc.pdf", "file_id": "f-123"}]
        result = _uploaded_files_reminder(files)
        assert "indexed in the vector database" in result
        assert "document_searcher" in result
        assert 'file_id: "f-123"' in result
        assert "[END UPLOADED FILES]" in result

    def test_mixed_inline_and_qdrant_files(self):
        files = [
            {"filename": "small.txt", "inline_text": "Inline content"},
            {"filename": "big.pdf", "file_id": "f-456"},
        ]
        result = _uploaded_files_reminder(files)
        assert "Inline content" in result
        assert "indexed in the vector database" in result
        assert 'file_id: "f-456"' in result
        assert "[END UPLOADED FILES]" in result

    def test_inline_file_with_none_inline_text_ignored(self):
        files = [{"filename": "empty.txt", "inline_text": None}]
        result = _uploaded_files_reminder(files)
        assert "--- FILE:" not in result
        assert "indexed" in result  # treated as qdrant

    def test_inline_file_with_empty_string_inline_text_ignored(self):
        files = [{"filename": "empty.txt", "inline_text": ""}]
        result = _uploaded_files_reminder(files)
        assert "--- FILE:" not in result


# ── _safe_json_loads ────────────────────────────────────────────────────────

class TestSafeJsonLoads:
    def test_valid_json_string_parsed(self):
        assert _safe_json_loads('{"a": 1}') == {"a": 1}

    def test_valid_json_list_parsed(self):
        assert _safe_json_loads("[1, 2, 3]") == [1, 2, 3]

    def test_invalid_json_returns_original(self):
        result = _safe_json_loads("not-valid-json")
        assert result == "not-valid-json"

    def test_non_string_passthrough_int(self):
        assert _safe_json_loads(42) == 42

    def test_non_string_passthrough_dict(self):
        original = {"key": "value"}
        assert _safe_json_loads(original) is original

    def test_non_string_passthrough_none(self):
        assert _safe_json_loads(None) is None

    def test_nested_valid_json_parsed(self):
        data = {"outer": {"inner": [1, 2]}}
        assert _safe_json_loads(json.dumps(data)) == data


# ── _normalize_tool_calls ───────────────────────────────────────────────────

class TestNormalizeToolCalls:
    def test_dict_with_name_key(self):
        raw = {"name": "search", "arguments": {"q": "hello"}}
        result = _normalize_tool_calls(raw)
        assert len(result) == 1
        assert result[0] == {"tool": "search", "arguments": {"q": "hello"}}

    def test_dict_with_tool_key(self):
        raw = {"tool": "search", "arguments": {"q": "hi"}}
        result = _normalize_tool_calls(raw)
        assert len(result) == 1
        assert result[0] == {"tool": "search", "arguments": {"q": "hi"}}

    def test_dict_args_as_json_string_parsed(self):
        raw = {"name": "search", "arguments": '{"q": "parsed"}'}
        result = _normalize_tool_calls(raw)
        assert result[0]["arguments"] == {"q": "parsed"}

    def test_dict_args_non_dict_string_fallback_to_empty(self):
        raw = {"name": "search", "arguments": "[1, 2]"}
        result = _normalize_tool_calls(raw)
        assert result[0]["arguments"] == {}

    def test_dict_with_name_and_tool_prefers_name(self):
        raw = {"name": "first", "tool": "second", "arguments": {}}
        result = _normalize_tool_calls(raw)
        assert result[0]["tool"] == "first"

    def test_dict_with_args_and_arguments_prefers_args(self):
        raw = {"name": "x", "args": {"a": 1}, "arguments": {"b": 2}}
        result = _normalize_tool_calls(raw)
        assert result[0]["arguments"] == {"a": 1}

    def test_dict_missing_name_returns_empty_list(self):
        assert _normalize_tool_calls({"arguments": {}}) == []

    def test_list_of_dicts(self):
        raw = [
            {"tool": "search", "arguments": {"q": "a"}},
            {"name": "calc", "arguments": {"expr": "1+1"}},
        ]
        result = _normalize_tool_calls(raw)
        assert len(result) == 2
        assert result[0] == {"tool": "search", "arguments": {"q": "a"}}
        assert result[1] == {"tool": "calc", "arguments": {"expr": "1+1"}}

    def test_list_of_strings(self):
        raw = ["search", "calc"]
        result = _normalize_tool_calls(raw)
        assert len(result) == 2
        assert result[0] == {"tool": "search", "arguments": {}}
        assert result[1] == {"tool": "calc", "arguments": {}}

    def test_list_mixed_dicts_and_strings(self):
        raw = [{"tool": "search", "arguments": {}}, "calc"]
        result = _normalize_tool_calls(raw)
        assert len(result) == 2

    def test_list_skips_non_dict_non_string_items(self):
        raw = [{"tool": "search", "arguments": {}}, 42, None, {"name": "ok"}]
        result = _normalize_tool_calls(raw)
        assert len(result) == 2

    def test_empty_list_returns_empty_list(self):
        assert _normalize_tool_calls([]) == []

    def test_non_list_non_dict_returns_empty_list(self):
        assert _normalize_tool_calls("just a string") == []
        assert _normalize_tool_calls(42) == []
        assert _normalize_tool_calls(None) == []

    def test_json_string_single_dict_parsed(self):
        raw = json.dumps({"tool": "search", "arguments": {"q": "x"}})
        result = _normalize_tool_calls(raw)
        assert len(result) == 1
        assert result[0]["tool"] == "search"

    def test_json_string_list_parsed(self):
        raw = json.dumps([{"tool": "a", "arguments": {}}, {"tool": "b", "arguments": {}}])
        result = _normalize_tool_calls(raw)
        assert len(result) == 2


# ── _sanitize_llm_output ────────────────────────────────────────────────────

class TestSanitizeLlmOutput:
    def test_respond_action(self):
        parsed = {"action": "respond", "answer": "Hello"}
        result = _sanitize_llm_output(parsed, "fallback")
        assert result == {
            "action": "respond",
            "answer": "Hello",
            "task": "fallback",
            "tool_calls": [],
        }

    def test_tools_action_with_tool_calls(self):
        parsed = {
            "action": "tools",
            "tool_calls": [{"tool": "search", "arguments": {"q": "x"}}],
        }
        result = _sanitize_llm_output(parsed, "fallback")
        assert result["action"] == "tools"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "search"

    def test_escalate_action(self):
        parsed = {"action": "escalate", "task": "do something complex"}
        result = _sanitize_llm_output(parsed, "fallback")
        assert result["action"] == "escalate"
        assert result["task"] == "do something complex"

    def test_function_legacy_format_normalizes_to_tools(self):
        parsed = {
            "action": "function",
            "function": {"name": "search", "arguments": {"q": "x"}},
        }
        result = _sanitize_llm_output(parsed, "fallback")
        assert result["action"] == "tools"
        assert len(result["tool_calls"]) == 1

    def test_function_format_without_tool_calls_escalates(self):
        parsed = {"action": "function", "function": {}}
        result = _sanitize_llm_output(parsed, "fallback")
        assert result["action"] == "escalate"

    def test_invalid_parsed_input_returns_escalate(self):
        result = _sanitize_llm_output("not a dict", "fallback task")
        assert result == {
            "action": "escalate",
            "task": "fallback task",
            "tool_calls": [],
            "answer": "",
        }

    def test_none_parsed_input_returns_escalate(self):
        result = _sanitize_llm_output(None, "fallback")
        assert result["action"] == "escalate"
        assert result["task"] == "fallback"

    def test_unknown_action_with_tool_calls_becomes_tools(self):
        parsed = {"action": "unknown", "tool_calls": [{"tool": "search", "arguments": {}}]}
        result = _sanitize_llm_output(parsed, "fallback")
        assert result["action"] == "tools"

    def test_unknown_action_without_tool_calls_becomes_escalate(self):
        parsed = {"action": "unknown"}
        result = _sanitize_llm_output(parsed, "fallback")
        assert result["action"] == "escalate"

    def test_none_action_field_becomes_empty_string_then_escalate(self):
        parsed = {"answer": "test", "tool_calls": []}
        result = _sanitize_llm_output(parsed, "fallback")
        assert result["action"] == "escalate"

    def test_answer_always_string(self):
        parsed = {"action": "respond", "answer": 123}
        result = _sanitize_llm_output(parsed, "fallback")
        assert result["answer"] == "123"

    def test_empty_dict_returns_escalate_with_fallback_task(self):
        result = _sanitize_llm_output({}, "my task")
        assert result["action"] == "escalate"
        assert result["task"] == "my task"


# ── _sanitize_tool_content ──────────────────────────────────────────────────

class TestSanitizeToolContent:
    def test_image_type_single_quotes_detected(self):
        content = "some text type='image' more text"
        result = _sanitize_tool_content(content, "gen")
        assert "[Image generated successfully by 'gen'" in result
        assert "type='image'" not in result

    def test_image_type_double_quotes_detected(self):
        content = 'some text type="image" more text'
        result = _sanitize_tool_content(content, "gen")
        assert "[Image generated successfully by 'gen'" in result

    def test_data_iVBOR_single_quotes_detected(self):
        content = "data='iVBORw0KGgoAAAANSUhEUg'"
        result = _sanitize_tool_content(content, "chart")
        assert "[Image generated successfully by 'chart'" in result

    def test_data_iVBOR_double_quotes_detected(self):
        content = 'data="iVBORw0KGgoAAAANSUhEUg"'
        result = _sanitize_tool_content(content, "chart")
        assert "[Image generated successfully by 'chart'" in result

    def test_starts_with_ivbor_detected(self):
        content = "iVBORw0KGgoAAAANSUhEUg..."
        result = _sanitize_tool_content(content, "draw")
        assert "[Image generated successfully by 'draw'" in result

    def test_non_image_text_passthrough(self):
        content = "Just a plain text result"
        result = _sanitize_tool_content(content, "search")
        assert result == "Just a plain text result"

    def test_empty_string_passthrough(self):
        result = _sanitize_tool_content("", "tool")
        assert result == ""


# ── _format_intermediate_steps ──────────────────────────────────────────────

class TestFormatIntermediateSteps:
    def test_normal_result_with_dict_content(self):
        steps = [{
            "tool_calls": [{"tool": "search", "arguments": {}}],
            "results": [{"ok": True, "result": {"content": "found it"}, "tool": "search"}],
        }]
        result = _format_intermediate_steps(steps)
        assert len(result) == 1
        assert result[0]["results"][0]["ok"] is True
        assert result[0]["results"][0]["content"] == "found it"

    def test_normal_result_with_non_dict_content_serialized(self):
        steps = [{
            "tool_calls": [],
            "results": [{"ok": True, "result": 42, "tool": "calc"}],
        }]
        result = _format_intermediate_steps(steps)
        assert json.loads(result[0]["results"][0]["content"]) == 42

    def test_error_result_extracted(self):
        steps = [{
            "tool_calls": [],
            "results": [{"ok": False, "error": {"message": "timeout"}, "tool": "search"}],
        }]
        result = _format_intermediate_steps(steps)
        assert result[0]["results"][0]["ok"] is False
        assert result[0]["results"][0]["error"] == "timeout"

    def test_error_result_with_string_error(self):
        steps = [{
            "tool_calls": [],
            "results": [{"ok": False, "error": "simple error msg"}],
        }]
        result = _format_intermediate_steps(steps)
        assert result[0]["results"][0]["error"] == "simple error msg"

    def test_non_dict_result_converted_to_string(self):
        steps = [{
            "tool_calls": [],
            "results": [42],
        }]
        result = _format_intermediate_steps(steps)
        assert len(result) == 1
        assert result[0]["results"][0]["ok"] is False
        assert result[0]["results"][0]["content"] == "42"

    def test_image_content_sanitized(self):
        steps = [{
            "tool_calls": [],
            "results": [{"ok": True, "result": {"content": "type='image' iVBOR"}, "tool": "gen"}],
        }]
        result = _format_intermediate_steps(steps)
        assert "[Image generated successfully by 'gen'" in result[0]["results"][0]["content"]

    def test_multiple_steps_formatted(self):
        steps = [
            {"tool_calls": [{"tool": "a", "arguments": {}}], "results": [{"ok": True, "result": {"content": "r1"}, "tool": "a"}]},
            {"tool_calls": [{"tool": "b", "arguments": {}}], "results": [{"ok": False, "error": "fail", "tool": "b"}]},
        ]
        result = _format_intermediate_steps(steps)
        assert len(result) == 2
        assert result[0]["results"][0]["content"] == "r1"
        assert result[1]["results"][0]["error"] == "fail"

    def test_step_with_empty_results(self):
        steps = [{"tool_calls": [], "results": []}]
        result = _format_intermediate_steps(steps)
        assert len(result) == 1
        assert result[0]["results"] == []


# ── _build_unknown_tool_error_step ──────────────────────────────────────────

class TestBuildUnknownToolErrorStep:
    def test_single_unknown_tool(self):
        result = _build_unknown_tool_error_step(["fake_tool"], ["search", "calc"])
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "fake_tool"
        assert len(result["results"]) == 1
        assert result["results"][0]["ok"] is False
        assert "Unknown tool 'fake_tool'" in result["results"][0]["error"]
        assert '"search"' in result["results"][0]["error"]
        assert '"calc"' in result["results"][0]["error"]

    def test_multiple_unknown_tools(self):
        result = _build_unknown_tool_error_step(["a", "b"], ["search"])
        assert len(result["tool_calls"]) == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["tool"] == "a"
        assert result["results"][1]["tool"] == "b"

    def test_empty_unknown_tools(self):
        result = _build_unknown_tool_error_step([], ["search", "calc"])
        assert result["tool_calls"] == []
        assert result["results"] == []
