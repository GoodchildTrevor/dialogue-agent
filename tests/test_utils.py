"""Tests for app.graph.utils pure utility functions."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.graph.utils import (
    _extract_token_estimate,
    _normalize_tool_calls,
    _parse_json_object,
    _router_fallback,
    build_history_section,
    inject_history_into_prompt,
)


class TestExtractTokenEstimate:
    """Tests for _extract_token_estimate."""

    def test_both_keys_present(self):
        response = {"prompt_eval_count": 10, "eval_count": 20}
        assert _extract_token_estimate(response) == 30

    def test_only_prompt_tokens(self):
        response = {"prompt_eval_count": 42}
        assert _extract_token_estimate(response) == 42

    def test_only_completion_tokens(self):
        response = {"eval_count": 15}
        assert _extract_token_estimate(response) == 15

    def test_neither_key_present(self):
        response = {"foo": "bar"}
        assert _extract_token_estimate(response) is None

    def test_both_none(self):
        response = {"prompt_eval_count": None, "eval_count": None}
        assert _extract_token_estimate(response) is None

    def test_zero_values(self):
        response = {"prompt_eval_count": 0, "eval_count": 0}
        assert _extract_token_estimate(response) == 0


class TestParseJsonObject:
    """Tests for _parse_json_object."""

    def test_valid_json_string(self):
        text = '{"key": "value", "num": 42}'
        result = _parse_json_object(text)
        assert result == {"key": "value", "num": 42}

    def test_valid_json_nested(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = _parse_json_object(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_invalid_json_no_fallback(self):
        """When no JSON-like pattern exists, returns None."""
        text = "just plain text with no braces"
        assert _parse_json_object(text) is None

    def test_regex_fallback_single_brace_pair(self):
        """Text with one {…} pair — extracts the inner content."""
        text = "Some preamble {\"key\": \"value\"} trailing"
        result = _parse_json_object(text)
        assert result == {"key": "value"}

    def test_regex_fallback_nested_braces(self):
        """Nested braces: outermost pair is used."""
        text = '{ "outer": { "inner": 1 } }'
        result = _parse_json_object(text)
        assert result == {"outer": {"inner": 1}}

    def test_empty_string(self):
        assert _parse_json_object("") is None

    def test_list_json_not_dict(self):
        """JSON array returns None (only objects accepted)."""
        text = "[1, 2, 3]"
        assert _parse_json_object(text) is None

    def test_malformed_regex_content_raises(self):
        """Regex finds braces but content is unparseable — returns None."""
        text = "{not valid json at all"
        assert _parse_json_object(text) is None


class TestRouterFallback:
    """Tests for _router_fallback."""

    def test_greeting_hi(self):
        result = _router_fallback("hi")
        assert result["is_simple"] is True
        assert result["needs_tools"] is False
        assert result["answer"] == "Hello! How can I help you today?"

    def test_greeting_hello(self):
        result = _router_fallback("hello")
        assert result["is_simple"] is True
        assert result["answer"] == "Hello! How can I help you today?"

    def test_greeting_thanks(self):
        result = _router_fallback("thanks")
        assert result["is_simple"] is True
        assert result["answer"] == "You're welcome!"

    def test_greeting_thank_you(self):
        result = _router_fallback("thank you")
        assert result["is_simple"] is True
        assert result["answer"] == "You're welcome!"

    def test_code_implementation(self):
        result = _router_fallback("Implement a binary search in Python")
        assert result["is_simple"] is False
        assert result["is_complex_task"] is True
        assert result["needs_reasoning_model"] is True
        assert result["answer"] == ""

    def test_code_function(self):
        result = _router_fallback("Can you write a function to sort a list?")
        assert result["is_complex_task"] is False
        assert result["needs_tools"] is True

    def test_generic_question_no_match(self):
        """Unrelated question — not simple, needs tools."""
        result = _router_fallback("What's the weather today?")
        assert result["is_simple"] is False
        assert result["needs_tools"] is True

    def test_empty_string(self):
        result = _router_fallback("")
        assert result["is_simple"] is False
        assert result["needs_tools"] is True

    def test_case_insensitive_greeting(self):
        result = _router_fallback("HELLO")
        assert result["is_simple"] is True

    def test_case_insensitive_code(self):
        result = _router_fallback("DEBUG this SQL query")
        assert result["is_complex_task"] is True


class TestNormalizeToolCalls:
    """Tests for _normalize_tool_calls."""

    def test_valid_list_of_dicts(self):
        raw = [{"tool": "foo", "arguments": {"a": 1}}]
        result = _normalize_tool_calls(raw)
        assert len(result) == 1
        assert result[0]["tool"] == "foo"
        assert result[0]["arguments"] == {"a": 1}

    def test_empty_list(self):
        assert _normalize_tool_calls([]) == []

    def test_non_list_input_string(self):
        """String input returns empty list (not wrapped)."""
        assert _normalize_tool_calls("single") == []

    def test_non_list_input_int(self):
        """Integer input returns empty list."""
        assert _normalize_tool_calls(42) == []

    def test_missing_tool_key(self):
        """Dict without 'tool' key is skipped."""
        raw = [{"arguments": {"a": 1}}]
        result = _normalize_tool_calls(raw)
        assert result == []

    def test_missing_arguments_key(self):
        """Dict without 'arguments' key defaults to empty dict."""
        raw = [{"tool": "bar"}]
        result = _normalize_tool_calls(raw)
        assert len(result) == 1
        assert result[0]["tool"] == "bar"
        assert result[0]["arguments"] == {}

    def test_non_dict_items_skipped(self):
        """Non-dict entries in list are skipped."""
        raw = [{"tool": "a", "arguments": {}}, "b", 42, None]
        result = _normalize_tool_calls(raw)
        assert len(result) == 1
        assert result[0]["tool"] == "a"

    def test_tool_not_string(self):
        """Tool key must be a string."""
        raw = [{"tool": 123, "arguments": {}}]
        result = _normalize_tool_calls(raw)
        assert result == []

    def test_arguments_not_dict(self):
        """Arguments must be a dict."""
        raw = [{"tool": "x", "arguments": "not-a-dict"}]
        result = _normalize_tool_calls(raw)
        assert result == []

    def test_none_input(self):
        assert _normalize_tool_calls(None) == []


class TestBuildHistorySection:
    """Tests for build_history_section."""

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.DISTANCE_THRESHOLD = 0.5
        return settings

    def test_no_matches(self, mock_settings):
        assert build_history_section([], mock_settings) == ""

    def test_all_below_threshold(self, mock_settings):
        matches = [
            {"distance": 0.9, "role": "user", "content": "q"},
        ]
        assert build_history_section(matches, mock_settings) == ""

    def test_valid_match_with_date(self, mock_settings):
        result = build_history_section(
            [
                {
                    "id": 1,
                    "role": "user",
                    "content": "What is Python?",
                    "distance": 0.3,
                    "created_at": "2024-06-15T14:30:00",
                },
            ],
            mock_settings,
        )
        assert "## Relevant past context" in result
        assert "What is Python?" in result
        assert "2024-06-15" in result

    def test_valid_match_without_date(self, mock_settings):
        result = build_history_section(
            [
                {
                    "id": 1,
                    "role": "assistant",
                    "content": "Python is a language.",
                    "distance": 0.2,
                },
            ],
            mock_settings,
        )
        assert "## Relevant past context" in result
        assert "Python is a language." in result

    def test_role_capitalized(self, mock_settings):
        result = build_history_section(
            [
                {"role": "user", "content": "hi", "distance": 0.1},
                {"role": "assistant", "content": "hello", "distance": 0.1},
            ],
            mock_settings,
        )
        assert "[User" in result
        assert "[Assistant" in result

    def test_default_role_unknown(self, mock_settings):
        result = build_history_section(
            [{"content": "something", "distance": 0.1}],
            mock_settings,
        )
        assert "[Unknown" in result

    def test_empty_content_handled(self, mock_settings):
        result = build_history_section(
            [{"role": "user", "content": "", "distance": 0.1}],
            mock_settings,
        )
        assert "## Relevant past context" in result

    def test_distance_exactly_at_threshold(self, mock_settings):
        """Distance == threshold should be included."""
        mock_settings.DISTANCE_THRESHOLD = 0.5
        result = build_history_section(
            [{"role": "user", "content": "borderline", "distance": 0.5}],
            mock_settings,
        )
        assert "## Relevant past context" in result

    def test_distance_above_threshold(self, mock_settings):
        """Distance > threshold should be excluded."""
        mock_settings.DISTANCE_THRESHOLD = 0.5
        result = build_history_section(
            [{"role": "user", "content": "too far", "distance": 0.51}],
            mock_settings,
        )
        assert result == ""

    def test_multiple_matches_numbered(self, mock_settings):
        result = build_history_section(
            [
                {"role": "user", "content": "First", "distance": 0.1},
                {"role": "user", "content": "Second", "distance": 0.2},
                {"role": "user", "content": "Third", "distance": 0.3},
            ],
            mock_settings,
        )
        assert "1. [User]: First" in result
        assert "2. [User]: Second" in result
        assert "3. [User]: Third" in result


class TestInjectHistoryIntoPrompt:
    """Tests for inject_history_into_prompt."""

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.DISTANCE_THRESHOLD = 0.5
        return settings

    def test_empty_matches_returns_base_unchanged(self, mock_settings):
        base = "You are a helpful assistant."
        result = inject_history_into_prompt(base, [], mock_settings)
        assert result == base

    def test_relevant_matches_appended(self, mock_settings):
        base = "You are a helpful assistant."
        matches = [
            {
                "id": 1,
                "role": "user",
                "content": "Previous question",
                "distance": 0.3,
            },
        ]
        result = inject_history_into_prompt(base, matches, mock_settings)
        assert base in result
        assert "## Relevant past context" in result
        assert "Previous question" in result

    def test_multiple_matches_injected(self, mock_settings):
        base = "Base prompt"
        matches = [
            {"role": "user", "content": "Q1", "distance": 0.3},
            {"role": "user", "content": "Q2", "distance": 0.4},
        ]
        result = inject_history_into_prompt(base, matches, mock_settings)
        assert "Base prompt" in result
        assert "Q1" in result
        assert "Q2" in result

    def test_below_threshold_matches_ignored(self, mock_settings):
        base = "You are helpful."
        matches = [
            {"role": "user", "content": "q", "distance": 0.9},
        ]
        result = inject_history_into_prompt(base, matches, mock_settings)
        assert result == base

    def test_history_appended_with_double_newline(self, mock_settings):
        base = "System prompt"
        matches = [
            {"role": "user", "content": "hi", "distance": 0.1},
        ]
        result = inject_history_into_prompt(base, matches, mock_settings)
        # History section is separated by \n\n from the base prompt
        assert "System prompt\n\n## Relevant past context" in result
