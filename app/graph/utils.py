import json
import logging
import re
from typing import Any

from app.graph.state import ToolCall

logger = logging.getLogger(__name__)


def _extract_token_estimate(response: dict[str, Any]) -> int | None:
    """Extract and sum prompt and completion token counts from an LLM response dict.

    :param response: A dictionary containing ``prompt_eval_count`` and/or
        ``eval_count`` keys as returned by an LLM API call.
    :returns: The total token count, or ``None`` when neither key is present.
    """
    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")
    if prompt_tokens is None and completion_tokens is None:
        return None
    return int(prompt_tokens or 0) + int(completion_tokens or 0)


def _parse_json_object(content: str) -> dict[str, Any] | None:
    """Parse a JSON object from *content*, falling back to regex extraction.

    :param content: A string that may contain valid JSON or a JSON-like object
        wrapped in curly braces.
    :returns: A ``dict`` if parsing succeeds, otherwise ``None``.
    """
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _router_fallback(user_message: str) -> dict[str, Any]:
    """Determine routing metadata for simple / complex messages when the LLM
    router does not return a structured response.

    :param user_message: The raw text input from the end user.
    :returns: A dictionary with keys ``is_simple``, ``needs_tools``,
        ``is_complex_task``, ``needs_reasoning_model``, and optionally ``answer``.
    """
    message = user_message.lower().strip()
    if message in {"hi", "hello", "hey", "thanks", "thank you"}:
        return {
            "is_simple": True,
            "needs_tools": False,
            "is_complex_task": False,
            "needs_reasoning_model": False,
            "answer": (
                "You're welcome!"
                if message in {"thanks", "thank you"}
                else "Hello! How can I help you today?"
            ),
        }
    code_words = ["code", "implement", "architecture", "debug", "analyze", "compare", "design"]
    is_complex = any(word in message for word in code_words)
    return {
        "is_simple": False,
        "needs_tools": not is_complex,
        "is_complex_task": is_complex,
        "needs_reasoning_model": is_complex,
        "answer": "",
    }


def _normalize_tool_calls(raw_calls: Any) -> list[ToolCall]:
    """Validate and normalise raw tool-call payloads into a consistent list.

    :param raw_calls: A list of dicts, each expected to contain ``tool`` (str)
        and ``arguments`` (dict) keys.  Non-dict entries are skipped.
    :returns: A list of ``ToolCall`` dicts with guaranteed ``tool`` and
        ``arguments`` keys.
    """
    normalized: list[ToolCall] = []
    if not isinstance(raw_calls, list):
        return normalized
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool")
        arguments = item.get("arguments", {})
        if isinstance(tool, str) and isinstance(arguments, dict):
            normalized.append({"tool": tool, "arguments": arguments})
    return normalized


def build_history_section(matches: list[dict[str, Any]], settings) -> str:
    """Convert search_history matches into a Relevant past context block.

    :param matches: List of match dicts returned by the ``search_history`` MCP tool.
        Each dict must contain at least ``role``, ``content``, and ``distance``.
        ``created_at`` (ISO-8601 string) is optional but rendered when present.
    :returns: A formatted markdown string starting with ``## Relevant past context``,
        or an empty string if there are no relevant matches.
    """
    if not matches:
        return ""

    filtered = [
        m for m in matches
        if isinstance(m.get("distance"), (int, float))
        and m["distance"] <= settings.DISTANCE_THRESHOLD
    ]

    if not filtered:
        logger.debug(
            "search_history returned %d match(es) but all exceeded distance threshold %.2f",
            len(matches),
            settings.DISTANCE_THRESHOLD,
        )
        return ""

    lines: list[str] = ["## Relevant past context", ""]
    for i, m in enumerate(filtered, start=1):
        role = str(m.get("role") or "unknown").capitalize()
        content = str(m.get("content") or "").strip()
        created_at = m.get("created_at") or ""
        date_suffix = f", {created_at[:10]}" if created_at else ""
        lines.append(f"{i}. [{role}{date_suffix}]: {content}")

    return "\n".join(lines)


def inject_history_into_prompt(base_prompt: str, matches: list[dict[str, Any]], settings) -> str:
    """Return *base_prompt* with the history section appended when relevant.

    This is the single entry-point intended to be called from the orchestrator
    node at the start of every turn, right after ``search_history`` resolves.

    :param base_prompt: The orchestrator's base system prompt string.
    :param matches: List of match dicts from ``search_history``. Pass ``[]`` on error.

    :returns: ``base_prompt`` unchanged when no relevant history exists,
    otherwise ``base_prompt + "\\n\\n" + history_section``.
    """
    history_section = build_history_section(matches, settings)
    if not history_section:
        return base_prompt
    return f"{base_prompt}\n\n{history_section}"
    
