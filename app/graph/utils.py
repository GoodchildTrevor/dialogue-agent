import json
import logging
import re
from typing import Any

from app.graph.state import ToolCall


logger = logging.getLogger(__name__)


def _extract_token_estimate(response: dict[str, Any]) -> int | None:
    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")
    if prompt_tokens is None and completion_tokens is None:
        return None
    return int(prompt_tokens or 0) + int(completion_tokens or 0)


def _parse_json_object(content: str) -> dict[str, Any] | None:
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
