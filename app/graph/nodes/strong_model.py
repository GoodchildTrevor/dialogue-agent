import json
import logging
from typing import Any

from app.core.tracing import _make_json_safe, trace
from app.graph.prompt_fragments import build_reasoning_prompt
from app.graph.state import AssistantState
from app.graph.tool_registry import ToolRegistry
from app.graph.utils import _parse_json_object

log = logging.getLogger(__name__)


class StrongModelNode():
    """A node that handles reasoning-based tasks using a strong language model.

    This node processes tasks through a reasoning model, optionally invoking
    tools when the task requires external actions, and returns final answers
    when tool usage is not needed.

    :param llm_client: The LLM client used to invoke reasoning models.
    :param settings: Configuration settings containing model parameters and timeouts.
    :param tool_registries: Optional list of tool registries providing available tools.
        Defaults to an empty list if not provided.
    """

    def __init__(self, llm_client, settings, tool_registries: list[ToolRegistry] | None = None):
        self.llm_client = llm_client
        self.settings = settings
        self.tool_registries = tool_registries or []

    def _all_tool_descriptions(self) -> list[dict[str, Any]]:
        """Collect tool descriptions from all registered tool registries.

        :return: A list of dictionaries, each containing the schema description
            of an available tool from all registries.
        """
        result = []
        for registry in self.tool_registries:
            result.extend(registry.describe_for_model())
        return result

    async def action(self, state: AssistantState) -> dict[str, Any]:
        """Execute the reasoning model to process the task.

        Invokes the reasoning model with the task, context, and intermediate steps.
        Routes to tool execution if tool calls are requested, otherwise returns
        the final answer.

        :param state: The current assistant state containing messages, context, and metadata.
        :return: A dictionary with either:
            - {"pending_tool_calls": [...], "next_action": "tools"} if tool calls are needed.
            - {"final_answer": str, "next_action": "end"} for a complete answer.
        """
        task = (
            state.get("context", {}).get("escalation_task")
            or state["messages"][-1]["content"]
        )
        payload = {
            "task": task,
            "context": state.get("context", {}),
            "intermediate_steps": state.get("intermediate_steps", []),
        }
        async with trace(
            step_name="reasoning_model",
            user_id=state["user_id"],
            request_id=state["request_id"],
            input=payload,
        ) as t:
            t.route_decision = "reasoning"
            t.model_used = self.settings.REASONING_MODEL
            result = await self._invoke_reasoning_model(task, state)
            t.output = result

        # If the reasoning model returned tool calls, route to the tool executor
        if result.get("tool_calls"):
            log.info(
                "[%s] reasoning model requested tool calls: %s",
                state["request_id"],
                [tc["tool"] for tc in result["tool_calls"]],
            )
            return {
                "pending_tool_calls": result["tool_calls"],
                "next_action": "tools",
            }

        # Otherwise return the text answer as final
        return {"final_answer": result.get("answer", ""), "next_action": "end"}

    async def _invoke_reasoning_model(self, task: str, state: AssistantState) -> dict[str, Any]:
        """Invoke the reasoning model with the task and context.

        Constructs the prompt with system instructions, tool descriptions (if available),
        and task context. Calls the LLM and parses the response to extract either
        tool calls or a direct answer.

        :param task: The task description or user message to process.
        :param state: The current assistant state containing context and intermediate steps.
        :return: A dictionary with keys:
            - "tool_calls": list of normalized tool call dicts (if tools are needed).
            - "answer": the model's response text (if no tools are needed).
        """
        safe_context = _make_json_safe(state.get("context", {}))
        safe_steps = _make_json_safe(state.get("intermediate_steps", []))

        tool_descriptions = self._all_tool_descriptions()

        # Build the system prompt: base reasoning prompt + tool info if available
        reasoning_prompt = build_reasoning_prompt()
        if tool_descriptions:
            tool_names = [t["name"] for t in tool_descriptions if isinstance(t, dict) and "name" in t]
            tool_section = (
                "\n\n## Available Tools\n"
                "You have access to the following tools. If the user's task requires an action "
                "that a tool can perform (searching the web, exporting files, saving documents, etc.), "
                "you MUST call the appropriate tool rather than just describing what to do.\n\n"
                "CRITICAL: Use ONLY these exact tool names:\n"
                + "\n".join(f"  - {n}" for n in tool_names)
                + "\n\nFull tool schemas:\n"
                + json.dumps(tool_descriptions, ensure_ascii=False)
                + '\n\n## Tool Call Output Format\n'
                'If you need to call tools, return valid JSON:\n'
                '{"tool_calls": [{"tool": "tool_name", "arguments": {"arg1": "value1"}}]}\n\n'
                'If you can answer directly without tools, return:\n'
                '{"answer": "your comprehensive response text"}\n\n'
                'You may include an optional "thought" field for reasoning:\n'
                '{"thought": "...", "tool_calls": [...]}\n'
                '{"thought": "...", "answer": "..."}\n\n'
                'IMPORTANT: Return ONLY valid JSON — no prose, no markdown fences.'
            )
            reasoning_prompt += tool_section

        prompt = [
            {"role": "system", "content": reasoning_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "context": safe_context,
                        "intermediate_steps": safe_steps,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        format_param = "json" if tool_descriptions else None

        response = await self.llm_client.chat(
            model=self.settings.REASONING_MODEL,
            messages=prompt,
            format=format_param,
            timeout=self.settings.REASONING_TIMEOUT_SECONDS,
        )
        content = response.get("message", {}).get("content", "")

        # Try to parse as JSON to extract tool calls
        if tool_descriptions:
            parsed = _parse_json_object(content)
            if isinstance(parsed, dict):
                raw_tool_calls = parsed.get("tool_calls")
                if raw_tool_calls and isinstance(raw_tool_calls, list):
                    normalized_calls = []
                    for tc in raw_tool_calls:
                        if isinstance(tc, dict):
                            name = tc.get("tool") or tc.get("name")
                            args = tc.get("arguments") or tc.get("args") or {}
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {}
                            if name:
                                normalized_calls.append({
                                    "tool": str(name),
                                    "arguments": args if isinstance(args, dict) else {},
                                })
                    if normalized_calls:
                        return {"tool_calls": normalized_calls, "answer": ""}
                # No tool calls — return the answer
                answer = parsed.get("answer", content)
                return {"answer": answer}

        return {"answer": content}
