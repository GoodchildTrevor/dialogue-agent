import json
from typing import Any

from app.core.tracing import _make_json_safe, trace
from app.graph.prompt_fragments import build_reasoning_prompt
from app.graph.state import AssistantState


class StrongModelNode():

    def __init__(self, llm_client, settings):
        self.llm_client = llm_client
        self.settings = settings

    async def action(self, state: AssistantState) -> dict[str, Any]:
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
            t.model_used = self.settings.REASONING_MODEL
            answer = await self._invoke_reasoning_model(task, state)
            t.output = {"answer": answer}
        return {"final_answer": answer, "next_action": "end"}

    async def _invoke_reasoning_model(self, task: str, state: AssistantState) -> str:
        safe_context = _make_json_safe(state.get("context", {}))
        safe_steps = _make_json_safe(state.get("intermediate_steps", []))
        reasoning_prompt = build_reasoning_prompt()
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
        response = await self.llm_client.chat(
            model=self.settings.REASONING_MODEL,
            messages=prompt,
            timeout=self.settings.REASONING_TIMEOUT_SECONDS,
        )
        return response.get("message", {}).get("content", "")
