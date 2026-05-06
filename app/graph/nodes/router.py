import hashlib
import logging
from typing import Any

from app.core.tracing import trace
from app.graph.prompt_fragments import build_router_prompt
from app.graph.state import AssistantState
from app.graph.utils import (
    _extract_token_estimate,
    _parse_json_object,
    _router_fallback
)

logger = logging.getLogger(__name__)


class RouterNode():
    def __init__(self, llm_client, settings):
        self.llm_client = llm_client
        self.settings = settings

    async def action(self, state: AssistantState) -> dict[str, Any]:
        user_message = state["messages"][-1]["content"]
        payload = {"message": user_message}
        async with trace(
            step_name="router",
            user_id=state["user_id"],
            request_id=state["request_id"],
            input=payload,
        ) as t:
            t.input_hash = hashlib.sha256(user_message.encode()).hexdigest()
            router_prompt = build_router_prompt()
            prompt = [
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": user_message},
            ]
            response = await self.llm_client.chat(
                model=self.settings.ROUTER_MODEL,
                messages=prompt,
                format="json",
            )
            t.estimated_tokens = _extract_token_estimate(response)
            t.model_used = self.settings.ROUTER_MODEL
            content = response.get("message", {}).get("content", "")
            parsed = _parse_json_object(content)
            update = _router_fallback(user_message) if parsed is None else parsed
            t.output = update

            # Routing logic:
            # - is_simple: answer immediately, no tools needed
            # - needs_tools / needs_reasoning_model / is_complex_task:
            #   ALL go through the orchestrator first so it can call tools.
            #   The orchestrator will escalate to reasoning if it decides to.
            #   Never skip the orchestrator — otherwise tool calls are impossible.
            if update.get("is_simple") and update.get("answer"):
                t.route_decision = "simple"
            elif parsed is None:
                t.route_decision = "fallback"
            else:
                t.route_decision = "orchestrator"

        logger.info(
            "[%s] router: route=%s is_simple=%s needs_tools=%s is_complex=%s needs_reasoning=%s",
            state["request_id"],
            t.route_decision,
            update.get("is_simple"),
            update.get("needs_tools"),
            update.get("is_complex_task"),
            update.get("needs_reasoning_model"),
        )

        if update.get("is_simple") and update.get("answer"):
            return {"final_answer": str(update["answer"]), "next_action": "end"}

        # Everything else — tools, complex tasks, reasoning hints — goes to orchestrator.
        # The orchestrator will call `escalate` if it needs the strong model.
        return {
            "is_complex_task": bool(update.get("is_complex_task", False)),
            "next_action": "orchestrator",
        }
