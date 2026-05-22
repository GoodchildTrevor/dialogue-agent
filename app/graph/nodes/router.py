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
    """Handles routing of user messages to appropriate handlers.
    
    Analyzes incoming user messages and determines whether they should be:
    - Answered directly (simple queries)
    - Routed through the orchestrator for tool usage or complex processing
    - Handled via fallback logic when routing fails
    """
    def __init__(self, llm_client: Any, settings: Any) -> None:
        """Initialize the RouterNode with LLM client and settings.
        
        :param llm_client: The LLM client for making chat completions.
        :param settings: Configuration settings containing model parameters.
        """
        self.llm_client = llm_client
        self.settings = settings

    async def action(self, state: AssistantState) -> dict[str, Any]:
        """Route the user's message to the appropriate handler.
        
        Analyzes the user's message and determines whether it should be answered
        directly (simple), routed through the orchestrator (tools/complex tasks),
        or handled via fallback logic.
        
        :param state: The current assistant state containing messages and metadata.
        :returns: A dictionary with routing decision and next action. Contains keys
            such as 'next_action' ('end', 'orchestrator'), 'final_answer' (for simple
            responses), and task classification flags like 'is_complex_task'.
        """
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
