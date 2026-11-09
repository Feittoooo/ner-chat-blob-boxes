"""The `ask` path: agent run + grounding. This is the entry point the tests call."""

import sys

from pydantic import ValidationError
from pydantic_ai import UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model

from ner_chat.agent import agent
from ner_chat.grounding import deterministic_reply, ground
from ner_chat.schemas import AskResponse, EntityRef
from ner_chat.store import EntityStore


def run_ask(question: str, store: EntityStore, model: Model | str) -> AskResponse:
    """Answer `question` from `store` only. Every returned entity is grounded against the store.

    If the model keeps producing invalid output until retries run out, warn on stderr and fall back to a
    deterministic reply built from the entities it last tried to cite. Never raises for model misbehaviour.
    """
    with capture_run_messages() as messages:
        try:
            result = agent.run_sync(question, deps=store, model=model)
        except UnexpectedModelBehavior as exc:
            print(f"warning: the model's answer failed validation ({exc}); answering from the store directly.", file=sys.stderr)
            grounded = ground(AskResponse(reply="", entities=_last_cited_entities(messages)), store)
            return AskResponse(reply=deterministic_reply(grounded.entities), entities=grounded.entities)
    return ground(result.output, store)


def _last_cited_entities(messages: list[ModelMessage]) -> list[EntityRef]:
    """Entities from the most recent output-tool call that still parses, or [] if there is none."""
    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, ToolCallPart) and part.tool_name.startswith("final_result"):
                try:
                    return AskResponse.model_validate(part.args_as_dict()).entities
                except (ValidationError, ValueError):
                    continue
    return []
