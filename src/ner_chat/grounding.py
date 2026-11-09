"""Grounding: make every cited entity match the store, regardless of what the model wrote.

This runs in plain code after the agent returns, so it holds even for a model that ignores its tools
(TestModel, for instance, returns schema-shaped dummy data without reading any tool result).
"""

from ner_chat.normalize import normalize_name
from ner_chat.schemas import NOT_MENTIONED, AskResponse, EntityRef
from ner_chat.store import EntityStore


def ground(response: AskResponse, store: EntityStore) -> AskResponse:
    """Overwrite name/label/count of each cited entity with store values; unknown names get count 0."""
    seen: set[str] = set()
    entities: list[EntityRef] = []
    for ref in response.entities:
        key = normalize_name(ref.name)
        if not key or key in seen:
            continue
        seen.add(key)
        record = store.lookup(ref.name)
        if record is None:
            entities.append(EntityRef(name=ref.name.strip(), label=NOT_MENTIONED, mention_count=0))
        else:
            entities.append(EntityRef(name=record.name, label=record.label, mention_count=record.mention_count))
    return AskResponse(reply=response.reply, entities=entities)


def deterministic_reply(entities: list[EntityRef]) -> str:
    """A reply built only from grounded entities, used when the model's own reply can't be trusted."""
    if not entities:
        return "Sorry, I couldn't produce an answer grounded in the document for that question."
    parts = [
        f"{e.name} ({e.label}) is mentioned {e.mention_count} time{'' if e.mention_count == 1 else 's'}"
        if e.mention_count
        else f"{e.name} is not mentioned in the document"
        for e in entities
    ]
    return "; ".join(parts) + "."
