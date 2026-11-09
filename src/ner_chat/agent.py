"""The `ask` agent. It reaches the store only through these tools: the inventory is never put in a prompt."""

import re

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext

from ner_chat.schemas import AskResponse
from ner_chat.store import EntityRecord, EntityStore

SEARCH_MAX = 10
LIST_MAX = 20

INSTRUCTIONS = """\
You answer questions about the named entities (people, organizations, places, ...) found in one document.
You cannot see the document or the list of entities: use the tools to look things up.
- For a specific name, call count_mentions. If it is not found, try search_entities for partial matches
  before concluding it is not mentioned.
- For "who/which ... are mentioned" questions, use list_entities (optionally filtered by label) or list_labels.
Reply in one or two short sentences in the user's language, acknowledging the question.
In `entities`, list only the entities your reply talks about, including names the user asked about that are
not in the document (mention_count 0). Copy names, labels and counts exactly as the tools return them.\
"""

agent = Agent(
    deps_type=EntityStore,
    output_type=AskResponse,
    instructions=INSTRUCTIONS,
    retries={"tools": 1, "output": 2},
)


class EntityInfo(BaseModel):
    name: str
    label: str
    mention_count: int


class CountResult(EntityInfo):
    found: bool


class ListResult(BaseModel):
    total: int
    returned: int
    items: list[EntityInfo]


def _info(record: EntityRecord) -> EntityInfo:
    return EntityInfo(name=record.name, label=record.label, mention_count=record.mention_count)


def _clamp(value: int, high: int) -> int:
    # Models (and TestModel's generated arguments) can send 0, negative or huge limits.
    return max(1, min(high, value))


@agent.tool
def count_mentions(ctx: RunContext[EntityStore], name: str) -> CountResult:
    """Exact lookup of one entity name (case and spacing insensitive). Returns found=false and 0 if absent."""
    record = ctx.deps.lookup(name)
    if record is None:
        return CountResult(name=name, label="not mentioned", mention_count=0, found=False)
    return CountResult(**_info(record).model_dump(), found=True)


@agent.tool
def search_entities(ctx: RunContext[EntityStore], query: str, limit: int = 5) -> list[EntityInfo]:
    """Entities whose name contains the query or shares a word with it, most mentioned first (max 10)."""
    return [_info(r) for r in ctx.deps.search(query, _clamp(limit, SEARCH_MAX))]


@agent.tool
def list_entities(ctx: RunContext[EntityStore], label: str | None = None, limit: int = 10) -> ListResult:
    """Most mentioned entities, optionally only those with a given label (max 20 returned)."""
    total, records = ctx.deps.top(label, _clamp(limit, LIST_MAX))
    return ListResult(total=total, returned=len(records), items=[_info(r) for r in records])


@agent.tool
def list_labels(ctx: RunContext[EntityStore]) -> dict[str, int]:
    """Entity labels present in the document, with the number of distinct entities per label."""
    return ctx.deps.labels()


# How many words away from an entity name a number still counts as "its" count.
_NEAR_WORDS = 3


def numbers_near(reply: str, name: str) -> list[int]:
    """Integers written as digits within a few words of `name` in `reply`."""
    words = re.findall(r"\w+", reply.casefold())
    target = re.findall(r"\w+", name.casefold())
    found = []
    for i in range(len(words) - len(target) + 1):
        if target and words[i : i + len(target)] == target:
            window = words[max(0, i - _NEAR_WORDS) : i] + words[i + len(target) : i + len(target) + _NEAR_WORDS]
            found += [int(w) for w in window if w.isdigit()]
    return found


@agent.output_validator
def counts_in_reply_match_store(ctx: RunContext[EntityStore], output: AskResponse) -> AskResponse:
    """Reject a reply whose digits next to a cited entity name differ from that entity's stored count."""
    problems = []
    for ref in output.entities:
        record = ctx.deps.lookup(ref.name)
        actual = record.mention_count if record else 0
        problems += [
            f"{ref.name} is mentioned {actual} times, not {n}" for n in numbers_near(output.reply, ref.name) if n != actual
        ]
    if problems:
        raise ModelRetry("Counts in the reply must match the store: " + "; ".join(problems) + ".")
    return output
