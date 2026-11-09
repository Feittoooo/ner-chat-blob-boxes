"""Structured output of `ask`."""

from pydantic import BaseModel, Field

NOT_MENTIONED = "not mentioned"


class EntityRef(BaseModel):
    name: str = Field(description="Entity name exactly as a tool returned it, or as the user wrote it if not found")
    label: str = Field(description=f"Entity type from the store, or '{NOT_MENTIONED}'")
    mention_count: int = Field(ge=0, description="Number of mentions from the store; 0 if not mentioned")


class AskResponse(BaseModel):
    reply: str = Field(description="Short conversational answer (one or two sentences) that acknowledges the question")
    entities: list[EntityRef] = Field(
        default_factory=list,
        max_length=10,
        description="Only the entities the reply talks about, never the full inventory",
    )
