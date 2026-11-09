"""The local entity store: a JSON file written by `extract` and read (only through tools) by `ask`.

Minimal valid file, e.g. for a test fixture:

    {"entities": [{"name": "Marta Ibarra", "label": "person", "mention_count": 3}]}

The normalized lookup key is never stored: it is recomputed with `normalize_name` on load, so extract and
lookup can't disagree on it.
"""

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from ner_chat.normalize import normalize_name

SCHEMA_VERSION = 1


class Mention(BaseModel):
    start: int
    end: int


class EntityRecord(BaseModel):
    name: str
    label: str
    mention_count: int = Field(ge=0)
    mentions: list[Mention] = Field(default_factory=list)


class SourceInfo(BaseModel):
    path: str | None = None
    sha256: str | None = None
    extracted_at: str | None = None


class NerInfo(BaseModel):
    model: str | None = None
    labels: list[str] = Field(default_factory=list)
    threshold: float | None = None


class StoreFile(BaseModel):
    """On-disk format. Everything except `entities[].name/label/mention_count` is optional."""

    schema_version: int = SCHEMA_VERSION
    source: SourceInfo = Field(default_factory=SourceInfo)
    ner: NerInfo = Field(default_factory=NerInfo)
    entities: list[EntityRecord] = Field(default_factory=list)


class RawMention(BaseModel):
    """One NER hit, before aggregation."""

    text: str
    label: str
    start: int
    end: int


class EntityStore:
    """In-memory store indexed by normalized name. All queries return records sorted deterministically."""

    def __init__(self, data: StoreFile):
        self.data = data
        self._by_key: dict[str, EntityRecord] = {}
        for record in data.entities:
            key = normalize_name(record.name)
            if not key:
                continue
            existing = self._by_key.get(key)
            self._by_key[key] = record if existing is None else _merge(existing, record)
        self.data.entities = self._sorted(self._by_key.values())

    @classmethod
    def load(cls, path: str | Path) -> "EntityStore":
        return cls(StoreFile.model_validate_json(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.data.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_mentions(
        cls, mentions: list[RawMention], source: SourceInfo | None = None, ner: NerInfo | None = None
    ) -> "EntityStore":
        """Aggregate NER hits into one record per normalized name.

        The displayed name is the most frequent surface form and the label the most frequent label
        (ties go to the first one seen, because Counter.most_common is stable).
        """
        groups: dict[str, list[RawMention]] = {}
        for mention in mentions:
            key = normalize_name(mention.text)
            if key:
                groups.setdefault(key, []).append(mention)

        records = [
            EntityRecord(
                name=Counter(m.text.strip() for m in group).most_common(1)[0][0],
                label=Counter(m.label for m in group).most_common(1)[0][0],
                mention_count=len(group),
                mentions=[Mention(start=m.start, end=m.end) for m in group],
            )
            for group in groups.values()
        ]
        return cls(StoreFile(source=source or SourceInfo(), ner=ner or NerInfo(), entities=records))

    def lookup(self, name: str) -> EntityRecord | None:
        """Exact match on the normalized name."""
        return self._by_key.get(normalize_name(name))

    def search(self, query: str, limit: int) -> list[EntityRecord]:
        """Entities whose normalized name contains the query, or shares a word with it."""
        q = normalize_name(query)
        if not q:
            return []
        q_tokens = set(q.split())
        hits = [r for key, r in self._by_key.items() if q in key or q_tokens & set(key.split())]
        return self._sorted(hits)[:limit]

    def top(self, label: str | None, limit: int) -> tuple[int, list[EntityRecord]]:
        """Most mentioned entities, optionally of one label. Returns (total matching, first `limit`)."""
        records = list(self._by_key.values())
        if label:
            wanted = normalize_name(label)
            records = [r for r in records if normalize_name(r.label) == wanted]
        return len(records), self._sorted(records)[:limit]

    def labels(self) -> dict[str, int]:
        """Number of distinct entities per label."""
        return dict(sorted(Counter(r.label for r in self._by_key.values()).items()))

    def __len__(self) -> int:
        return len(self._by_key)

    @staticmethod
    def _sorted(records) -> list[EntityRecord]:
        return sorted(records, key=lambda r: (-r.mention_count, normalize_name(r.name)))


def _merge(a: EntityRecord, b: EntityRecord) -> EntityRecord:
    """Two stored records that normalize to the same key: sum counts, keep the bigger one's name and label."""
    keep = a if a.mention_count >= b.mention_count else b
    return EntityRecord(
        name=keep.name,
        label=keep.label,
        mention_count=a.mention_count + b.mention_count,
        mentions=a.mentions + b.mentions,
    )
