"""NER with GLinker's L1 layer only (GLiNER), without any knowledge base.

GLinker is an entity *linking* pipeline (L1 NER -> L2 candidate DB -> L3 linker -> L0). We only need mentions,
so we build the registered `l1_gliner` processor directly and never touch the DAG or a database.
"""

import re

from glinker import processor_registry

from ner_chat.ner_defaults import DEFAULT_LABELS, DEFAULT_NER_MODEL, DEFAULT_THRESHOLD
from ner_chat.store import RawMention

# GLiNER truncates inputs longer than the model's max_len with only a warning. Chunks of whole sentences up to
# this many words stay far below any GLiNER window (the edge model allows 2048 words).
MAX_CHUNK_WORDS = 200

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def prepare_text(raw: str) -> str:
    """Collapse every whitespace run to one space.

    OCR and PDF extraction break lines mid-sentence, even mid-name ("Marta\\n\\nIbarra"). Mention offsets
    in the store refer to this normalized text.
    """
    return _WHITESPACE.sub(" ", raw).strip()


def sentence_chunks(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[tuple[int, str]]:
    """Split text into (offset, chunk) pairs made of whole sentences, each at most `max_words` words.

    A single sentence longer than `max_words` is split on word boundaries.
    """
    pieces: list[tuple[int, int]] = []  # (start, end) spans of sentences, or of word runs for huge sentences
    start = 0
    for match in [*_SENTENCE_END.finditer(text), None]:
        end = match.start() if match else len(text)
        words = list(re.finditer(r"\S+", text[start:end]))
        for i in range(0, len(words), max_words):
            run = words[i : i + max_words]
            pieces.append((start + run[0].start(), start + run[-1].end()))
        if match:
            start = match.end()

    chunks: list[tuple[int, str]] = []
    chunk_start = chunk_end = None
    chunk_words = 0
    for piece_start, piece_end in pieces:
        n = len(text[piece_start:piece_end].split())
        if chunk_start is not None and chunk_words + n > max_words:
            chunks.append((chunk_start, text[chunk_start:chunk_end]))
            chunk_start = None
        if chunk_start is None:
            chunk_start, chunk_words = piece_start, 0
        chunk_end = piece_end
        chunk_words += n
    if chunk_start is not None:
        chunks.append((chunk_start, text[chunk_start:chunk_end]))
    return chunks


def detect_entities(
    text: str,
    labels: list[str] = DEFAULT_LABELS,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = DEFAULT_NER_MODEL,
) -> list[RawMention]:
    """Run GLiNER (via GLinker L1) over `text`; offsets in the result are relative to `text`."""
    chunks = sentence_chunks(text)
    if not chunks:
        return []

    make_l1 = processor_registry.get("l1_gliner")
    processor = make_l1({"model": model, "labels": labels, "threshold": threshold, "device": "cpu"})
    output = processor(texts=[chunk for _, chunk in chunks])

    return [
        RawMention(
            text=entity.text,
            label=entity.label or "unknown",
            start=offset + entity.start,
            end=offset + entity.end,
        )
        for (offset, _), entities in zip(chunks, output.entities, strict=True)
        for entity in entities
    ]
