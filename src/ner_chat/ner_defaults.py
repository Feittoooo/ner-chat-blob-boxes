"""NER defaults, kept apart from ner.py so the CLI can show them without importing torch."""

DEFAULT_NER_MODEL = "knowledgator/gliner-bi-edge-v2.0"  # smallest GLiNER bi-encoder (60M)
# "country" is needed: with only "location", the 60M model misses "Spain" at any threshold (see DECISIONS.md).
DEFAULT_LABELS = ["person", "organization", "location", "country"]
DEFAULT_THRESHOLD = 0.5
