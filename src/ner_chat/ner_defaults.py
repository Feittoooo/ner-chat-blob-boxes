"""NER defaults, kept apart from ner.py so the CLI can show them without importing torch."""

DEFAULT_NER_MODEL = "knowledgator/gliner-bi-edge-v2.0"  # smallest GLiNER bi-encoder (60M)
DEFAULT_LABELS = ["person", "organization", "location"]
DEFAULT_THRESHOLD = 0.5
