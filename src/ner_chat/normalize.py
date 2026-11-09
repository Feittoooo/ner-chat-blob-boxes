"""Name normalization shared by extract (building the store) and ask (looking names up).

Both sides must call the same function, otherwise "Marta Ibarra" stored at extract time would never match
"marta  ibarra" typed at ask time.
"""

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
# Possessive suffix: "Lumora Energy's" and "Lumora Energy" are the same entity.
_POSSESSIVE = re.compile(r"['’]s$")
_EDGE_PUNCTUATION = "\"'`.,;:!?()[]{}<>‘’“”"


def normalize_name(name: str) -> str:
    """Return the lookup key for an entity name: NFKC, casefolded, single spaces, no edge punctuation."""
    key = unicodedata.normalize("NFKC", name).casefold()
    key = _WHITESPACE.sub(" ", key).strip().strip(_EDGE_PUNCTUATION).strip()
    return _POSSESSIVE.sub("", key).strip()
