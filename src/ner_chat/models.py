"""Turn the --model string into a Pydantic AI model.

Any Pydantic AI model string works. `ollama:<name>` is built explicitly so the server URL can come from
--ollama-url / OLLAMA_BASE_URL / a localhost default, instead of failing when the env var is unset.
"""

import os

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

DEFAULT_MODEL = "ollama:qwen3"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"

ALTERNATIVES = "Use --model ollama:<model> (free, local) or --model test (offline, no key)."


class ModelConfigError(Exception):
    """The model can't be built (unknown provider, missing API key); the message is meant for the CLI user."""


def ollama_url(override: str | None = None) -> str:
    return override or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL


def resolve_model(spec: str, ollama_base_url: str | None = None) -> Model:
    if spec.startswith("ollama:"):
        return OllamaModel(spec.removeprefix("ollama:"), provider=OllamaProvider(base_url=ollama_url(ollama_base_url)))
    try:
        return infer_model(spec)
    except UserError as exc:
        # e.g. "Set the `ANTHROPIC_API_KEY` environment variable ..." for a provider without its key.
        raise ModelConfigError(f"Cannot use model {spec!r}: {exc}\n{ALTERNATIVES}") from exc
