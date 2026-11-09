"""`ner-chat extract <file>` then `ner-chat ask "<question>"`."""

import argparse
import hashlib
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_STORE = "entities.json"
# Per model request, SDK retries included. Without a cap the OpenAI client Ollama uses waits up to 600 s per
# attempt and retries twice, so a failed model load could look like a 30-minute hang.
DEFAULT_TIMEOUT = 120.0


def _quiet_third_party_warnings() -> None:
    # glinker, torch and transformers emit deprecation/shadowing warnings on import that say nothing about our
    # run. GLiNER's truncation warning can't fire either: ner.py chunks the text below the model window.
    for module in ("glinker", "gliner", "torch", "transformers", "huggingface_hub", "pydantic"):
        warnings.filterwarnings("ignore", category=UserWarning, module=module)
        warnings.filterwarnings("ignore", category=FutureWarning, module=module)


def cmd_extract(args: argparse.Namespace) -> int:
    from ner_chat.extract import ExtractError, extract_text
    from ner_chat.ner import detect_entities, prepare_text  # imports torch: only on this path
    from ner_chat.store import EntityStore, NerInfo, SourceInfo

    path = Path(args.file)
    if not path.is_file():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1
    try:
        text = prepare_text(extract_text(path, force_ocr=args.ocr))
    except ExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    mentions = detect_entities(text, labels=labels, threshold=args.threshold, model=args.ner_model)
    store = EntityStore.from_mentions(
        mentions,
        source=SourceInfo(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            extracted_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ),
        ner=NerInfo(model=args.ner_model, labels=labels, threshold=args.threshold),
    )
    store.save(args.store)

    print(f"{len(text)} characters extracted, {len(mentions)} mentions, {len(store)} entities -> {args.store}")
    for record in store.data.entities:
        print(f"  {record.mention_count:>3}  {record.name}  [{record.label}]")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    import pydantic_ai
    from pydantic_ai.exceptions import ModelAPIError, ModelConnectionError, ModelNotFoundError, ModelRequestTimeout

    from ner_chat.ask import run_ask

    pydantic_ai.BANNER_ENABLED = False  # this CLI owns its output (stdout may be --json)
    from ner_chat.models import ALTERNATIVES, ModelConfigError, ollama_url, resolve_model
    from ner_chat.store import EntityStore

    store_path = Path(args.store)
    if not store_path.is_file():
        print(f"error: store {store_path} not found. Run `ner-chat extract <file>` first.", file=sys.stderr)
        return 1
    store = EntityStore.load(store_path)

    def ollama_hint() -> str:
        if not args.model.startswith("ollama:"):
            return ""
        name = args.model.removeprefix("ollama:")
        return (
            f"\nIs Ollama running at {ollama_url(args.ollama_url)}? Start it with `ollama serve`, download the "
            f"model with `ollama pull {name}`, and check `ollama ps` (loading needs free RAM)."
        )

    try:
        model = resolve_model(args.model, args.ollama_url)
        # stderr, so --json output on stdout stays clean. Without this a slow model load looks like a hang.
        print(f"Asking {args.model}... (a local model can take a minute to load on first use)", file=sys.stderr, flush=True)
        response = run_ask(args.question, store, model, request_timeout=args.timeout)
    except ModelConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ModelRequestTimeout:
        print(
            f"error: model {args.model!r} did not answer within {args.timeout:g} s.{ollama_hint()}\n"
            f"Retry, or raise --timeout. {ALTERNATIVES}",
            file=sys.stderr,
        )
        return 2
    except (ModelConnectionError, ModelNotFoundError) as exc:
        print(f"error: cannot reach model {args.model!r}: {exc}{ollama_hint()}\n{ALTERNATIVES}", file=sys.stderr)
        return 2
    except ModelAPIError as exc:
        print(f"error: model {args.model!r} failed: {exc}{ollama_hint()}\n{ALTERNATIVES}", file=sys.stderr)
        return 2

    if args.json:
        print(response.model_dump_json(indent=2))
        return 0
    print(response.reply)
    for entity in response.entities:
        count = f"{entity.mention_count} mention(s)" if entity.mention_count else "not mentioned"
        print(f"  - {entity.name} [{entity.label}]: {count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from ner_chat.models import DEFAULT_MODEL
    from ner_chat.ner_defaults import DEFAULT_LABELS, DEFAULT_NER_MODEL, DEFAULT_THRESHOLD

    parser = argparse.ArgumentParser(prog="ner-chat", description="Ask questions about the entities in a document.")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="extract text (Kreuzberg), detect entities (GLinker), write the store")
    extract.add_argument("file", help="document to read (PDF, DOCX, image, ...)")
    extract.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="comma-separated entity labels")
    extract.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="GLiNER confidence threshold")
    extract.add_argument("--ner-model", default=DEFAULT_NER_MODEL, help="GLiNER model id on Hugging Face")
    extract.add_argument("--store", default=DEFAULT_STORE, help="output JSON store")
    extract.add_argument("--ocr", action="store_true", help="force OCR (needs Tesseract language data)")
    extract.set_defaults(func=cmd_extract)

    ask = sub.add_parser("ask", help="answer one question using only the store")
    ask.add_argument("question")
    ask.add_argument("--store", default=DEFAULT_STORE, help="JSON store written by extract")
    ask.add_argument("--model", default=DEFAULT_MODEL, help=f"any Pydantic AI model string (default {DEFAULT_MODEL}; 'test' = offline)")
    ask.add_argument("--ollama-url", default=None, help="Ollama OpenAI-compatible URL (default $OLLAMA_BASE_URL or localhost)")
    ask.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"max seconds per model request (default {DEFAULT_TIMEOUT:g})"
    )
    ask.add_argument("--json", action="store_true", help="print the structured response as JSON")
    ask.set_defaults(func=cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows pipes default to cp1252
    _quiet_third_party_warnings()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
