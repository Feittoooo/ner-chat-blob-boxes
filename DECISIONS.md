# Decisions

Non-obvious choices, why they were made, and the alternative that was rejected.

## Tooling

### CPU-only torch index on Linux/Windows
`torch` is pinned and routed to `https://download.pytorch.org/whl/cpu` via `[tool.uv.sources]` (macOS uses PyPI,
whose wheels are already CPU/MPS-only). PyPI's Linux torch wheel pulls several GB of CUDA libraries that a 60M
GLiNER model on CPU never needs. Rejected: plain PyPI torch (a huge `uv sync` for anyone installing the project).

### Transitive deps pinned directly
`gliner` and `torch` are transitive deps of `glinker`, but they are pinned in `pyproject.toml` anyway because our
code depends on their behaviour (GLiNER truncation, CPU wheels). Everything else is pinned by `uv.lock`.

### Clean-install check (no system libraries)
A fresh `git clone` (no `.venv`) followed by `uv sync --locked` installs and imports `glinker`, `kreuzberg`
and `pydantic_ai`. A script over `uv.lock` confirmed that every locked package has a wheel for linux x86_64/aarch64,
macOS arm64 and Windows amd64 (cp312/cp313), so nothing compiles from an sdist. The likely suspect,
`psycopg2-binary` (pulled in by glinker's unused L2 PostgreSQL layer), ships wheels that bundle libpq.
Known gap: torch no longer publishes Intel macOS (x86_64) wheels, so Intel Macs are unsupported.

### argparse instead of Typer/Click
The CLI is two subcommands with a handful of flags. stdlib keeps one fewer pinned dependency.

## ner-chat

### Verified library APIs (installed versions, 2026-11-09)
| Library | Version | What was checked |
|---|---|---|
| pydantic-ai-slim | 2.66.0 | `infer_model("test")` returns `TestModel`; retry budgets are `Agent(retries={"output": n})` (there is no `output_retries` kwarg in this version); missing key raises `UserError` at model construction |
| glinker | 0.1.1 | `processor_registry.get("l1_gliner")` builds the NER layer alone; no DAG, no L2 database |
| gliner | 0.2.30 | texts longer than `max_len` are **truncated** with only a `UserWarning` (`gliner/data_processing/processor.py`) |
| kreuzberg | 4.10.4 | `extract_file_sync(file_path, mime_type=None, config=None)` |
| torch | 2.15.0+cpu | installed from the CPU index |

### GLinker: run only L1 (NER), no knowledge base
GLinker is an entity *linking* framework (L1 NER → L2 candidate DB → L3 linker → L0 aggregation). We only need
mentions, so we build the registered `l1_gliner` processor directly and skip the DAG.
Rejected: a full DAG with an empty L2 database (pointless config, and L0 expects L2/L3 outputs).

### OCR is optional: Kreuzberg bundles the Tesseract engine, not the language data
With kreuzberg 4.10.4 and no system Tesseract, OCR on `samples/news_article.png` fails with
`Failed loading language 'eng'`. The engine is compiled into the wheel, but `eng.traineddata` is not shipped.
After downloading `tessdata_fast/eng.traineddata` and setting `TESSDATA_PREFIX`, OCR returns the full text.
Since `uv sync` alone can't provide that file, the PNG/OCR path is documented as optional, and `extract` fails
with a clear message when it gets no text. Rejected: committing the 4 MB traineddata into the repo, or downloading
it at runtime (a hidden network side effect).

### Collapse whitespace, then chunk by sentence
Tesseract put a blank line *inside* a name ("Marta\n\nIbarra"), and PDF extraction breaks lines mid-sentence.
So before NER, all whitespace runs collapse to one space, and chunks are built from whole sentences. Mention
offsets therefore refer to this normalized text, not to the raw extraction. Rejected: splitting on blank lines
(this splits entities on OCR output).

### Chunk text before NER
GLiNER silently truncates long inputs (verified above), so a whole document passed as one string would lose every
mention past the first few hundred words. `ner.py` splits the text into chunks and keeps each chunk's offset.
