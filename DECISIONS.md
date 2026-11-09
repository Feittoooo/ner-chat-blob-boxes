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
The edge model's `max_len` is 2048 words, so the sample would fit in one piece, but the 200-word chunks
cost nothing and protect against long documents and models with smaller windows.
Verified: the OCR output (with "Marta\n\nIbarra") gives exactly the same counts as the PDF.

### NER threshold 0.5
Measured on `samples/news_article.pdf` (7 expected entities):

| threshold | expected entities found, exact count | false positives |
|---|---|---|
| 0.3 | 6 | 3 ("port", "Local business groups", ...) |
| 0.4 | 6 | 2 |
| **0.5** | **6** | **0** |
| 0.6 | 5 (Valencia drops from 4 to 1) | 0 |

With the labels `person, organization, location`, "Spain" is missed at every threshold. The next decision fixes
it with the label set, not the threshold.

### Default labels add "country" (and not "city")
GLiNER is zero-shot: it scores spans against the label *names*, so label wording matters as much as the
threshold. Measured at threshold 0.5 on the same PDF:

| labels | Spain | other 6 entities | false positives |
|---|---|---|---|
| person, organization, location | missed | exact counts | 0 |
| **+ country** | **1 [country]** | identical counts and labels | 0 |
| + country + city | 1 [country] | same counts; Valencia and Rotterdam relabelled `city` | 0 |

`+ country` is the smallest change that finds Spain without side effects. `city` finds nothing new; it only
splits places between `location` and `city`, so `list_entities(label="location")` would stop returning
Valencia and Rotterdam. Caveat: this was tuned on a single short document, and `--labels` overrides it.

### JSON store, normalized key recomputed on load
JSON rather than SQLite: the store is a few hundred records at most, the test fixture `tests/entities.json` must be
hand-written, and a JSON file is readable in a diff. The lookup key is not saved. It is recomputed with
`normalize_name` on load, so a hand-written fixture can't carry a key that disagrees with the code. Records that
normalize to the same key merge (counts summed). There are no aliases or coreference: "Ibarra" is not
"Marta Ibarra".

### Grounding is a plain function after the run
`grounding.ground()` overwrites name/label/count of every cited entity with store values (count 0 /
"not mentioned" if absent) after `agent.run_sync` returns. It does not depend on the model reading its tool
results, so it holds for `TestModel`, which never does. Rejected: grounding inside an output validator.
That depends on model retries, and a pure function is deterministic and easy to test in isolation.

### Output validator: deliberately minimal
The structured fields are grounded in code, but `reply` is free text, so a model could write "mentioned 7 times"
next to a grounded count of 3. The validator does the simplest check that catches this:
- it takes the words within 3 of each cited entity name in `reply`;
- any word made only of digits that differs from that entity's stored count raises `ModelRetry` with the
  correct count (`retries={"output": 2}`).
Spelled-out numbers ("three") are not parsed, and there is no NLP. A year next to a name ("Valencia in 2029")
is a known false positive. It costs a retry and, at worst, the fallback below, which is still correct.
When retries run out, Pydantic AI raises `UnexpectedModelBehavior`. `run_ask` catches it, prints a warning
with the cause to stderr, and returns a deterministic reply built from the grounded entities of the last
output-tool call (recovered with `capture_run_messages`). There is no extra `needs_fallback` field: this one
safety net covers both cases. Rejected: a richer number/entity matcher. More code, and it would still only be
a defence for text whose structured counts are already guaranteed.

### Heavy imports only where needed
`ask` never imports glinker/torch (the CLI imports `ner` inside the extract command, and the NER defaults live in
the lightweight `ner_defaults.py`). `ask --model test` runs in ~1.5 s instead of loading a transformer stack it
doesn't use.

### Model selection
`--model` takes any Pydantic AI model string; the default is `ollama:qwen3` (free, local). `ollama:<name>` is
built explicitly with `OllamaProvider(base_url=--ollama-url | $OLLAMA_BASE_URL | http://localhost:11434/v1)`, so
it works without the env var. A missing key (`UserError`) or an unreachable Ollama (`ModelConnectionError` /
`ModelNotFoundError`) produces a clear message suggesting `--model test`, with exit code 2.
The pydantic-ai 2.66 first-run banner is disabled with `pydantic_ai.BANNER_ENABLED = False` because the CLI owns
its output.
