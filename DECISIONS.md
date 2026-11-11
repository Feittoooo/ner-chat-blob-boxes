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
- it takes the 3 words **after** each cited entity name in `reply`;
- any word made only of digits that differs from that entity's stored count raises `ModelRetry` with the
  correct count (`retries={"output": 2}`).
Only words after the name are checked. The first version also looked at the 3 words before it, and with a real
model (`qwen3`) the common answer "Lumora Energy (5 mentions) and European Investment Bank (2 mentions)" was
rejected: the 5 right before the bank's name was taken as the bank's count. Every retry failed the same way and
the run ended in the fallback. Counts follow the name in English and Spanish ("X is mentioned 3 times",
"X (3 menciones)"); a count before the name ("3 mentions of X") is no longer checked. The structured counts are
grounded anyway, so a missed check costs less than rejecting a correct answer.
Spelled-out numbers ("three") are not parsed, and there is no NLP. A year right after a name ("Valencia in
2029") is a known false positive. It costs a retry and, at worst, the fallback below, which is still correct.
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

### Progress message and request timeout for `ask`
The first real run against Ollama looked like a hang: the first model load after installing (5 GB, with
2.9 GB of free RAM) failed after ~50 s with an HTTP 500 (Ollama's `server.log`), and the CLI printed nothing
while it waited. Two changes:
- `ask` prints `Asking <model>...` to stderr before the call, so `--json` on stdout stays clean.
- `--timeout` (default 120 s) sets `ModelSettings(request_timeout=...)`. It is one deadline over the whole
  request, provider SDK retries included, unlike `timeout`, which each retry re-arms. Without it the OpenAI
  client Ollama uses waits up to 600 s per attempt with 2 retries. When it runs out, `ModelRequestTimeout`
  becomes a clear error (with Ollama hints) and exit code 2.
`run_ask` takes `request_timeout` as an optional keyword (default none), so the offline test is unaffected.

## blob-boxes

### Dependencies
`numpy==2.5.4` (already locked through gliner/torch, now pinned directly), `opencv-python-headless==5.0.0.93`
(the headless build: no GUI windows are needed, so no GUI libraries are pulled in) and `scipy==1.18.1`.

### Verified: what `cv2.minAreaRect` returns in OpenCV 5.0.0
Rectangles of 60×20 px were drawn without OpenCV (a pixel is filled when its centre lies inside the rotated
rectangle), at known angles of the long side measured from +x towards +y (y grows downwards, so positive is
clockwise on screen). Their filled pixels went through `cv2.minAreaRect`:

| long side at | returned angle | returned (w, h) | `w` is the |
|---|---|---|---|
| 0° | −90.00 | (20.0, 60.0) | short side |
| 30° | −60.02 | (20.0, 60.0) | short side |
| 80° | −10.01 | (19.9, 59.9) | short side |
| 90° | −90.00 | (60.0, 20.0) | long side |
| 135° (= −45°) | −45.00 | (59.4, 19.8) | long side |
| −80° | −79.99 | (59.9, 19.9) | long side |

The returned angle is always in [−90, 0), and which side is called `w` flips with the orientation. Converting
to this project's convention (width = long side, angle of the long side in [−90, 90)):
- if `w < h`: width = h, height = w, angle = returned angle + 90;
- otherwise: width = w, height = h, angle = returned angle.

The CSV therefore never exposes OpenCV's convention, and `minAreaRect` is used only as a starting point.

### 8-connectivity
Pixels that touch only at a corner belong to the same blob. A 1-pixel-wide diagonal line is drawn exactly that
way, and with 4-connectivity the sample mask's line falls apart: 53 blobs instead of 8. On a larger 2048×2048
test mask (147 blobs) both give the same count, so the choice matters only for thin diagonal shapes.

### Stable blob ids: the SAUF labelling algorithm
Blob ids are OpenCV's labels, which follow reading order (row by row) with `cv2.CCL_SAUF`. OpenCV's default
algorithm does not promise that order and, run in parallel on the 2048×2048 mask, it broke it. SAUF keeps the ids
stable across machines and CPU counts.

### Pixel-centre IoU, nothing outside the image
Pixel (row r, column c) has its centre at x = c, y = r. A rectangle covers a pixel when that centre lies inside
it, edges included (with a 1e-9 tolerance, so floating-point rounding of the rotation cannot flip a centre that
is exactly on an edge; the sample generator uses the same rule, which is why its rectangles score exactly 1).
Rejected: the exact overlapping area of each pixel square. It is costlier and scores a box against a shape the
mask does not have: the mask is pixels, so the box is judged as pixels.

Pixels of the rectangle that fall outside the image do not count. A rectangle cut by the border then scores 1
with its full-size box; counting the outside part would drop that sample to 0.70 and push the box to shrink
into the image, which misdescribes the blob. Several boxes can then tie (any extension past the border); the
search keeps the first it finds.

### Search: smooth IoU, then exact IoU
The pixel IoU moves in jumps: shifting an 8×2 bar's box changes nothing until 0.49 px, then drops from 1.0 to
0.78. A local optimizer sees flat ground and stops. So the search runs in two phases:
1. **Smooth IoU** (`soft_iou`): each pixel counts by the share of it inside the rectangle along the rectangle's
   axes, a ramp from 0 to 1 over one pixel, half at the edge. Exact for axis-aligned boxes, and a perfect fit
   scores 1. On 300 random boxes around a 60×20 blob it tracks the exact IoU with correlation 0.9998 (mean gap
   0.0014) and picks the same best box. Nelder-Mead (derivative-free, robust to the remaining kinks) maximizes
   it over (cx, cy, width, height, angle). A first version used a sigmoid per pixel; it scored a perfect 8×2 fit
   at 0.49, because on thin shapes every pixel sits close to an edge, so it was dropped.
2. **Exact IoU**: a pattern search (± a step on each parameter, steps shrinking from 1 px / 2° to 0.05 px / 0.1°,
   strict improvements only) polishes every candidate on the real objective.

Starting points: OpenCV's `minAreaRect` (raw, and grown by 1 px per side since it hugs pixel centres), the
tightest box along the principal axes, and the box with the blob's spread (a filled w×h rectangle has variance
w²/12 along its width), which cuts through ragged ends such as an ellipse's. The raw `minAreaRect` is kept
because on a staircase line its sub-pixel thickness scores higher: without it, the sample line scored 0.79,
below `minAreaRect`'s 0.89. The highest exact IoU wins, ties to the earliest, so the result is never below
`minAreaRect` (checked on every blob of both masks).

Small blobs have many local optima: polishing every candidate instead of only the best improved 85 of the
147 blobs, by up to 0.08. Then random restarts from the winner (nudge, polish, keep strict improvements):

| restarts | mean IoU (147 blobs) | extra time |
|---|---|---|
| 0 | 0.940 | — |
| 10 | 0.947 | 8 s |
| 20 | 0.948 | 14 s |
| 40 | 0.951 | 27 s |

30 is the trade-off (~20 s for the whole mask). The random generator has a fixed seed and is created per blob, so
the output is identical on every run and a blob's box does not depend on the others. Raw `minAreaRect` averages
0.833 on that mask; the fit averages 0.949.

### Width, height and angle convention
`width` is the long side and `angle` its direction, from +x towards +y (clockwise on screen), normalized to
[-90, 90): a rectangle and the same one turned 180° are identical. A square is identical every 90°, so its angle
goes to [-45, 45). Normalizing never changes the covered pixels (checked on 2000 random boxes).

### Rounding is part of the fit
The search tends to end with an edge exactly on a row of pixel centres (that is where the IoU jumps). Rounding
the CSV to 4 decimals then moved some edges by 0.00005 px and dropped that row: two sample rectangles fell from
1.0 to 0.99. The fit therefore rounds the box itself, also trying it grown by 0.001 px, and reports the IoU of the
rounded box, so recomputing the IoU from the CSV gives exactly the reported value.

### Edge cases
- **1-pixel blob**: a 1×1 box at the pixel, IoU 1 (the search is skipped once a start scores 1).
- **Thin lines**: the box may be thinner than a pixel (0.80 px for the sample's staircase line, IoU 1), since
  only pixel centres count.
- **Border blobs**: see "nothing outside the image" above.
- **Squares**: angle in [-45, 45), as above.
