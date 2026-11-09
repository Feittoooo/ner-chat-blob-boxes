# Architecture of `ner_chat`

UML diagrams of `src/ner_chat`, in Mermaid (GitHub renders them). Notation:

| Notation | Meaning |
|---|---|
| `<<BaseModel>>` | Pydantic DTO, validated on construction |
| `<<module>>` | a `.py` file: module-level functions and variables (like a class with `static` members) |
| `<<external>>` | third-party library |
| `method$` | static: `@classmethod` or `@staticmethod` |
| `-member` | private by convention (leading `_`) |
| `Optional~T~` | `T \| None` |

## 1. Data model

`StoreFile` is the format of `entities.json`; `EntityStore` wraps it with an index by normalized name.
`AskResponse` is the structured output of `ask`; `EntityInfo`, `CountResult` and `ListResult` are what the tools
return to the LLM.

```mermaid
classDiagram
  direction LR
  class StoreFile {
    <<BaseModel>>
    +int schema_version
    +SourceInfo source
    +NerInfo ner
    +List~EntityRecord~ entities
  }
  class SourceInfo {
    <<BaseModel>>
    +Optional~str~ path
    +Optional~str~ sha256
    +Optional~str~ extracted_at
  }
  class NerInfo {
    <<BaseModel>>
    +Optional~str~ model
    +List~str~ labels
    +Optional~float~ threshold
  }
  class EntityRecord {
    <<BaseModel>>
    +str name
    +str label
    +int mention_count
    +List~Mention~ mentions
  }
  class Mention {
    <<BaseModel>>
    +int start
    +int end
  }
  class RawMention {
    <<BaseModel>>
    +str text
    +str label
    +int start
    +int end
  }
  class EntityStore {
    +StoreFile data
    -Dict~str,EntityRecord~ _by_key
    +load(path) EntityStore$
    +from_mentions(mentions, source, ner) EntityStore$
    +save(path) None
    +lookup(name) Optional~EntityRecord~
    +search(query, limit) List~EntityRecord~
    +top(label, limit) tuple
    +labels() Dict~str,int~
    -_sorted(records) List~EntityRecord~$
  }
  class AskResponse {
    <<BaseModel>>
    +str reply
    +List~EntityRef~ entities
  }
  class EntityRef {
    <<BaseModel>>
    +str name
    +str label
    +int mention_count
  }
  class EntityInfo {
    <<BaseModel>>
    +str name
    +str label
    +int mention_count
  }
  class CountResult {
    <<BaseModel>>
    +bool found
  }
  class ListResult {
    <<BaseModel>>
    +int total
    +int returned
    +List~EntityInfo~ items
  }
  StoreFile *-- "1" SourceInfo
  StoreFile *-- "1" NerInfo
  StoreFile *-- "0..*" EntityRecord
  EntityRecord *-- "0..*" Mention
  EntityStore o-- "1" StoreFile : data
  EntityStore ..> RawMention : from_mentions aggregates
  AskResponse *-- "0..10" EntityRef
  EntityInfo <|-- CountResult
  ListResult *-- "0..20" EntityInfo
```

## 2. Modules and dependencies

Dashed arrows mean "uses". `cli` imports `extract` and `ner` only inside the extract command, so `ask` never loads
GLinker or torch.

```mermaid
classDiagram
  direction TB
  class cli {
    <<module>>
    +main(argv) int
    +build_parser() ArgumentParser
    +cmd_extract(args) int
    +cmd_ask(args) int
  }
  class extract {
    <<module>>
    +extract_text(path, force_ocr) str
    ExtractError
  }
  class ner {
    <<module>>
    +MAX_CHUNK_WORDS = 200
    +prepare_text(raw) str
    +sentence_chunks(text, max_words) list
    +detect_entities(text, labels, threshold, model) List~RawMention~
  }
  class ner_defaults {
    <<module>>
    +DEFAULT_NER_MODEL
    +DEFAULT_LABELS
    +DEFAULT_THRESHOLD = 0.5
  }
  class models {
    <<module>>
    +DEFAULT_MODEL = ollama:qwen3
    +resolve_model(spec, ollama_base_url) Model
    +ollama_url(override) str
    ModelConfigError
  }
  class ask {
    <<module>>
    +run_ask(question, store, model) AskResponse
    -_last_cited_entities(messages) List~EntityRef~
  }
  class agent {
    <<module>>
    +Agent agent
    +count_mentions(ctx, name) CountResult
    +search_entities(ctx, query, limit) List~EntityInfo~
    +list_entities(ctx, label, limit) ListResult
    +list_labels(ctx) Dict~str,int~
    +counts_in_reply_match_store(ctx, output) AskResponse
    +numbers_near(reply, name) List~int~
  }
  class grounding {
    <<module>>
    +ground(response, store) AskResponse
    +deterministic_reply(entities) str
  }
  class normalize {
    <<module>>
    +normalize_name(name) str
  }
  class store {
    <<module>>
    EntityStore and the JSON DTOs
  }
  class schemas {
    <<module>>
    AskResponse, EntityRef
  }
  class Kreuzberg {
    <<external>>
  }
  class GLinker {
    <<external>>
  }
  class PydanticAI {
    <<external>>
  }
  cli ..> extract : extract
  cli ..> ner : extract, lazy import
  cli ..> ner_defaults : defaults
  cli ..> store
  cli ..> models : ask
  cli ..> ask : ask
  extract ..> Kreuzberg
  ner ..> GLinker : L1 layer only
  ner ..> ner_defaults
  ner ..> store : RawMention
  ask ..> agent
  ask ..> grounding
  ask ..> PydanticAI : capture_run_messages
  agent ..> PydanticAI : Agent, RunContext
  agent ..> store : ctx.deps
  agent ..> schemas
  models ..> PydanticAI : infer_model, OllamaModel
  grounding ..> store : lookup
  grounding ..> schemas
  grounding ..> normalize
  store ..> normalize
```

## 3. `extract` sequence

From file to JSON. Each mention's offset is added to its chunk's offset, so offsets refer to the whole normalized
text.

```mermaid
sequenceDiagram
  actor U as User
  participant CLI as cli.cmd_extract
  participant EX as extract
  participant K as Kreuzberg
  participant N as ner
  participant GL as GLinker L1 / GLiNER
  participant ST as EntityStore
  U->>CLI: ner-chat extract doc.pdf
  CLI->>EX: extract_text(path, force_ocr)
  EX->>K: extract_file_sync(path, config)
  K-->>EX: result.content
  EX-->>CLI: text, or ExtractError if there is none
  CLI->>N: prepare_text(text) collapses whitespace
  CLI->>N: detect_entities(text, labels, threshold)
  N->>N: sentence_chunks(text) gives (offset, chunk) pairs
  N->>GL: processor(texts=chunks)
  GL-->>N: L1Output.entities per chunk
  N-->>CLI: list[RawMention] with global offsets
  CLI->>ST: from_mentions(mentions, source, ner)
  ST->>ST: group by normalize_name, pick majority name and label
  CLI->>ST: save(entities.json)
  CLI-->>U: table of entities and counts
```

## 4. `ask` sequence

The LLM never receives the inventory in its prompt, only what tools return. Grounding is applied in code after the
run, on both the normal and the fallback path.

```mermaid
sequenceDiagram
  actor U as User
  participant CLI as cli.cmd_ask
  participant ST as EntityStore
  participant M as models
  participant A as ask.run_ask
  participant AG as agent / Pydantic AI
  participant LLM as LLM, Ollama or TestModel
  participant G as grounding
  U->>CLI: ner-chat ask "question"
  CLI->>ST: load(entities.json)
  CLI->>M: resolve_model(--model)
  M-->>CLI: Model, or ModelConfigError and exit 2
  CLI->>A: run_ask(question, store, model)
  A->>AG: run_sync(question, deps=store, model)
  AG->>LLM: instructions, question and tool schemas, no store data
  loop while the LLM calls tools
    LLM->>AG: tool call, e.g. count_mentions(name)
    AG->>ST: lookup, search, top or labels
    ST-->>AG: records
    AG->>LLM: tool result as JSON
  end
  LLM->>AG: final_result with AskResponse JSON
  AG->>AG: validate AskResponse and counts_in_reply_match_store
  alt valid answer
    AG-->>A: AskResponse
    A->>G: ground(response, store)
  else wrong digit next to a name and retries left
    AG->>LLM: ModelRetry with the correct count
    Note over AG,LLM: back to the loop, at most 2 retries
  else retries exhausted
    AG-->>A: UnexpectedModelBehavior
    A->>A: warning on stderr and entities of the last final_result
    A->>G: ground() and deterministic_reply()
  end
  G->>ST: lookup(name) for each cited entity
  G-->>A: AskResponse with store data, 0 if absent
  A-->>CLI: AskResponse
  CLI-->>U: reply and entities, or JSON with --json
```
