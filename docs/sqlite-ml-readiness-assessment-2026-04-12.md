# SQLite ML-Readiness Assessment

Date: 2026-04-12

Status update:
- Phase 25 added a native SQLite dataset-ingestion layer.
- Lumen now has first-class SQLite storage for:
  - dataset import batches and versions
  - canonical imported/curated dataset rows
  - split assignment and ingestion state
  - human/correction/canonical label history
- The assessment below remains useful as the grounding snapshot, but the “partially ready” gap identified here has now been closed by the new dataset-layer migration and ingestion service.

Database inspected:
- `C:\Users\aleks\Desktop\lumen1.1\data\persistence\lumen.sqlite3`

Live row counts inspected:
- `sessions`: 183
- `messages`: 4032
- `session_summaries`: 1506
- `tool_runs`: 634
- `memory_items`: 286
- `memory_item_embeddings`: 286
- `trainability_traces`: 1618

Important current-state note:
- Lumen already has a dataset export path in code via [dataset_exporter.py](/C:/Users/aleks/Desktop/lumen1.1/src/lumen/labeling/dataset_exporter.py).
- There is currently no live exported labeled dataset directory at `data/labeled_datasets`.
- Lumen is already capable of deriving labeled examples, and after Phase 25 it also has a first-class SQLite dataset-ingestion layer for imported and curated supervised examples.

## A. Current SQLite Assessment

### Current ML-relevant tables

#### `sessions`
- Purpose: session-level grouping and metadata for chat/tool interactions.
- Key fields: `id`, `project_id`, `title`, `mode`, `status`, `started_at`, `updated_at`, `summary_id`, `metadata_json`.
- ML value: good provenance root for grouping examples by session, mode, and task family.

#### `messages`
- Purpose: canonical turn-level interaction storage.
- Relationship: `messages.session_id -> sessions.id`.
- Key fields: `id`, `session_id`, `turn_key`, `role`, `content`, `intent_domain`, `confidence_tier`, `response_depth`, `conversation_phase`, `tool_usage_intent`, `route_decision_json`, `message_metadata_json`.
- ML value: strongest current anchor for supervised examples.
- Why it matters:
  - user/assistant pairs are already present
  - route metadata is already persisted
  - assistant rows can embed the full `interaction_record` in `message_metadata_json`

Representative live shape:
- user row includes prompt text like `run anh ...fits`
- assistant row includes summary text plus embedded `interaction_record`, extracted entities, interaction path, and routing details

#### `session_summaries`
- Purpose: condensed session state snapshots.
- Relationship: `session_summaries.session_id -> sessions.id`.
- Key fields: `summary_text`, `confidence_tier`, `summary_scope`, `metadata_json`.
- ML value: useful for summarization targets and session-level abstract generation, but secondary to `messages`.

#### `tool_runs`
- Purpose: structured tool execution history.
- Relationships:
  - `tool_runs.session_id -> sessions.id`
  - optional `tool_runs.message_id -> messages.id`
- Key fields: `tool_name`, `capability`, `success`, `output_summary`, `archive_path`, `run_dir`, `metadata_json`.
- ML value:
  - tool routing examples
  - tool outcome/evaluation examples
  - “should use tool / which tool / what happened” datasets

Live distribution highlights:
- `anh / spectral_dip_scan`: 218
- `math / solve_equation`: 131
- `content / generate_ideas`: 127
- smaller counts exist for `data`, `paper`, `invent`, `experiment`, `astronomy`, `knowledge`

#### `memory_items`
- Purpose: retrieval and memory-layer content, not primary supervised dataset storage.
- Relationship:
  - optional `memory_items.session_id -> sessions.id`
  - optional `memory_items.project_id -> projects.id`
- Key fields: `source_type`, `source_id`, `category`, `domain`, `content`, `confidence_tier`, `status`, `metadata_json`.
- ML value: can support retrieval-ranking or memory-write datasets, but should not be treated as the main imported-example table.

#### `memory_item_embeddings`
- Purpose: embedding storage attached to `memory_items`.
- Relationship: `memory_item_embeddings.memory_item_id -> memory_items.id`.
- Key fields: `source_id`, `source_type`, `model_name`, `embedding_blob`, `content_hash`, `status`.
- ML value: useful for retrieval experiments and ranking evaluation, not for canonical supervised ingestion.

#### `trainability_traces`
- Purpose: decision-trace and trainability/evaluation support.
- Relationships:
  - `trainability_traces.session_id -> sessions.id`
  - optional `trainability_traces.message_id -> messages.id`
- Key fields: `decision_type`, `input_context_summary`, `chosen_action`, `outcome`, `label`, `confidence_tier`, `model_assist_used`, `evaluation_score`, `metadata_json`.
- ML value: cleanest current table for decision-supervision datasets.

Live distribution highlights:
- `decision_type`: only `interaction_decision` currently
- `outcome` distribution:
  - `research`: 585
  - `tool`: 508
  - `conversation`: 210
  - `planning`: 165
  - `clarification`: 150
- `label`: currently null for all 1618 rows

Important live metadata already present:
- `available_training_surfaces`
  - `intent_domain_classification`
  - `route_recommendation_support`
  - `memory_relevance_ranking`
  - `tool_use_decision_support`
  - `response_style_selection`
  - `confidence_calibration_support`
- `deterministic_surfaces`
  - `system_invariants`
  - `safety_boundaries`
  - `hard_execution_constraints`

### What the current schema already supports well
- Turn-paired conversational examples from `messages`
- Decision classification examples from `trainability_traces`
- Tool routing and tool outcome examples from `messages + tool_runs`
- Summarization targets from `session_summaries`
- Retrieval/memory support examples from `memory_items`
- File-backed labeled example export via evaluation + labeling pipeline

### What the schema did not yet have at assessment time
- no dedicated SQLite table for imported training examples
- no dedicated SQLite table for dataset versions/import batches
- no dedicated SQLite table for human corrections or canonical labels
- no dedicated SQLite table for split assignment, dataset provenance, and ingestion state

### Current post-implementation state
- These gaps are now covered by:
  - `dataset_import_runs`
  - `dataset_examples`
  - `dataset_example_labels`

### Bottom line
- The original SQLite schema was already usable for dataset derivation and export.
- After Phase 25, Lumen is now natively ready for supervised dataset ingestion and curation in SQLite with a dedicated canonical dataset layer.
- The runtime/source-of-truth interaction tables still remain separate from imported/curated ML dataset storage.

## B. What Dataset Types Fit Right Now

### Fits now

#### `JSON`
- Best-supported current shape.
- Reason:
  - `message_metadata_json`, `route_decision_json`, and trace metadata are already JSON-heavy
  - exported labeled examples are already JSON

#### `JSONL`
- Best immediate supervised-ingestion format.
- Reason:
  - each example can be flattened from `messages`, `trainability_traces`, and optional `tool_runs`
  - preserves provenance cleanly without forcing a DB migration first

#### `QA pairs`
- Fits now.
- Source:
  - `messages` user rows + assistant rows by shared `turn_key` and `session_id`

#### `instruction/response`
- Fits now.
- Source:
  - same turn pairing from `messages`
  - assistant metadata can add route, mode, and provenance

#### `classification examples`
- Fits now and is the strongest first use case.
- Source:
  - `trainability_traces`
  - `messages.intent_domain`
  - route metadata
  - evaluation-driven labeled exports

### Fits partially

#### `CSV`
- Usable, but only for simpler projections.
- Good for:
  - intent classification
  - route outcome labels
  - coarse QA pairs
- Weak for:
  - nested provenance
  - tool metadata
  - trace/evaluation context

#### `reasoning/explanation pairs`
- Partially supported now.
- Source candidates:
  - `interaction_record.summary`
  - `pipeline_trace`
  - `trainability_trace`
  - route metadata
- Constraint:
  - should be treated as bounded explanation/provenance pairs
  - should not be treated as unrestricted chain-of-thought training material

### Originally did not fit cleanly without schema additions

#### canonical imported datasets inside SQLite
- Was not cleanly supported at assessment time.
- Reason:
  - the original DB stored interactions and traces, not imported curated datasets as first-class records

#### human label management and correction workflows
- Was not cleanly supported at assessment time.
- Reason:
  - `trainability_traces.label` exists but is currently unused
  - there was no dedicated human label or dataset import table

## C. What Schema Additions Are Needed, If Any

### If Lumen only needs immediate dataset export/use
- No SQLite schema change is strictly required.
- Lumen can derive examples now from:
  - `messages`
  - `trainability_traces`
  - `tool_runs`
  - `session_summaries`
- Recommended immediate path:
  - export normalized JSONL datasets outside SQLite first

### If Lumen needs persistent in-DB imported/curated datasets
- Minimum necessary addition:
  - `dataset_import_runs`
  - `dataset_examples`

### Minimum `dataset_import_runs` table
- `id`
- `dataset_name`
- `source_path`
- `source_format`
- `imported_at`
- `example_count`
- `schema_version`
- `notes_json`

Purpose:
- track provenance and versioning for each import

### Minimum `dataset_examples` table
- `id`
- `import_run_id`
- `example_type`
- `source_format`
- `split_assignment`
- `input_text`
- `target_text`
- `label_category`
- `label_value`
- `explanation_text` nullable
- `source_session_id` nullable
- `source_message_id` nullable
- `source_interaction_path` nullable
- `metadata_json`
- `created_at`

Purpose:
- canonical curated/imported supervised examples
- separate from live runtime interactions

### Where new dataset records should go

#### Keep here as-is
- source-of-truth runtime interactions:
  - `messages`
  - `trainability_traces`
  - `tool_runs`
- runtime memory/retrieval content:
  - `memory_items`
  - `memory_item_embeddings`

#### Keep here for current export artifacts
- `data/labeled_datasets`

#### Put new imported canonical dataset rows here
- `dataset_import_runs`
- `dataset_examples`

### What not to do
- do not stuff imported training rows into `messages`
- do not use `memory_items` as a training-example bucket
- do not overload `trainability_traces` into a full dataset registry

## D. Recommended First Dataset Strategy

### First examples to ingest
1. `trainability_traces` decision/classification examples
2. `messages` instruction/response examples
3. `messages + tool_runs` tool-routing and tool-outcome examples
4. bounded explanation/provenance examples only after the first three are working cleanly

### Best first task families
- route selection / tool-use decision support
- intent-domain classification
- response-style selection
- confidence calibration support

These are strongest because they already map directly to `available_training_surfaces` in live traces.

### Starting size
- Start with `250-1,000` examples.
- Prefer smaller, high-quality, provenance-clean data over bulk import.

### Fields that matter most
- `input_text`
- `target_text` or `label_value`
- `label_category`
- `example_type`
- `source_session_id`
- `source_message_id`
- `source_interaction_path`
- `split_assignment`
- `label_source`
- `evaluation_score`
- `created_at`
- deterministic-surface exclusions from trace metadata

### Recommended immediate normalized JSONL shapes

#### Decision-classification example
```json
{
  "example_type": "decision_classification",
  "input_text": "run anh C:\\Users\\aleks\\Desktop\\lumen1.1\\Proof\\HST\\COS\\lb6f07nrq_x1d.fits",
  "label_category": "route_outcome",
  "label_value": "tool",
  "chosen_action": "tool.command_alias",
  "decision_type": "interaction_decision",
  "source_session_id": "anh-path-routing-explicit-path",
  "source_message_id": "anh-path-routing-explicit-path:20260413T000621186750Z:assistant",
  "source_interaction_path": "C:\\Users\\aleks\\Desktop\\lumen1.1\\data\\interactions\\anh-path-routing-explicit-path\\20260413T000621186750Z.json",
  "metadata": {
    "available_training_surfaces": [
      "intent_domain_classification",
      "route_recommendation_support",
      "tool_use_decision_support"
    ],
    "deterministic_surfaces": [
      "system_invariants",
      "safety_boundaries",
      "hard_execution_constraints"
    ]
  }
}
```

#### Instruction/response example
```json
{
  "example_type": "instruction_response",
  "input_text": "run anh C:\\Users\\aleks\\Desktop\\lumen1.1\\Proof\\HST\\COS\\lb6f07nrq_x1d.fits",
  "target_text": "ANH analyzed 1 file(s) and found 1 candidate file(s). Strongest candidate: lb6f07nrq_x1d.fits at 43.357443248540456 km/s with depth 1.0.",
  "source_session_id": "anh-path-routing-explicit-path",
  "source_message_id": "anh-path-routing-explicit-path:20260413T000621186750Z:assistant",
  "metadata": {
    "intent_domain": "technical_engineering",
    "tool_usage_intent": "nlu_hint_alias",
    "conversation_phase": "execution"
  }
}
```

## E. Concrete Next Steps

### Immediate use with current DB
1. Use `messages` as the source for QA and instruction/response extraction.
2. Use `trainability_traces` as the source for decision/classification datasets.
3. Join `tool_runs` where needed for tool-routing and tool-outcome labeling.
4. Export the first pass as JSONL, not CSV.
5. Keep exported examples outside SQLite at first unless persistent in-DB curation is required.

### Minimal schema upgrade
1. Add `dataset_import_runs`.
2. Add `dataset_examples`.
3. Keep the new tables separate from runtime interaction tables.
4. Require provenance fields on every imported example.
5. Keep `label_source` distinct between:
   - programmatic evaluation
   - imported external data
   - future human curation

### First dataset import path
1. Extract `trainability_traces + linked assistant message` rows.
2. Normalize them into JSONL.
3. Manually review a first batch of `250-500`.
4. Add split assignment and task-family tags.
5. Only then, if persistent curation inside SQLite is needed, import those reviewed examples into `dataset_examples`.

## Hygiene and Drift Risks

### Main current risks
- Duplication if the same interaction becomes:
  - a runtime message
  - a JSON interaction file
  - an exported labeled example
  - an imported dataset row
  without stable provenance keys
- Training drift if programmatic evaluation labels are mixed with human labels without `label_source`
- Schema confusion if exported filesystem datasets and SQLite-ingested datasets are treated as one layer
- Overreach if `pipeline_trace` is treated as raw reasoning-target material instead of bounded observability/provenance
- Misuse of `memory_items` as a training corpus instead of a retrieval/memory subsystem

### Practical recommendation
- Treat runtime tables as source data.
- Treat filesystem JSON/JSONL as the first staging layer.
- Add only a minimal dedicated SQLite dataset layer when persistent in-DB curation becomes necessary.

## Final Assessment
- Lumen is already strong enough for supervised dataset derivation.
- The current DB does not need a rewrite.
- The cleanest next move is:
  - derive from `messages`, `trainability_traces`, and `tool_runs`
  - use JSONL first
  - add a minimal `dataset_import_runs + dataset_examples` layer only if persistent curated ingestion inside SQLite is required
