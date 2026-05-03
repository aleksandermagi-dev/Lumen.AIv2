# Phase 27: Supervised Dataset Curation, Export, and Evaluation Loop

> Historical note: this document records an earlier implementation/audit phase and is not the current runtime authority. For current status, see [README.md](../README.md), [LUMEN_V2_RELEASE_STATUS.md](../LUMEN_V2_RELEASE_STATUS.md), and [LUMEN_V2_ARCHITECTURE.md](../LUMEN_V2_ARCHITECTURE.md).

Status: implemented

## What was added

- SQLite-backed dataset review and curation workflow on top of:
  - `dataset_import_runs`
  - `dataset_examples`
  - `dataset_example_labels`
- CLI dataset workflow under `lumen dataset ...`
- JSONL export for curated train/validation/test and evaluation-only exports
- review-batch sampling for human curation
- canonical-label and correction-label update flow
- dataset-run comparison for overlap and balance inspection
- export quality checks:
  - empty input/target detection
  - duplicate-group detection
  - split-leakage warnings
  - missing-canonical-label warnings for canonical-only exports

## Main commands

- `lumen dataset derive-runtime-dataset <dataset_name> --strategy <derived_trainability|derived_instruction_response|derived_tool_runs>`
- `lumen dataset import-dataset <dataset_name> --source-format <json|jsonl|csv> --dataset-kind <...> --source-path <path>`
- `lumen dataset sample-dataset-review --dataset-name <name>`
- `lumen dataset label-dataset-example <dataset_example_id> --label-role <...> --label-value <...>`
- `lumen dataset export-dataset-jsonl <dataset_name>`
- `lumen dataset compare-dataset-runs <left_import_run_id> <right_import_run_id>`

## Intended first dataset families

- `lumen_route_v1`
  - primary source: `trainability_traces`
  - type: `classification_examples`
- `lumen_conversation_v1`
  - primary source: paired `messages`
  - type: `instruction_response`
- `lumen_academic_v1`
  - primary source: curated external JSON/JSONL plus selected runtime examples
  - types: `instruction_response`, bounded `classification_examples`, bounded `reasoning_explanation_pairs`

## Boundaries

- Runtime interaction tables remain the operational source-of-truth.
- Imported or curated training data does not get written back into:
  - `messages`
  - `memory_items`
  - `trainability_traces`
- This phase prepares datasets for external model training; it does not add in-app training orchestration or ML ops.

## Validation snapshot

- focused dataset + CLI + validation slices passed during implementation
- dataset workflow coverage includes:
  - import
  - review sampling
  - canonical labeling
  - JSONL export
  - dataset-run comparison
