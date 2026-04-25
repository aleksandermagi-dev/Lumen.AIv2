# Lumen V2 Architecture

This document is the current high-level architecture map for Lumen v2.

For strict implementation authority, use [docs/runtime_architecture_authority.md](C:\Users\aleks\Desktop\lumen1.1\docs\runtime_architecture_authority.md). This file explains how the major pieces fit together; the authority document defines which layer is allowed to make runtime decisions.

## Current Architecture Summary

Lumen is a local-first desktop assistant and reasoning system. It combines a Windows desktop shell, CLI entrypoints, local persistence, curated knowledge, bounded memory, route authority, response realization, tool execution, and validation surfaces.

The core architecture is centered on:

- A single input -> NLU -> routing -> context assembly -> reasoning -> response -> persistence flow.
- `DomainRouter` as the route authority.
- `InteractionService` as the main turn orchestration layer.
- Tone modes as wording/personality profiles only.
- Memory and retrieval as bounded advisory context, not route authority.
- Tool execution as explicit capability execution.
- SQLite-first structured persistence with file-backed raw artifacts and guarded fallbacks.
- Offline evaluation, trainability traces, dataset curation, and scripted long-conversation QA.

## Runtime Flow

The normal turn flow is:

1. A prompt enters through the desktop app or CLI.
2. `PromptNLU` builds the canonical prompt-understanding contract.
3. `DomainRouter` selects the route lane.
4. Context assembly gathers bounded recent thread, project, memory, and local knowledge context.
5. Reasoning and support layers shape confidence, continuity, clarification, safety, and tool-use posture.
6. Response realization creates the user-facing answer in the selected tone mode.
7. Persistence records messages, sessions, interaction metadata, memory decisions, traces, tool runs, and diagnostics.

The route decision happens before memory and response shaping. Later layers may influence content and wording, but they must not silently re-route the turn.

## Desktop and CLI Surfaces

The desktop app is the primary user-facing surface.

It owns:

- Chat UI and visible thinking/responding state.
- Mode selection for `default`, `collab`, and `direct`.
- Saved conversations, archived conversations, memory, archived memory, and settings.
- Attachment handoff for files, folders, and zip inputs.
- Lazy construction of heavier surfaces so startup and navigation stay responsive.
- Source vs packaged runtime-path resolution.

The CLI exposes inspection, execution, validation, dataset, persistence, session, memory, archive, and ask/repl surfaces. CLI output formatting is report-facing and should not own live conversational reply composition.

## Prompt Understanding and Route Authority

`PromptNLU` is the canonical prompt-understanding builder. It owns normalized prompt shape, topic, intent, entities, language/structure signals, and advisory profile hints.

`DomainRouter` is the only route selector. It chooses the major lane, such as:

- conversation
- knowledge/research
- planning
- tool
- clarification
- safety

Recent routing work hardens the boundary between casual conversation and research. Social prompts, assistant-self prompts, and lightweight follow-ups should stay conversational unless there is clear knowledge, task, research, or tool intent.

## Context Assembly: Thread, Project, Memory, Knowledge

Context assembly is selective by design.

The assistant context can include:

- Recent conversation turns.
- Active thread summary and objective.
- Project id/name and recent project-scoped interactions.
- Current work-thread continuity signals.
- Bounded memory retrieval.
- Curated local knowledge entries and aliases.
- Tool/workspace continuity when relevant.

Current defaults:

- Live thread and project context outrank older cross-session memory.
- Personal memory is restrained during ordinary chat.
- Explicit memory recall prompts can surface personal memory more directly.
- Knowledge prompts should use local knowledge when coverage exists.
- Unknown or weakly covered prompts should keep the honest local-knowledge fallback.

## Conversation, Tone, and Response Realization

The visible tone modes are:

- `default`: calm, clear, lightly warm.
- `collab`: warmer, more partner-like, still disciplined.
- `direct`: concise, crisp, low-friction.

These modes change wording, not intelligence. They must not change route authority, knowledge access, safety posture, tool access, or reasoning depth.

Conversation support now includes:

- Casual greeting and check-in handling.
- Assistant-self answers such as "who are you" or "what about you".
- Long-chat continuity metadata.
- Repetition and follow-up-offer restraint.
- Soft pivot and return-to-thread handling.
- Work-thread follow-ups such as "what next", "keep going", and "summarize where we are".

Response realization must suppress internal scaffold wording in ordinary user-facing answers.

## Tool Execution and ANH

Tool execution is explicit capability execution, not passive retrieval.

The tool layer is manifest-backed and includes bundles such as:

- `anh`
- `astronomy`
- `content`
- `data`
- `design`
- `experiment`
- `invent`
- `knowledge`
- `math`
- `memory`
- `paper`
- `physics`
- `report`
- `simulate`
- `system`
- `viz`
- `workspace`

ANH is a tool-backed astronomy workflow for spectral dip scanning. "What is ANH?" belongs to local knowledge/explanation. "Run ANH" with a suitable input belongs to tool execution.

Tool results are archived and persisted with structured metadata; large raw artifacts remain file-backed.

## Persistence and Local Data Model

Lumen uses SQLite-first persistence for structured runtime state.

In source mode, the primary DB is normally:

```text
data/persistence/lumen.sqlite3
```

In packaged Windows mode, writable app data defaults to:

```text
%LOCALAPPDATA%/Lumen/data
```

SQLite is the primary structured source for:

- projects
- sessions
- messages
- session summaries
- memory items
- tool runs
- preferences
- trainability traces
- knowledge entries and aliases
- dataset import runs and examples

Files remain useful for:

- large tool/archive artifacts
- raw research-note hydration
- compatibility fallback
- rollback safety
- bundled examples and packaged resources

Structured reads should prefer SQLite. File reads should be explicit raw hydration or guarded compatibility fallback, not parallel structured authority.

## Evaluation, Datasets, and QA

Lumen includes local evaluation and dataset-curation surfaces.

Evaluation covers:

- route quality
- intent-domain quality
- memory relevance
- tool-use justification
- confidence calibration
- supervised-support recommendations
- long-conversation drift

Dataset support is local curation and export support. It can import, label, review, split, and export dataset examples, but it does not mean Lumen autonomously trains a model.

Scripted long-conversation evaluation now checks for:

- route drift
- tone drift
- memory over-injection
- scaffold leakage
- repetition/follow-up fatigue
- project anchoring
- work-thread continuity

## Safety and Authority Boundaries

Deterministic authority remains non-learned.

Advisory ML or future learned support may recommend labels, confidence, or provenance for bounded surfaces, but it must not override:

- route authority
- safety boundaries
- execution gating
- memory admission rules
- provider/runtime capability checks

Lumen should be explicit when a request exceeds local knowledge, configured providers, available tools, or safe support boundaries.

## Known Architectural Limits

The current architecture does not claim:

- universal local knowledge
- unrestricted live web research
- autonomous ML training
- learned replacement of route/safety/tool authority
- complete retirement of all legacy file fallback paths
- a full semantic retrieval overhaul beyond the current bounded memory-assist layer
- release-grade packaged 20-30 turn scripted desktop QA as a fully automated loop

These are intentionally treated as future work or release-QA extensions, not hidden completed capabilities.

## Relationship To Other Docs

- [README.md](C:\Users\aleks\Desktop\lumen1.1\README.md): app-reviewer-facing overview.
- [docs/runtime_architecture_authority.md](C:\Users\aleks\Desktop\lumen1.1\docs\runtime_architecture_authority.md): strict runtime authority and ownership rules.
- [docs/lumen_v2_authority_model.md](C:\Users\aleks\Desktop\lumen1.1\docs\lumen_v2_authority_model.md): authority-policy notes and deferred cleanup context.
- [docs/desktop_packaging.md](C:\Users\aleks\Desktop\lumen1.1\docs\desktop_packaging.md): packaged runtime path model and desktop packaging notes.
