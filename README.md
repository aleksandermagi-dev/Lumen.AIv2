# Lumen

Lumen is a local-first desktop AI assistant, research partner, and engineering-support system for conversation, project reasoning, memory, and tool-backed workflows.

It is designed to feel like one coherent app: you can chat with it, ask it to explain things, continue a project thread, inspect saved conversations, use local memory, and route into supported tools without leaving the desktop shell.

Lumen is not presented as a frontier model trained from scratch. Its strength comes from a local orchestration stack: intent detection, routing, curated local knowledge, bounded memory, reasoning support, tool manifests, SQLite persistence, and validation surfaces that make its limits visible.

## Current Status

This repository is the Lumen v2 source release line. The source is published on GitHub at [aleksandermagi-dev/Lumen.AIv2](https://github.com/aleksandermagi-dev/Lumen.AIv2), tagged as `v2.0.0`, and marked functionally complete from the latest local validation pass.

The latest packaged target is:

```text
dist/lumen.exe
```

Current release status is tracked in [LUMEN_V2_RELEASE_STATUS.md](LUMEN_V2_RELEASE_STATUS.md).

The app currently supports:

- A Windows desktop chat shell.
- Local-first conversation and reasoning.
- Three tone modes: `default`, `collab`, and `direct`.
- Saved sessions, all conversations, archived chats, memory, archived memory, and settings.
- Local SQLite-backed persistence.
- Curated local knowledge and broad-topic answering where coverage exists.
- Research/tool routing for supported bundles.
- ANH spectral-analysis support when the runtime bundle and dependencies are available.
- Offline validation, decision evaluation, and scripted long-conversation QA foundations.

## What Lumen Does

Lumen can help with ordinary conversation, explanations, planning, research support, and local tool workflows.

Core user-facing capabilities include:

- Conversational chat that can handle greetings, follow-ups, self-questions, pivots, and longer back-and-forth sessions.
- General explanations across covered local knowledge domains such as astronomy, physics, chemistry, biology, earth science, engineering, history, math, science, and computing.
- Project-aware continuity when there is an active thread or recent project context.
- Local memory use that is intentionally bounded so old memories do not flood ordinary chat.
- Saved conversation restore, archive browsing, memory browsing, and settings persistence.
- Tool routing for supported local capabilities such as bounded symbolic math, data, knowledge graph operations, paper workflows, simulation, visualization, system inspection, and ANH.
- Offline evaluation of persisted interactions and scripted long-conversation behavior.

Lumen is built to answer when it has enough local support and to say when it does not.

## How Lumen Works

Lumen routes each turn through a local pipeline:

1. User input enters through the desktop app or CLI.
2. NLU and routing classify the request as conversation, knowledge, research, planning, memory, or tool intent.
3. Context assembly selects a bounded set of recent turns, active-thread state, project context, local knowledge, and relevant memory.
4. The reasoning and response layers choose the answer shape, confidence posture, and whether a tool should be used.
5. The tone layer applies `default`, `collab`, or `direct` wording without changing the underlying route, reasoning depth, or knowledge access.
6. Persistence saves messages, sessions, interactions, memory records, tool runs, and diagnostics locally.

The goal is not to make every prompt use every subsystem. The goal is to choose the smallest useful path for the turn.

## Desktop App

The desktop app is the main Lumen experience.

It includes:

- Chat with visible thinking/responding state.
- Mode selection for `default`, `collab`, and `direct`.
- Saved conversations and archived conversations.
- Memory and archived memory views.
- Popup-first settings.
- Theme and profile persistence.
- Attachment handoff for files, folders, and zip inputs.
- Lazy loading for heavier views so the shell does not eagerly load everything at startup.

In source mode, local app data defaults to the repository `data/` directory. In packaged Windows mode, writable desktop data defaults to:

```text
%LOCALAPPDATA%/Lumen/data
```

The packaged app separates read-only runtime resources from writable user data.

## Conversation, Tone, and Long-Chat Behavior

Lumen has three tone modes:

- `default`: calm, clear, and lightly warm.
- `collab`: warmer, more present, and more partner-like.
- `direct`: concise, crisp, and low-friction.

These modes are personality and wording profiles only. They do not make Lumen smarter or weaker. The same prompt should keep the same route, knowledge access, safety boundaries, and reasoning authority across all three modes.

Recent conversation work focuses on:

- Keeping casual chat conversational instead of over-triggering research.
- Handling self-referential prompts such as "who are you" or "what about you" conversationally.
- Reducing repeated closers and canned follow-up offers over longer chats.
- Anchoring "what next", "keep going", "summarize where we are", and similar prompts to the active work thread when one exists.
- Keeping memory helpful without over-injecting old personal details.

## Memory and Local Persistence

Lumen is local-first by default.

Structured runtime state is stored in SQLite and local files under the active data root. In source mode that is normally:

```text
data/
data/persistence/lumen.sqlite3
```

The persistence layer stores and indexes things such as:

- Sessions and messages.
- Interaction records.
- Session summaries.
- Memory items.
- Research notes and artifacts.
- Tool runs.
- Preferences and settings.
- Knowledge entries and aliases.
- Dataset curation records.

Memory is selective. Lumen does not blindly inject every old note into chat. Personal memory is used most strongly when the user asks for recall or when the current turn clearly benefits from it.

## Knowledge, Research, and Tools

Lumen has a local knowledge surface for covered topics and a tool-routing surface for supported workflows.

The current registry includes bundles such as:

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

Examples of supported capability areas include:

- Bounded symbolic math for common single-variable equations, plus symbolic/numerical helpers.
- Data description, correlation, regression, clustering, and visualization.
- Knowledge graph linking, paths, clustering, and contradiction checks.
- Paper search, summary, comparison, and method extraction.
- Simulation helpers for systems, orbits, populations, and diffusion.
- Visualization helpers for graphs, networks, timelines, and parameter spaces.
- Workspace inspection and system analysis.

Some capabilities are provider-gated or runtime-gated. Lumen should report those limits instead of pretending a missing provider or missing resource exists.

## ANH Capability

ANH is Lumen's astronomy-oriented spectral dip scan workflow.

In the current validated packaged runtime, ANH can analyze supplied FITS spectra when the required runtime resources and dependencies are present. It can report candidate spectral dips, including candidate velocity/depth information, and it is treated as a tool-backed analysis path rather than a generic chat answer.

The intended distinction is:

- "What is ANH?" should answer from local knowledge.
- "Run ANH on this FITS file" should route to the ANH tool path.

## Datasets and Evaluation

Lumen includes local evaluation and dataset-curation foundations.

It can:

- Evaluate persisted interactions offline.
- Review routing, memory relevance, tool-use decisions, confidence calibration, and response behavior.
- Export labeled examples for downstream review.
- Import JSON, JSONL, and mapped CSV datasets into SQLite.
- Track dataset batches, versions, splits, labels, corrections, and provenance.
- Run scripted long-conversation evaluation for local QA.

This does not mean Lumen trains a model by itself. Dataset support is for local curation, review, export, and evaluation workflows.

## Safety and Limits

Lumen is designed to be honest about capability boundaries.

It does not claim to provide:

- Medical diagnosis or treatment guidance.
- Legal, financial, or investment authority.
- Live news authority without a configured live source.
- Universal web research.
- Autonomous unrestricted automation.
- Fabrication-grade engineering signoff.
- Complete academic ghostwriting.
- Exhaustive local knowledge of every topic.
- A full computer algebra system for every mathematical form.

For unsupported or weakly grounded prompts, the desired behavior is to say that it does not have enough local knowledge or runtime support.

## What Lumen Is Not

Lumen is not:

- A cloud-hosted frontier model.
- A massive LLM trained from scratch.
- A replacement for professional judgment.
- A universal live-web agent.
- An autonomous ML training platform.
- A hidden black box with no local persistence trail.

It is a local desktop AI system that combines conversation, reasoning, memory, curated knowledge, tools, and validation into one app.

## Run Lumen

Install for development:

```bash
python -m pip install -e .[dev]
```

Launch the desktop app from source:

```bash
lumen-desktop
```

Run the packaged Windows app:

```text
dist/lumen.exe
```

The executable is built locally and intentionally not committed to the repository. A public binary release asset is deferred until final binary/data hygiene is checked.

## Build The Windows Executable

Lumen uses PyInstaller and the checked-in `main.spec` file to build the desktop executable.

From a fresh checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m PyInstaller main.spec
```

The rebuilt executable is written to:

```text
dist/lumen.exe
```

Before distributing a rebuilt executable, run at least a focused validation pass:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_interaction_service.py tests\unit\test_chat_ui_support.py tests\unit\test_persistence_manager.py -q
.\.venv\Scripts\python.exe -m lumen.validation.system_sweep --mode fast --force-fresh-packaged --packaged-executable dist\lumen.exe
```

For a final release-candidate gate, run the full sweep and include the selected ANH MAST probe when available:

```powershell
.\.venv\Scripts\python.exe -m lumen.validation.system_sweep --mode full --force-fresh-packaged --packaged-executable dist\lumen.exe --anh-probe "MAST file\MAST_2025-08-20T21_51_26.049Z.zip"
```

The `dist/`, `build/`, local `data/`, debug logs, and MAST files are intentionally ignored by Git. They are local build/runtime artifacts, not source files.

Initialize a workspace:

```bash
lumen init
python -m lumen init
```

Check local readiness:

```bash
lumen doctor
python -m lumen --format text doctor
```

Ask from the CLI:

```bash
lumen ask "hello"
lumen ask "tell me about space"
lumen ask "compare black holes and neutron stars"
lumen ask "solve 2x + 5 = 13"
```

Start the CLI REPL:

```bash
lumen
lumen repl
```

## Useful CLI Commands

Inspect bundles and capabilities:

```bash
lumen list-tools
lumen list-capabilities
lumen bundle inspect anh
lumen bundle inspect data
lumen bundle inspect knowledge
```

Inspect sessions, interactions, and persistence:

```bash
lumen interaction list
lumen interaction summary
lumen interaction evaluate
lumen persistence status
lumen persistence doctor
lumen session current default
lumen session inspect default
```

Inspect memory and research artifacts:

```bash
lumen memory notes
lumen memory artifacts
```

Curate and export datasets:

```bash
lumen dataset derive-runtime-dataset lumen_eval --strategy derived_trainability
lumen dataset sample-dataset-review --dataset-name lumen_eval
lumen dataset export-dataset-jsonl lumen_eval --split train --export-name lumen_eval_train
```

Execute a tool capability directly:

```bash
lumen run workspace.inspect_structure
```

## Validation and QA

The repository includes validation coverage for:

- Desktop shell behavior.
- Startup and packaged smoke checks.
- Input routing and reasoning.
- Tone-mode separation.
- Conversation, project, and memory continuity.
- Local knowledge routing.
- Tool and ANH routing.
- Persistence and saved-session behavior.
- Safety and refusal behavior.
- Dataset curation and export.
- Scripted long-conversation evaluation.

Recent local validation snapshots have included focused suites with hundreds of passing tests and successful packaged rebuilds. Treat those as local release-readiness evidence, not as a permanent guarantee. Any release candidate should still run the focused test slices, packaged smoke validation, and manual desktop QA before distribution.

Useful validation commands:

```bash
python -m pytest
python -m lumen.validation.system_sweep --mode full
python -m PyInstaller main.spec
```

## Known Limits

- Local knowledge is broad but not universal.
- Some content-generation behavior depends on provider configuration.
- Live external research requires configured sources or tools.
- Long-conversation quality is now evaluated locally, but real packaged 20-30 turn scripted sessions should still be run as release QA.
- Packaged behavior should be smoke-tested after rebuilds that touch runtime paths, desktop startup, tools, or persistence.
- The executable is large because it bundles local runtime dependencies and tool support.

## Bottom Line

Lumen v2 is a local-first desktop AI assistant with a real app shell, local persistence, memory, curated knowledge, tool routing, ANH support, evaluation surfaces, and explicit capability boundaries.

It is meant to be useful, inspectable, and honest: conversational when the user is chatting, tool-capable when a supported workflow is needed, and clear when local knowledge or runtime support is not enough.
