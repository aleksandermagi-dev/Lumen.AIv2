# Lumen v2 Release Notes

Lumen v2 is the current source release line for the GitHub repository [aleksandermagi-dev/Lumen.AIv2](https://github.com/aleksandermagi-dev/Lumen.AIv2).

## Release State

- Tag: `v2.0.0`
- Source release: published
- Binary asset: deferred until binary/data hygiene is checked
- Current local build target: `dist/lumen.exe`
- Validation posture: latest fast source/package sweep passed with parity aligned and blockers `0`

## What v2 Includes

- Windows desktop chat shell with saved conversations, archives, memory, archived memory, and settings.
- Local-first SQLite persistence for sessions, messages, memory, preferences, tool runs, knowledge, and evaluation data.
- Conversation handling for greetings, casual chat, acknowledgments, assistant-self prompts, long-chat continuity, and project/work-thread continuity.
- Tone modes: `default`, `collab`, and `direct` as wording/personality profiles only.
- Curated local knowledge across supported academic, technical, and cultural domains.
- Bounded symbolic math support for common single-variable equations and local math-tool diagnostics.
- Tool routing and execution for supported bundles, including data, visualization, paper, simulation, system, workspace, memory, math, and ANH.
- ANH archive/FITS intake and spectral dip scan support when valid inputs and runtime dependencies are present.
- Runtime diagnostics for missing input, validation failure, execution failure, dependency gaps, and unsupported operations.
- Local validation and evaluation surfaces, including full system sweep and long-conversation QA.

## What v2 Does Not Claim

- It is not a frontier model trained from scratch.
- It is not a universal live-web authority.
- It is not a professional medical, legal, financial, or engineering signoff tool.
- It is not an unrestricted automation system.
- It does not autonomously train models from datasets.
- It does not claim exhaustive knowledge of every topic.

## Building The Executable

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m PyInstaller main.spec
```

The executable is written to:

```text
dist/lumen.exe
```

## Final Binary Release Gate

Before uploading `lumen.exe` as a GitHub Release asset, run:

```powershell
.\.venv\Scripts\python.exe -m lumen.validation.system_sweep --mode full --force-fresh-packaged --packaged-executable dist\lumen.exe --anh-probe "MAST file\MAST_2025-08-20T21_51_26.049Z.zip"
```

The binary release should require source `pass`, packaged `pass`, parity `aligned`, and blockers `0`.
