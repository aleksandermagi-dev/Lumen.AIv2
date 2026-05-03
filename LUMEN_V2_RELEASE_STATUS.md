# Lumen v2 Release Status

Status: Lumen v2 source release complete.

Lumen v2 is marked functionally complete as of the latest Phase 94 validation pass. The source release is published on GitHub and tagged as `v2.0.0`.

## GitHub Release State

- Repository: [aleksandermagi-dev/Lumen.AIv2](https://github.com/aleksandermagi-dev/Lumen.AIv2)
- Current tag: `v2.0.0`
- Tagged source commit: `1c89644`
- Latest pushed commit: see the GitHub repository history; documentation-only updates may be newer than the `v2.0.0` tag.
- Source branches: `main` and `master` are aligned.
- Binary asset: deferred intentionally. A fresh `dist/lumen.exe` can be built locally, but the exe is not uploaded or committed until binary/data hygiene is checked.

The current packaged target builds locally at:

```text
dist/lumen.exe
```

## Latest Validation Snapshot

- Source validation: `pass`
- Packaged validation: `pass`
- Source/package parity: `aligned`
- Blockers: `0`
- Regressions recorded in latest fast sweep: `0`
- ANH live MAST probe: deferred to the final release-candidate gate by choice, not because of a known failure

Latest reports:

- `Debug Logs/tmp_phase94_fast_sweep.md`
- `Debug Logs/tmp_phase94_fast_sweep.json`

## Release Scope

Lumen v2 includes:

- Local-first Windows desktop assistant shell
- Conversational chat with `default`, `collab`, and `direct` tone modes
- Curated local knowledge access across supported academic and technical domains
- Bounded symbolic math support
- Saved conversations, archives, memory, and archived memory surfaces
- Local SQLite-backed persistence
- Tool routing and supported local tool execution
- ANH archive/FITS intake and spectral dip scan capability
- Runtime diagnostics for validation, execution, and missing-input failures
- Source/package validation and sweep tooling

## Local Artifacts

The workspace intentionally keeps local artifacts on disk for QA and release work:

- `dist/` contains the locally rebuilt executable.
- `data/` contains local development/runtime validation state.
- `Debug Logs/` contains sweep reports and QA logs.
- `MAST file/` contains local ANH validation inputs.
- `New UI/` and `bug ref txt/` contain local QA references.

These are ignored by Git so the GitHub source tree stays clean, but they are not deleted or moved because they are still useful locally.

## Remaining Binary Release Gate

Before publishing a final binary asset, run the final full gate with the current MAST file:

```powershell
.\.venv\Scripts\python.exe -m lumen.validation.system_sweep --mode full --force-fresh-packaged --packaged-executable dist\lumen.exe --anh-probe "MAST file\MAST_2025-08-20T21_51_26.049Z.zip"
```

If that gate reports source `pass`, packaged `pass`, parity `aligned`, and blockers `0`, Lumen v2 is ready for binary release packaging.
