# Lumen System Sweep Status

This document summarizes the current v2 validation posture. For release status, also see [../LUMEN_V2_RELEASE_STATUS.md](../LUMEN_V2_RELEASE_STATUS.md).

## Current Snapshot

- Source release: published on GitHub at [aleksandermagi-dev/Lumen.AIv2](https://github.com/aleksandermagi-dev/Lumen.AIv2).
- Current tag: `v2.0.0`.
- Latest pushed commit at time of this note: `1c89644`.
- Latest fast source/package sweep: `pass`.
- Source/package parity: `aligned`.
- Blockers: `0`.
- Recorded regressions in latest fast sweep: `0`.
- ANH live MAST probe: deferred to final binary-release gate by choice.

Latest local reports are kept outside Git in `Debug Logs/`:

- `Debug Logs/tmp_phase94_fast_sweep.md`
- `Debug Logs/tmp_phase94_fast_sweep.json`

## Systems Verified Stable In The Latest v2 Pass

- Source full-system validation.
- Packaged smoke validation.
- UI and desktop shell regression coverage.
- Input, NLU, routing, reasoning, and interaction flow.
- Tone mode separation for `default`, `collab`, and `direct`.
- Local knowledge routing and broad-topic access.
- Tool routing and execution paths.
- Chat save/restore and recent-session behavior.
- Memory, archived memory, and persistence diagnostics.
- Safety and refusal behavior.
- Full validation contract and packaged smoke hook.

## Remaining Release Gate

The source release is complete. Before publishing a downloadable binary release asset, rerun the full sweep against a fresh executable and selected MAST input:

```powershell
.\.venv\Scripts\python.exe -m lumen.validation.system_sweep --mode full --force-fresh-packaged --packaged-executable dist\lumen.exe --anh-probe "MAST file\MAST_2025-08-20T21_51_26.049Z.zip"
```

Acceptance target:

- Source verdict: `pass`
- Packaged verdict: `pass`
- Parity verdict: `aligned`
- Blockers: `0`
- ANH zip/FITS intake reaches structured result or a specific actionable partial reason

## Historical Note

Older sweep documents and phase readouts in this repository record earlier implementation phases. They are useful audit history, but they are not the current runtime authority. Current status lives in this file, [../README.md](../README.md), [../LUMEN_V2_RELEASE_STATUS.md](../LUMEN_V2_RELEASE_STATUS.md), and [../LUMEN_V2_ARCHITECTURE.md](../LUMEN_V2_ARCHITECTURE.md).
