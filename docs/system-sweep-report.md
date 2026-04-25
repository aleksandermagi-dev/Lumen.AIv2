# Lumen Full System Sweep

- Schema version: `full_system_sweep.v1`
- Sweep runner: `python -m lumen.validation.system_sweep`
- Last completed full sweep timestamp: `2026-04-13T00:07:42.821347+00:00`
- Last completed full sweep audit mode: `full`
- Last completed full sweep audit data mode: `clean`
- Last completed full sweep audit data root: `data/validation/audit_workspaces/...`
- Last completed full sweep source verdict: `pass`
- Last completed full sweep packaged verdict: `pass`
- Last completed full sweep parity verdict: `aligned`
- Last completed full sweep report authority: `requested_path`
- Latest Phase 26 focused verification:
  - focused regression slices: `528 passed, 1 skipped`
  - full repo verification: `1061 passed, 2 skipped` on `pytest -q`
  - source full validation on isolated audit data: `pass`
  - direct packaged smoke validation: requested-path authority confirmed, boot `ok`, blockers `0`
  - refreshed `system_sweep --mode fast`: completed in about `42.6s`, with stale packaged parity correctly skipped instead of relaunching a known-stale EXE
  - rebuilt [lumen.exe](/C:/Users/aleks/Desktop/lumen1.1/dist/lumen.exe) and reran `system_sweep --mode fast --force-fresh-packaged`: completed in about `46.9s`, packaged verdict `pass`, parity `aligned`, requested-path authority confirmed

## Regressions Found

- No active product regressions were found in the Phase 26 code-side audit.
- No active QA/runtime regression remains after the packaged-smoke isolation fix and fresh rebuild verification.

## Root Causes Identified During This Pass

- No new product root cause was identified in Phase 26.
- The prior fast-sweep timeout came from two stacked costs:
  - relaunching a stale packaged executable during local reruns
  - keeping the heavy `tests/integration/test_tool_signal_ask_routing.py` integration path inside the fast routing slice
- The prior fresh packaged-smoke timeout came from validating against the live shared data root instead of an isolated audit workspace.

## Fixes Applied

- The fast local sweep now reuses a fresh packaged smoke artifact when possible and skips relaunching a known-stale packaged build.
- The heavy `tool_signal_ask_routing` integration coverage was kept in the broader validation surface instead of the fast routing slice.
- Hidden packaged smoke validation now uses an isolated clean audit data root instead of the live shared data directory.
- The pass was a confirmation audit after the SQLite dataset-layer addition.
- Standing docs were updated so supervised-ML readiness wording matches the current SQLite-backed dataset-ingestion and curation state.

## Systems Verified Stable

- Source full-system validation passed without blockers on isolated audit data.
- Direct packaged smoke validation completed without blockers against [lumen.exe](/C:/Users/aleks/Desktop/lumen1.1/dist/lumen.exe), with requested-path report authority and boot status `ok`.
- UI and desktop shell regression slice passed.
- Input, routing, reasoning, and interaction-flow regression slice passed.
- Chat state, memory, persistence, and diagnostics regression slice passed.
- Tool routing, execution, CLI, and safety regression slice passed.
- Academic-support and supervised-ML dataset-ingestion surfaces remained stable.
- Full validation contract and packaged smoke hook regression slice passed.
- Full repo suite passed in this pass: `1061 passed, 2 skipped`.

## Remaining Risks

- No active product release-blocking risks were found in the code-side audit.

## Weaknesses and Gaps

- No material runtime regressions remain open from this pass.
- Fast local sweep cost is now under control.
- Fresh packaged parity also returns quickly again after the packaged-smoke isolation fix.
- Provider-gated and not-promised capability surfaces remain bounded by design, not by regression:
  - hosted content-generation availability
  - live news and politics authority
  - investing guidance
  - health-risk inference
  - first-class audio and vision systems

## Intentionally Unchanged

- Harmless duplicate or overlapping capability surfaces were left in place when they did not create conflicting runtime behavior.
- No architecture-wide refactor was performed.
- The desktop UI architecture and current interaction design were preserved.

## Recommendations

### v2 Stabilization

- Keep `python -m lumen.validation.system_sweep --mode fast` as the default local audit path.
- Keep `python -m lumen.validation.system_sweep --mode full` as the release-grade audit path.
- Keep packaged smoke in the release checklist so requested-path authority remains continuously verified.
- Rebuild [lumen.exe](/C:/Users/aleks/Desktop/lumen1.1/dist/lumen.exe) before release-grade parity checks when source changes land, then run `system_sweep` with `--force-fresh-packaged`.

### v3 Improvements

- Add a lightweight scripted desktop smoke layer for navigation and state transitions if UI coverage needs to expand further.

## System Coherence

- Source runtime coherence: `yes`
- Packaged runtime coherence: `yes`
- Supervised ML dataset readiness: `yes, SQLite-backed dataset ingestion and curation for local/user-supplied datasets`
- Overall conclusion: Lumen currently behaves like one coherent system across source and packaged runtimes, and its supervised-ML support is now ready in the bounded SQLite-backed ingestion and curation sense rather than only advisory evaluation support.
