# Lumen Desktop Packaging

This note captures the intended v1 desktop packaging layout and runtime path model.

## Supported v1 Runtime Surface

The first packaged desktop release supports these active bundles:

- `workspace`
- `report`
- `memory`
- `math`
- `system`
- `knowledge`

The legacy `anh` wrapper remains in the repo for future work and is not part of the v1 desktop packaging target.

## Runtime Path Model

Lumen desktop now resolves two separate path roots at startup:

- `runtime_root`
  - read-only app resources
  - bundle manifests and bundled examples live here
- `data_root`
  - writable user/app data
  - sessions, interactions, archive records, research notes, and tool runs live here

Source/dev mode:

- `runtime_root` defaults to the repo root
- `data_root` defaults to `<runtime_root>/data`

Frozen/package mode:

- `runtime_root` resolves from the frozen app resource directory
- `data_root` defaults to `%LOCALAPPDATA%/Lumen/data` on Windows

The desktop entrypoint must not fall back to `Path.cwd()` as the runtime root.

## Required Packaged Resources

The packaged app must include:

- `tool_bundles/`
- `data/examples/`
- `lumen.toml.example`

If required runtime resources are missing, startup diagnostics should surface that immediately instead of silently launching with an empty tool registry.

## Validation Checklist

Before shipping a packaged build, verify:

1. launching outside the repo still discovers the active bundle set
2. `doctor` reports the expected runtime root, data root, and bundle count
3. the desktop app does not write `data/` into the launcher working directory
4. core prompts reach the expected bundles:
   - math solve
   - system analysis
   - knowledge contradictions / links
