# Lumen Desktop Packaging

This note captures the current Lumen v2 desktop packaging model.

## Supported v2 Runtime Surface

The packaged desktop target is a Windows executable built with PyInstaller from `main.spec`.

The packaged runtime is expected to include the source-managed runtime resources needed by the desktop app:

- `tool_bundles/`
- `lumen.toml.example`
- runtime package code under `src/lumen`
- packaging hooks under `packaging_hooks/`
- bundled tool support, including ANH intake/scan support

ANH is part of the current v2 tool surface. "What is ANH?" should answer from local knowledge; "run ANH" with a valid FITS/MAST input should route to the ANH tool path.

## Runtime Path Model

Lumen desktop resolves two path roots at startup:

- `runtime_root`: read-only packaged resources and bundle manifests.
- `data_root`: writable user/app data.

Source/dev mode:

- `runtime_root` defaults to the repository root.
- `data_root` defaults to `<runtime_root>/data`.

Frozen/package mode:

- `runtime_root` resolves from the frozen app resource directory.
- `data_root` defaults to `%LOCALAPPDATA%\Lumen\data` on Windows.

The desktop entrypoint must not fall back to `Path.cwd()` as the runtime root.

## Building The Executable

From a fresh checkout:

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

`dist/`, `build/`, local data, debug logs, screenshots, and MAST files are intentionally ignored by Git.

## Validation Checklist

Before publishing a binary release asset, verify:

1. Launching outside the repo still discovers the active bundle set.
2. `doctor` reports the expected runtime root, data root, and bundle count.
3. The desktop app writes packaged user state under `%LOCALAPPDATA%\Lumen\data`.
4. Core prompts reach expected lanes:
   - conversation: `hey buddy`
   - knowledge: `tell me about space`
   - math: `solve x^4 - 10x^2 + 9 = 0`
   - ANH concept: `what is ANH`
   - ANH execution: `run ANH <FITS-or-MAST-path>`
5. Source/package parity is aligned.
6. Blockers are `0`.

Useful release-gate command:

```powershell
.\.venv\Scripts\python.exe -m lumen.validation.system_sweep --mode full --force-fresh-packaged --packaged-executable dist\lumen.exe --anh-probe "MAST file\MAST_2025-08-20T21_51_26.049Z.zip"
```
