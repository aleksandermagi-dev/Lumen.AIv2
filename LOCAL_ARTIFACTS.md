# Local Artifacts

This workspace keeps several local-only folders for QA, packaged runtime checks, screenshots, and science inputs. They are intentionally ignored by Git and should not be required for a clean source checkout.

## Kept Locally

- `dist/`: local packaged executable output.
- `build/`: PyInstaller build cache.
- `data/`: local runtime, validation, and development data.
- `Debug Logs/`: sweep reports, packaged smoke traces, and QA logs.
- `MAST file/`: local ANH/FITS validation inputs.
- `New UI/`: screenshot/reference material from UI QA.
- `bug ref txt/`: local bug-reference notes.
- `tmp_history_debug/`: temporary history/debug workspace.

## Source Of Truth For GitHub

The source tree needed to build and review Lumen is:

- `src/`
- `tests/`
- `tool_bundles/`
- `tools/`
- `packaging_hooks/`
- `docs/`
- `README.md`
- `LUMEN_V2_ARCHITECTURE.md`
- `LUMEN_V2_RELEASE_STATUS.md`
- `pyproject.toml`
- `main.spec`
- `lumen.toml.example`
- `.env.example`
- `lumen.ico`

Do not delete local artifact folders just to make the root look cleaner. The `.gitignore` keeps GitHub clean while preserving local QA history.
