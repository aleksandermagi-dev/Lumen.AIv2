@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ================================================================
REM  CONFIG — edit these two lines if needed
REM ================================================================
set "DATA_DIR=.\data"
set "PY_ENTRY=main.py"
REM ================================================================

REM --- Allow overriding DATA_DIR via command-line arg:  run_analysis.bat "D:\path\to\data"
if not "%~1"=="" set "DATA_DIR=%~1"

REM --- Create a locale-safe timestamp (YYYY-MM-DD_HH-mm-ss)
for /f %%i in ('powershell -NoP -C "(Get-Date).ToString(\"yyyy-MM-dd_HH-mm-ss\")"') do set "TS=%%i"

REM --- Prepare results directory structure
set "RESULTS_ROOT=.\results"
set "RUN_DIR=%RESULTS_ROOT%\%TS%"
set "RUN_GOOD=%RUN_DIR%\good"
set "RUN_FAILED=%RUN_DIR%\failed"
mkdir "%RUN_DIR%" >nul 2>&1
mkdir "%RUN_GOOD%" >nul 2>&1
mkdir "%RUN_FAILED%" >nul 2>&1

REM --- Optional: activate a local venv if present (.\.venv\)
if exist ".\.venv\Scripts\activate.bat" (
    call ".\.venv\Scripts\activate.bat"
)

REM --- Snapshot raw inputs in manifest
if exist "%DATA_DIR%" (
  echo [INPUT LIST @ %DATE% %TIME%]>"%RUN_DIR%\inputs_manifest.txt"
  dir /b "%DATA_DIR%" >> "%RUN_DIR%\inputs_manifest.txt"
) else (
  echo [WARN] DATA_DIR not found: %DATA_DIR% > "%RUN_DIR%\inputs_manifest.txt"
)

REM --- Log header
echo [INFO] Starting job at %DATE% %TIME% > "%RUN_DIR%\run.log"
echo [INFO] Data dir: %DATA_DIR% >> "%RUN_DIR%\run.log"
echo [INFO] Results dir: %RUN_DIR% >> "%RUN_DIR%\run.log"

REM --- 1) Pre-sort inputs into GOOD vs FAILED using orchestrator.py
echo [INFO] Sorting inputs into GOOD and FAILED... >> "%RUN_DIR%\run.log"
python orchestrate_run.py --input "%DATA_DIR%" --good "%RUN_GOOD%" --failed "%RUN_FAILED%" --manifest_dir "%RUN_DIR%" >> "%RUN_DIR%\run.log" 2>&1

REM --- 2) Run your main Python job ONLY on GOOD files
echo [INFO] Running main job on GOOD files... >> "%RUN_DIR%\run.log"
set "PY_ARGS=--input "%RUN_GOOD%" --output ".""
echo [INFO] Command: python "%PY_ENTRY%" %PY_ARGS% >> "%RUN_DIR%\run.log"
python "%PY_ENTRY%" %PY_ARGS% >> "%RUN_DIR%\run.log" 2>&1

REM --- 3) Collect typical outputs into the timestamped results folder
for %%E in (png jpg jpeg pdf csv tsv txt json parquet html svg log) do (
  for %%F in (*.^%%E) do (
    move "%%F" "%RUN_DIR%" >nul 2>&1
  )
)

REM --- If your script writes to a fixed .\output folder, move it too
if exist ".\output" (
  if not exist "%RUN_DIR%\output" (
    move ".\output" "%RUN_DIR%\output" >nul 2>&1
  ) else (
    move ".\output" "%RUN_DIR%\output_%TS%" >nul 2>&1
  )
)

echo [DONE] Results saved to: %RUN_DIR%
echo [GOOD] %RUN_GOOD%
echo [FAILED] %RUN_FAILED%
endlocal
