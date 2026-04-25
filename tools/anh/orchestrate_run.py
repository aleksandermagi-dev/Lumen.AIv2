#!/usr/bin/env python3
import argparse, os, json, shutil, traceback
from pathlib import Path

def is_good_file(p: Path):
    if p.is_dir():
        return False, "is_directory"
    try:
        with open(p, "rb") as f:
            chunk = f.read(256)
            if chunk is None:
                return False, "empty_read"
        ext = p.suffix.lower()
        if ext in {".csv", ".tsv", ".txt"}:
            with open(p, "r", encoding="utf-8", errors="strict") as f:
                f.readline()
        elif ext == ".json":
            with open(p, "r", encoding="utf-8", errors="strict") as f:
                json.load(f)
        return True, "ok"
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--good", required=True)
    ap.add_argument("--failed", required=True)
    ap.add_argument("--manifest_dir", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    good_dir = Path(args.good); good_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = Path(args.failed); failed_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir); manifest_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    failed = []

    for item in sorted(inp.iterdir() if inp.exists() else []):
        if item.is_dir():
            continue
        ok, reason = is_good_file(item)
        if ok:
            dest = good_dir / item.name
            try:
                shutil.copy2(item, dest)
                processed.append(item.name)
            except Exception as e:
                failed.append((item.name, f"CopyError: {e}"))
        else:
            dest = failed_dir / item.name
            try:
                shutil.copy2(item, dest)
            except Exception:
                pass
            failed.append((item.name, reason))

    (manifest_dir / "good_manifest.txt").write_text("\n".join(processed), encoding="utf-8")
    with open(manifest_dir / "failed_manifest.txt", "w", encoding="utf-8") as f:
        for name, reason in failed:
            f.write(f"{name}\t{reason}\n")

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(traceback.format_exc())
        raise
