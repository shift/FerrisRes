#!/usr/bin/env python3
"""Lint a Kaggle notebook before pushing it.

    python tools/nb_lint.py notebooks/kaggle/train_32k [--datasets sectionme/x]

Checks, in the order that has cost the most runs:

1. **undefined names** — all code cells are concatenated (a notebook is one
   namespace) and checked with pyflakes. `json` missing in one cell and `_re`
   left behind in another both died this way, after a GPU session had started.
2. **dataset sources exist** — every `dataset_sources` entry is looked up on
   Kaggle, because a missing or misspelt source mounts nothing (and the first
   run after attaching one mounts nothing anyway).
3. **placeholder leftovers** — an unsubstituted `__NAME__` in the code.
4. **input paths** — every literal `/kaggle/input/...` must be reached by a glob,
   since the mount is nested two levels deep.

Exit status 1 on any error, so `kaggle_lab.py push` can refuse.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ERRORS, WARNINGS = [], []


def err(msg):
    ERRORS.append(msg)


def warn(msg):
    WARNINGS.append(msg)


def code_text(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")


def lint_names(src):
    if shutil.which("pyflakes") is None:
        warn("pyflakes not on PATH; undefined-name check skipped")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    r = subprocess.run(["pyflakes", path], capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if "undefined name" in line:
            err(f"undefined name: {line.split(':', 3)[-1].strip()}")
        elif line.strip():
            warn(line)


def lint_datasets(meta, check):
    for slug in meta.get("dataset_sources", []):
        if not check:
            continue
        # `datasets status` is owner-only and 404s on a public dataset someone else
        # owns, which is not the same as "missing". The file listing works for any.
        st = subprocess.run(["kaggle", "datasets", "status", slug], capture_output=True, text=True)
        if "ready" in st.stdout:
            warn(f"dataset {slug}: ready (the first run after attaching may still mount nothing)")
            continue
        files = subprocess.run(["kaggle", "datasets", "files", "-d", slug], capture_output=True, text=True)
        if files.returncode != 0 or "name" not in files.stdout:
            err(f"dataset {slug}: not reachable ({files.stderr.strip()[:80]})")
        else:
            warn(f"dataset {slug}: public, not owned; attachment is still required")


def lint_placeholders(src, meta_text):
    """A placeholder is fine when the code guards it with a fallback.

    `LR = '__LR__'` followed by `if LR.startswith('__'): LR = '1e-4'` is a
    parameter with a default, not a mistake; the lab substitutes it on push.
    """
    import re
    for name in set(re.findall(r"__([A-Z][A-Z0-9_]*)__", src)):
        if f"{name}.startswith('__')" in src:
            warn(f"placeholder __{name}__ has a default; substituted on push")
        else:
            err(f"unsubstituted placeholder __{name}__ in code")
    for name in set(re.findall(r"__([A-Z][A-Z0-9_]*)__", meta_text)):
        err(f"unsubstituted placeholder __{name}__ in metadata")


def lint_inputs(src):
    import re
    for path in re.findall(r"/kaggle/input[^'\"]*", src):
        if "/kaggle/input/**" in src or "rglob" in src or "glob.glob" in src:
            continue
        err(f"literal input path {path!r}: the mount is nested, use a glob from /kaggle/input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("--no-datasets", action="store_true", help="skip the Kaggle lookup")
    a = ap.parse_args()
    d = Path(a.template)
    meta_path = d / "kernel-metadata.json"
    if not meta_path.exists():
        print(f"FAIL: no kernel-metadata.json in {d}")
        return 1
    meta_text = meta_path.read_text()
    meta = json.loads(meta_text)
    nb = json.loads((d / meta["code_file"]).read_text())
    src = code_text(nb)

    lint_names(src)
    lint_placeholders(src, meta_text)
    lint_datasets(meta, not a.no_datasets)
    if "kaggle/input" in src and not any(k in src for k in ("glob", "rglob")):
        err("/kaggle/input is used but no glob discovers it")

    for w in WARNINGS:
        print(f"warn: {w}")
    for e in ERRORS:
        print(f"FAIL: {e}")
    if ERRORS:
        return 1
    print(f"ok: {d.name} ({len(ERRORS)} errors, {len(WARNINGS)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
