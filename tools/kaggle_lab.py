#!/usr/bin/env python3
"""Drive Kaggle experiments: push a parameterised notebook, collect the result.

    python tools/kaggle_lab.py push  --template notebooks/kaggle/train_32k \
        --kernel sectionme/ferrisres-lab-lr3e4 --set LR=3e-4 --set SMOKE=1
    python tools/kaggle_lab.py status sectionme/ferrisres-lab-lr3e4
    python tools/kaggle_lab.py collect sectionme/ferrisres-lab-lr3e4
    python tools/kaggle_lab.py table

`push` copies the template into a temporary kernel directory, substitutes every
`__NAME__` placeholder in the notebook and metadata, and pushes it. `collect`
waits for a finished kernel, downloads its output and appends the run's
`session.json` / `probe.json` / `manifest.json` to /tmp/opencode/results.jsonl,
so a sweep accumulates into one table instead of a dozen chat messages.

Requires KAGGLE_CONFIG_DIR and the kaggle CLI on PATH (nix shell
nixpkgs#python3Packages.kaggle).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RESULTS = Path("/tmp/opencode/results.jsonl")
DONE = {"KernelWorkerStatus.COMPLETE", "KernelWorkerStatus.ERROR"}


def kaggle(*args, **kw):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, **kw)


def status(slug):
    out = kaggle("kernels", "status", slug).stdout.strip().splitlines()
    return out[-1].split('"')[-2] if out else "UNKNOWN"


def push(template, slug, sets, title):
    src = Path(template)
    meta = json.loads((src / "kernel-metadata.json").read_text())
    nb_file = meta["code_file"]
    text = (src / nb_file).read_text()
    meta_text = json.dumps(meta)
    for key, value in sets.items():
        text = text.replace(f"__{key}__", value)
        meta_text = meta_text.replace(f"__{key}__", value)
    if "__" in text:
        leftover = sorted({t for t in text.split("__")[1::2] if t.isupper()})
        print(f"warning: unsubstituted placeholders: {leftover}")
    meta = json.loads(meta_text)
    meta["id"] = slug
    meta["title"] = title
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / nb_file).write_text(text)
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        r = kaggle("kernels", "push", "-p", str(d))
        print(r.stdout.strip() or r.stderr.strip())


def collect(slug, wait):
    deadline = time.time() + wait
    while True:
        s = status(slug)
        print(f"{slug}: {s}")
        if s in DONE or time.time() > deadline:
            break
        time.sleep(60)
    if s not in DONE:
        print("still running; not collected")
        return
    out = Path(tempfile.mkdtemp(prefix="kaggle-"))
    kaggle("kernels", "output", slug, "-p", str(out))
    record = {"kernel": slug, "status": s, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for name in ("session.json", "probe.json", "manifest.json"):
        hits = list(out.rglob(name))
        if hits:
            record[name.split(".")[0]] = json.loads(hits[0].read_text())
    with RESULTS.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "manifest"}, indent=2)[:1200])


def table():
    if not RESULTS.exists():
        print("no results yet")
        return
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    print(f"{'kernel':38s} {'status':10s} {'steps':>6s} {'init':>8s} {'final':>8s} {'s/step':>7s} {'tok/s':>7s}")
    for r in rows:
        s = r.get("session") or r.get("probe") or {}
        print(f"{r['kernel'][:38]:38s} {r['status'].split('.')[-1]:10s} "
              f"{s.get('steps','-'):>6} {s.get('initial_loss','-'):>8} {s.get('final_loss','-'):>8} "
              f"{s.get('step_time_s','-'):>7} {s.get('tokens_per_s','-'):>7}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push"); p.add_argument("--template", required=True); p.add_argument("--kernel", required=True)
    p.add_argument("--title", default="FerrisRes lab"); p.add_argument("--set", action="append", default=[])
    s = sub.add_parser("status"); s.add_argument("kernel")
    c = sub.add_parser("collect"); c.add_argument("kernel"); c.add_argument("--wait", type=int, default=5400)
    sub.add_parser("table")
    a = ap.parse_args()
    if a.cmd == "push":
        sets = dict(x.split("=", 1) for x in a.set)
        push(a.template, a.kernel, sets, a.title)
    elif a.cmd == "status":
        print(f"{a.kernel}: {status(a.kernel)}")
    elif a.cmd == "collect":
        collect(a.kernel, a.wait)
    else:
        table()


if __name__ == "__main__":
    main()
