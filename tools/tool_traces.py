#!/usr/bin/env python3
"""Generate tool-use trace seeds, and harvest traces that pass the six gates.

    python tools/tool_traces.py seeds --out /tmp/seeds --n 200 --corpus <dir>
    python tools/tool_traces.py harvest --runs runs.jsonl --out traces.jsonl

A trace is *observed* while a synthetic task is executed with real tools in a
disposable sandbox. Seeds are generated, never harvested from real runs: the
harness's own `dataset export` is metadata-only because real task text, prompts
and paths must never be published.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

# The tool surface a seed may ask for. Kept small on purpose: a trace that uses
# three tools well teaches more than one that uses ten badly.
TOOLS = {
    "read_file": {"path": "str"},
    "list_dir": {"path": "str"},
    "grep": {"pattern": "str", "path": "str"},
    "run_command": {"cmd": "str"},
    "write_file": {"path": "str", "content": "str"},
}

TEMPLATES = [
    ("read_file", "What does {path} contain? Quote the first line verbatim."),
    ("read_file", "How many lines in {path} mention {token}? Answer with the count."),
    ("list_dir", "List the files in {dir} and say which is largest."),
    ("grep", "Find every line containing {token} under {dir} and quote one."),
    ("run_command", "Run `wc -l {path}` in the sandbox and report the number."),
    ("write_file", "Create {dir}/answer.txt containing exactly the word {token}."),
]


def make_sandbox(root: Path, seed_id: str, corpus_files: list[Path]) -> dict:
    """A small tree with known content, so every task is answerable and checkable."""
    d = root / seed_id
    (d / "src").mkdir(parents=True, exist_ok=True)
    (d / "docs").mkdir(parents=True, exist_ok=True)
    taken = corpus_files[:2] if corpus_files else []
    if taken:
        for i, f in enumerate(taken):
            body = f.read_text(errors="replace")[:8000]
            (d / ("src" if i == 0 else "docs") / f.name).write_text(body)
    else:
        (d / "src" / "main.rs").write_text("fn main() {\n    println!(\"hello\");\n}\n")
        (d / "docs" / "notes.md").write_text("# notes\n\nalpha beta gamma\n")
    (d / "data.txt").write_text("alpha\nbeta\ngamma\ndelta\n")
    files = [p for p in d.rglob("*") if p.is_file()]
    return {"dir": str(d), "files": [str(p.relative_to(d)) for p in files],
            "tokens": ["alpha", "beta", "gamma", "delta"]}


def make_seeds(out: Path, n: int, corpus: Path | None) -> None:
    corpus_files = sorted(p for p in corpus.rglob("*") if p.is_file())[:40] if corpus else []
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        tool, templ = TEMPLATES[i % len(TEMPLATES)]
        sid = f"seed-{i:05d}"
        sb = make_sandbox(out, sid, corpus_files)
        target = sb["files"][i % max(1, len(sb["files"]))] if sb["files"] else "data.txt"
        token = sb["tokens"][i % len(sb["tokens"])]
        task = templ.format(path=target, dir="src" if "src" in target else "docs",
                            token=token)
        rows.append({"seed": sid, "tool": tool, "task": task, "sandbox": sb["dir"],
                     "allowed_tools": list(TOOLS), "template": templ})
    (out / "seeds.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} seeds to {out/'seeds.jsonl'}")


def gate_execution(run) -> str | None:
    for call in run.get("calls", []):
        name = call.get("name")
        if name not in TOOLS:
            return f"unknown tool {name!r}"
        for arg in call.get("args", {}):
            if arg not in TOOLS[name]:
                return f"tool {name} has no argument {arg!r}"
    return None


def gate_termination(run) -> str | None:
    if run.get("outcome") not in ("done", "failed", "escalate"):
        return f"no control tool: outcome={run.get('outcome')!r}"
    if not run.get("answer"):
        return "empty answer"
    return None


def gate_grounding(run) -> str | None:
    """The answer must reuse a value that a tool actually returned."""
    results = [r.get("content", "") for r in run.get("results", [])]
    if not results:
        return "no tool results; nothing can ground the answer"
    answer = run["answer"]
    # a 12-character window of the answer appearing in a result is evidence of use
    windows = [answer[i:i + 12] for i in range(0, max(1, len(answer) - 11))]
    if any(w and any(w in r for r in results) for w in windows):
        return None
    return "answer shares no span with any tool result"


def gate_recovery(run) -> str | None:
    calls, results = run.get("calls", []), run.get("results", [])
    for i, res in enumerate(results[:-1]):
        if res.get("error") and i + 1 < len(calls):
            same = calls[i].get("name") == calls[i + 1].get("name") and \
                   calls[i].get("args") == calls[i + 1].get("args")
            if same:
                return f"repeated the identical failing call after {res['error'][:30]!r}"
    return None


def gate_leakage(run) -> str | None:
    task, answer = run.get("task", ""), run.get("answer", "")
    if len(answer) > 24 and answer[:24] in task:
        return "the task contains its own answer"
    return None


def gate_diversity(run, seen: set) -> str | None:
    key = hashlib.sha256(
        (run.get("task", "") + "|" + ",".join(c.get("name", "") for c in run.get("calls", []))).encode()
    ).hexdigest()[:16]
    if key in seen:
        return "duplicate (task, tool sequence)"
    seen.add(key)
    return None


GATES = [("execution", gate_execution), ("termination", gate_termination),
         ("grounding", gate_grounding), ("recovery", gate_recovery),
         ("leakage", gate_leakage)]


def harvest(runs_path: Path, out_path: Path) -> None:
    seen, kept, reasons = set(), [], {}
    total = 0
    for line in runs_path.read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        run = json.loads(line)
        reason = None
        for name, gate in GATES:
            reason = gate(run)
            if reason:
                reasons[f"{name}: {reason.split(':')[0]}"] = reasons.get(f"{name}: {reason.split(':')[0]}", 0) + 1
                break
        if not reason:
            reason = gate_diversity(run, seen)
            if reason:
                reasons["diversity"] = reasons.get("diversity", 0) + 1
        if reason:
            continue
        kept.append({
            "system": run.get("system", ""), "user": run["task"],
            "calls": run["calls"], "results": run["results"],
            "answer": run["answer"], "outcome": run["outcome"],
            "seed": run.get("seed"), "sandbox_hash": run.get("sandbox_hash"),
        })
    out_path.write_text("\n".join(json.dumps(k) for k in kept) + ("\n" if kept else ""))
    rate = len(kept) / total if total else 0.0
    print(f"runs {total}, kept {len(kept)} ({rate:.0%}), rejected {total - len(kept)}")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5d}  {reason}")
    if total and rate < 0.30:
        print("STOP: acceptance below 30% — the seeds are wrong, not the model", file=sys.stderr)
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seeds"); s.add_argument("--out", required=True); s.add_argument("--n", type=int, default=200)
    s.add_argument("--corpus", default=None)
    h = sub.add_parser("harvest"); h.add_argument("--runs", required=True); h.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "seeds":
        make_seeds(Path(a.out), a.n, Path(a.corpus) if a.corpus else None)
    else:
        harvest(Path(a.runs), Path(a.out))


if __name__ == "__main__":
    main()
