#!/usr/bin/env python3
"""Tests for the six gates. Pure Python; no GPU, no network, no harness."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TOOL = HERE / "tool_traces.py"


def run(*args, expect=0):
    r = subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)
    assert r.returncode == expect, f"exit {r.returncode} != {expect}\n{r.stdout}\n{r.stderr}"
    return r


def ok_run(**over):
    base = {
        "seed": "seed-00000",
        "task": "How many lines in data.txt mention alpha? Answer with the count.",
        "system": "you are a tool-using agent",
        "calls": [{"name": "grep", "args": {"pattern": "alpha", "path": "data.txt"}}],
        "results": [{"content": "data.txt:1:alpha"}],
        "answer": "Exactly one line mentions alpha: data.txt:1:alpha",
        "outcome": "done",
    }
    base.update(over)
    return base


def test_good_run_is_kept(tmp=Path("/tmp/traces-t1")):
    tmp.mkdir(exist_ok=True)
    (tmp / "runs.jsonl").write_text(json.dumps(ok_run()) + "\n")
    run("harvest", "--runs", str(tmp / "runs.jsonl"), "--out", str(tmp / "out.jsonl"))
    kept = [json.loads(l) for l in (tmp / "out.jsonl").read_text().splitlines()]
    assert len(kept) == 1 and kept[0]["answer"].startswith("Exactly one")


def test_unknown_tool_is_rejected(tmp=Path("/tmp/traces-t2")):
    tmp.mkdir(exist_ok=True)
    (tmp / "runs.jsonl").write_text(json.dumps(ok_run(calls=[{"name": "sudo_rm", "args": {}}])) + "\n")
    r = run("harvest", "--runs", str(tmp / "runs.jsonl"), "--out", str(tmp / "out.jsonl"), expect=2)
    assert "execution" in r.stdout, r.stdout


def test_ungrounded_answer_is_rejected(tmp=Path("/tmp/traces-t3")):
    tmp.mkdir(exist_ok=True)
    (tmp / "runs.jsonl").write_text(json.dumps(ok_run(answer="I believe it is four, from memory.")) + "\n")
    r = run("harvest", "--runs", str(tmp / "runs.jsonl"), "--out", str(tmp / "out.jsonl"), expect=2)
    assert "grounding" in r.stdout, r.stdout


def test_no_control_tool_is_rejected(tmp=Path("/tmp/traces-t4")):
    tmp.mkdir(exist_ok=True)
    (tmp / "runs.jsonl").write_text(json.dumps(ok_run(outcome=None)) + "\n")
    r = run("harvest", "--runs", str(tmp / "runs.jsonl"), "--out", str(tmp / "out.jsonl"), expect=2)
    assert "termination" in r.stdout, r.stdout


def test_repeated_failing_call_is_rejected(tmp=Path("/tmp/traces-t5")):
    tmp.mkdir(exist_ok=True)
    call = {"name": "grep", "args": {"pattern": "alpha", "path": "data.txt"}}
    r0 = ok_run(calls=[call, dict(call)],
                results=[{"content": "data.txt:1:alpha", "error": "no such file"},
                         {"content": "data.txt:1:alpha"}])
    (tmp / "runs.jsonl").write_text(json.dumps(r0) + "\n")
    r = run("harvest", "--runs", str(tmp / "runs.jsonl"), "--out", str(tmp / "out.jsonl"), expect=2)
    assert "recovery" in r.stdout, r.stdout


def test_leaked_answer_is_rejected(tmp=Path("/tmp/traces-t6")):
    tmp.mkdir(exist_ok=True)
    answer = "Exactly one line mentions alpha: data.txt:1:alpha"
    (tmp / "runs.jsonl").write_text(json.dumps(ok_run(task=f"The answer is {answer}. Confirm it.", answer=answer)) + "\n")
    r = run("harvest", "--runs", str(tmp / "runs.jsonl"), "--out", str(tmp / "out.jsonl"), expect=2)
    assert "leakage" in r.stdout, r.stdout


def test_duplicates_are_rejected(tmp=Path("/tmp/traces-t7")):
    # One of two identical runs survives; that is a 50% acceptance, so the
    # below-30% stop does not fire. The gate must still name itself.
    tmp.mkdir(exist_ok=True)
    row = json.dumps(ok_run())
    (tmp / "runs.jsonl").write_text(row + "\n" + row + "\n")
    r = run("harvest", "--runs", str(tmp / "runs.jsonl"), "--out", str(tmp / "out.jsonl"))
    assert "diversity" in r.stdout, r.stdout
    kept = [l for l in (tmp / "out.jsonl").read_text().splitlines() if l.strip()]
    assert len(kept) == 1, kept


def test_seeds_build_a_sandbox_with_real_files(tmp=Path("/tmp/traces-t8")):
    if tmp.exists():
        import shutil; shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    run("seeds", "--out", str(tmp / "seeds"), "--n", "6")
    seeds = [json.loads(l) for l in (tmp / "seeds" / "seeds.jsonl").read_text().splitlines()]
    assert len(seeds) == 6
    for s in seeds:
        d = Path(s["sandbox"])
        assert d.is_dir(), s
        assert any(d.iterdir()), f"empty sandbox {d}"
        assert s["task"], s
    tools = {s["tool"] for s in seeds}
    assert len(tools) >= 4, tools


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
