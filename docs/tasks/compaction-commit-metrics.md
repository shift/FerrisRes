# Compaction distortion and evidence-gated commitment

Status: **task, ready to delegate.** Source: `research/hf-papers-2026-11-25-agentic.md`
(RECAP, RLVB). Repository: **agentic-loop** (`evals/`), not ferrisres.

## Behaviour (two metrics, one script each; one test each)

1. **Behaviour distortion of a compaction.** Given a recorded session with a
   full context and its compacted notebook, compare the next assistant action.
   The metric is a distance over the two action texts (exact match / normalised
   edit distance); a summary that changes the action is flagged.
2. **Commits with insufficient evidence.** Given a recorded session, flag an
   assistant final answer while the notebook contains an unresolved
   `open_question` or `memory_unresolved` item at that turn.

## Files

- `evals/compaction_distortion.py` and `evals/commit_evidence.py` (new, stdlib
  only; read the session JSONL and notebook entities).
- `evals/README.md` — how to run and where the fixtures live.

## Tests (name them)

- `compaction_distortion.py --selftest`: one fixture where the action is
  identical (pass) and one where a fact was dropped and the action differs (fail,
  with the dropped item named).
- `commit_evidence.py --selftest`: one clean session (pass) and one with an open
  question at answer time (fail, naming the question id).

## Verify

```
VERIFY: python3 evals/compaction_distortion.py --selftest && python3 evals/commit_evidence.py --selftest
```

## Out of scope

Training the behaviours (RECAP/RLVB), the eval runner, changing the notebooks.
These scripts read existing sessions; they do not run a model.

## Authoring rules

Fixtures are authored and small; no real transcripts. Push the branch; no merge.
