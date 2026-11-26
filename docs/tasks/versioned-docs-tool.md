# Versioned docs tool

Status: **design, 2026-11-26.** The fallback for docs the model has not seen:
answer from the running build, not from memory.

## Why

The docs change every release. A memorised flag is a wrong flag. The model must
be able to look up the command, flag, error or ADP operation **for the version
that is executing**, cite it, and say "not in these docs" when it is not.

## Tools

| tool | source | returns |
|---|---|---|
| `build_info` | the binary | version, git commit, build profile |
| `cli_help` | clap, at runtime | the exact help for a subcommand: flags, defaults, errors |
| `docs_search` | the docs installed with this build | top matching paths and headings, with a snippet |
| `docs_read` | the same | the section text, path, heading, version, commit |

`cli_help` is the strongest of the four because it cannot drift from the binary.
`docs_search`/`docs_read` cover the prose: quickstart, workflows, Telegram,
skills, ADP claims/leases/trust, the feature register.

## Contract

1. For any question about a command, flag, error or ADP operation, call the
   tool before answering.
2. Cite the path and heading, and name the version from `build_info`.
3. If the tool has no match, say so. Do not fill the gap from memory.
4. Never invent a flag or an ADP operation. A removed flag is answered from
   `cli_help`; if it is absent, it is absent.
5. The tools are **read-only** and scoped to the installed docs.

## Where the docs come from

- **Packaged run:** the `.deb` installs the user docs under
  `/usr/share/doc/ferrisres/` and the harness docs under its own data dir; the
  package records the git commit.
- **Source run:** the checkout's `docs/` and `README`, at the same commit as the
  binary.
- **Kaggle/Colab:** the package, so the docs match the `.deb`.

If no docs are installed, `docs_search` returns a typed
`not_available_for_version` result; `cli_help` and `build_info` still work.

## Index

Build a small index at package time: `(path, heading, first paragraph, hash)`
per doc. Regenerate on every release. The index is part of the artefact, so a
stale index is a packaging bug, not a runtime surprise.

## Security

Only publishable docs. The deploy repo's endpoint/credential docs, incident
logs, licence keys and security configuration are not installed and not
indexed. A doc that cannot be published is not shipped to the model.

## Training

Doc-QA pairs with citations teach the behaviour, and tool-use examples teach the
call. But the tool works with an untrained instruction-follower; training raises
the rate, it is not the mechanism. The gates are in `eval-plan.md` §8: no answer
about a command or ADP operation without a citation, and no stale-doc answer
from memory.

## Phases

1. `build_info` + `cli_help` — no docs asset, exact, immediately useful.
2. `docs_search`/`docs_read` over an installed index.
3. ADP docs and the feature register in the index.
4. A version-drift eval: a flag removed in this build must be answered from the
   tool or refused.

## Risks

- **Docs absent** — the typed fallback, never a guess.
- **Stale index** — regenerate in the package, and put the commit in the result.
- **Large docs** — search top-k sections, read one; never inject the whole tree.
- **Citation rot** — a path that moved; the result's commit is the citation's
  anchor.
