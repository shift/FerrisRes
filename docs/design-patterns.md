# Design patterns in the corpus

Status: **training direction, 2026-11-26.** Design patterns belong in the weights
as **vocabulary and judgement**, not as a rule to apply everywhere. A model that
knows the names but not the forces produces ceremony; a model that knows the
forces can also say "this does not need a pattern".

## What is trained

| layer | content |
|---|---|
| **vocabulary** | name → intent → forces → structure → tradeoff |
| **recognition** | find the shape in real code, cite the file and line |
| **application** | a refactor that preserves behaviour and passes the tests |
| **critique** | name an over-engineered design and propose the simpler one |
| **decision** | an ADR that states the forces and the rejected alternative |

## The pattern set

### Object and module patterns (original summaries, not book text)

Strategy, adapter, facade, decorator, composite, observer, command, state,
visitor, template method, proxy, factory method, abstract factory, builder,
singleton (and why it is usually a dependency instead).

Each card carries: intent, forces, a Rust shape, the failure mode, and **when
not to use it**. The last field is the one that stops cargo-culting.

### Rust idioms

Newtype for units and invariants; RAII and `Drop` for cleanup; typestate for
protocol order; builder for many optional fields; trait objects vs generics;
extension traits; sealed traits; error enums with `thiserror` and `From`;
iterator chains over index loops; `Cow` and zero-copy; interior mutability and
when it is a smell; `Arc<Mutex>` vs `RwLock` vs channels; actor-style ownership;
async cancellation safety; `Pin` only where it is required; `#[non_exhaustive]`
for public enums; feature flags and additive compatibility; module and crate
boundaries as the real interface.

### Architecture

Ports and adapters (the project's own trait-first rule); dependency inversion;
layered vs hexagonal; event-driven with an outbox; CQRS and event sourcing with
their costs; saga and compensation; idempotency and retry with backoff; circuit
breaker; bulkhead; cache-aside with invalidation; backpressure; supervision tree;
control plane vs data plane; the anti-corruption layer.

### Nix

Module system with `mkIf`/`mkMerge`; overlays and overriding; flakes as pinned
interfaces; option types and defaults; fixed points; devShell composition;
package sets vs modules; the cost of a global overlay.

### Anti-patterns

God object; primitive obsession; stringly-typed APIs; boolean parameters;
shotgun surgery; leaky abstraction; error swallowing; global mutable state;
copy-paste inheritance; premature abstraction; abstraction with one
implementation and no test seam; a factory that returns one type.

## Data shape

A **pattern card** and its **before/after pair**:

```json
{
  "name": "strategy",
  "intent": "select an algorithm at run time without branching at every call site",
  "forces": ["multiple algorithms", "caller should not know which", "tests need a fake"],
  "rust": "trait Strategy { fn run(&self, x: &Input) -> Output; }",
  "when_not": "one implementation; a match on two variants is clearer",
  "failure": "a trait object per call hides a hot path and blocks inlining",
  "test": "the same behaviour tests pass for each implementation"
}
```

A refactor pair is accepted only when the **existing tests pass unchanged**; a
behaviour-preserving refactor with no test is a claim, not an example.

## Sources and licences

| source | use | rule |
|---|---|---|
| authored pattern cards | theory, Rust shapes, anti-patterns | ours, CC0 |
| authored before/after pairs | application and critique | ours, CC0; tests included |
| permissive OSS code | recognition examples, real shapes | per-file MIT/Apache/BSD; no verbatim book text |
| Rust API guidelines, clippy docs, the Book | idioms | MIT/Apache; quote sparingly, cite |
| nixpkgs, nix.dev | Nix patterns | MIT code; verify nix.dev's CC terms before quoting |
| our own repositories (agentic-loop, ferrisres) | trait-first, adapters, control plane | ours; provenance recorded; weights licence separate |

**Do not train on the GoF book text.** The ideas can be summarised in original
words; the expression cannot be copied. The same rule as the security curricula.

## Curriculum

1. **Vocabulary** — name, intent, forces, one Rust shape, one "when not".
2. **Recognition** — a real file, the pattern or its absence, cited lines.
3. **Application** — refactor with behaviour-preserving tests.
4. **Critique** — an over-engineered diff; propose the simpler design and say
   what it costs.
5. **Decision** — an ADR: context, forces, options, decision, consequence.

Patterns are introduced only after the model can read and test code; otherwise it
learns to decorate.

## Eval gates

| gate | pass |
|---|---|
| pattern named for a held-out snippet | >= 80% on the clear cases; confusion matrix reported |
| refactor preserves the existing tests | 100% of accepted pairs |
| **unnecessary pattern** introduced on a simple task | 0; a one-variant factory is a failure |
| "when not" explanation present | required for every application answer |
| critique of an over-engineered diff | names the simpler alternative and the cost |
| ADR completeness | forces, options, decision, consequence |
| compile/test at the patched revision | passes |

The anti-ceremony gate is the important one. A model trained only on pattern
examples will add an abstraction to every problem; the gate punishes exactly
that.

## Where it lives

- Corpus: `source-index.md` §11c; SFT and code mixes.
- Eval: `eval-plan.md` §20.
- Decision: FR-029 in `adr-index.md`.
- The project's own patterns are already the reference: trait-first, adapters,
  one control authority, evidence-bound results. The model should be able to
  explain why those are the patterns here, and when a simpler shape would do.
