# Formal Verification Research — AST Head, TreeSitter Oracle, Z3/Lean Integration

## Overview
Research for formal verification in code generation:
- Tree-sitter grammar integration
- AST-by-construction generation
- Z3/Lean proof checking
- Logit-level grammatical validity

## Tree-Sitter Integration

### Grammar-Aware Generation
Instead of flat text tokens, generate typed AST nodes:
```
Traditional: [token, token, token, ...]
AST-aware:   [FunctionDef(name=foo), Param(x), BinOp(+), Number(1), ...]
```

### TreeSitterOracle
Validates grammatical structure at logit level:
```rust
// Can't have CatchClause without TryStatement
match prev_node {
    Some(TryStatement) => allow!(CatchClause | FinallyClause),
    _ => block!(CatchClause),
}
```

### Tiered Validation
| Tier | Check | Latency | Use Case |
|------|-------|--------|---------|
| 1 | Per-token AST | ~1μs | Every token |
| 2 | Type check | ~100μs | Periodic |
| 3 | Z3 proof | ~1s | End of sequence |

## Z3 Integration

### WASM-Compiled Z3 (z3-wasm)
For fuel-limited execution in wasmi sandbox:
```javascript
// z3-wasm API
const z3 = await Z3.init();
const x = z3.int_const('x');
const theorem = z3.forall(x, x.add(x).eq(x.mul(2)));
```

### SAT/UNSAT Core Extraction
Targeted retroactive loss:
```rust
// Which tokens caused the contradiction?
let unsat_core = z3.unsat_core();
for (token_id, reason) in unsat_core {
    loss_gradient[token_id] += reason.contribution();
}
```

## Lean4 Integration

### Lean4 Kernel
Higher-order theorem proving:
```lean4
# Check type correctness
def foo (n : Nat) : Nat := n + 1
#check foo  -- Nat → Nat
```

### Metaprogramming
DSL for proof automation:
```lean4
macro "by_chain" $ tac => tac.run <| tactic.apply `allGoals
```

## AST Head Design

### Typed Node Operations
```rust
enum AstNode {
    FunctionDef { name: String, body: Expr },
    Param { name: String, ty: Type },
    BinOp { op: BinOp, lhs: Expr, rhs: Expr },
    // ...
}

enum Expr {
    Number(n: i64),
    Var(name: String),
    App(func: Expr, arg: Expr),
    Lambda { param: Param, body: Expr },
}
```

### Generation Head
```rust
struct AstHead {
    grammar: TreeSitterGrammar,
    vocab_size: usize,
}

impl AstHead {
    fn next_token(&self, state: &AstState) -> Vec<f32> {
        // Predict valid AST operations given current state
        let valid = self.grammar.valid_children(state.last_node());
        self.logits_for(valid)
    }
}
```

## References
- TreeSitter: https://tree-sitter.github.io/
- Z3: https://github.com/Z3Prover/z3
- Lean4: https://github.com/leanprover/lean4
- z3-wasm: https://github.com/xxuejie/z3-wasm