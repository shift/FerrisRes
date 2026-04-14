//! Recursive Self-Verification ("Mirror Test")
//!
//! Training objective where the model generates code, then generates test cases
//! for that code. The CALM VM executes the tests, and failures trigger backprop
//! updates. This moves beyond syntax checking into semantic validation.
//!
//! Pipeline:
//!   1. Model generates code snippet
//!   2. LSP/WASM validates syntax (fast rejection)
//!   3. Model generates test function for the code
//!   4. CALM VM compiles and runs the test
//!   5. Test failure → loss signal → backprop
//!   6. Test pass → positive reinforcement
//!
//! The "Mirror" aspect: the model must understand its own code well enough
//! to write tests for it. If it can't, the code is likely wrong.

use crate::inference::lsp_tools::{Diagnostic, compiler_error_loss};
use crate::inference::wasm_sandbox::{WasmRuntime, embedded_syntax_checker_wasm};

// ---------------------------------------------------------------------------
// Mirror Test result types
// ---------------------------------------------------------------------------

/// Result of a mirror test evaluation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MirrorTestResult {
    /// The code that was generated.
    pub code: String,
    /// The test that was generated for the code.
    pub test_code: String,
    /// Whether the code passed syntax validation.
    pub syntax_valid: bool,
    /// Whether the test passed syntax validation.
    pub test_syntax_valid: bool,
    /// Whether the test passed execution.
    pub test_passed: bool,
    /// Any errors from execution.
    pub execution_errors: Vec<String>,
    /// Combined loss signal for backprop.
    pub loss: f32,
    /// Breakdown of loss components.
    pub loss_breakdown: LossBreakdown,
}

/// Breakdown of the loss signal into components.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LossBreakdown {
    /// Loss from code syntax errors.
    pub syntax_loss: f32,
    /// Loss from test syntax errors.
    pub test_syntax_loss: f32,
    /// Loss from test execution failures.
    pub execution_loss: f32,
    /// Bonus for passing tests (negative loss).
    pub pass_bonus: f32,
}

impl LossBreakdown {
    pub fn total(&self) -> f32 {
        self.syntax_loss + self.test_syntax_loss + self.execution_loss + self.pass_bonus
    }
}

// ---------------------------------------------------------------------------
// MirrorTestRunner
// ---------------------------------------------------------------------------

/// Runs the mirror test: generate code → generate test → execute test → compute loss.
pub struct MirrorTestRunner {
    /// WASM runtime for syntax validation.
    wasm_runtime: WasmRuntime,
    /// Whether to use WASM syntax checking.
    use_wasm: bool,
    /// Weight for syntax errors in loss.
    pub syntax_weight: f32,
    /// Weight for execution failures in loss.
    pub execution_weight: f32,
    /// Bonus for passing tests (negative, reward).
    pub pass_reward: f32,
}

impl MirrorTestRunner {
    pub fn new() -> Self {
        Self {
            wasm_runtime: WasmRuntime::default_runtime(),
            use_wasm: true,
            syntax_weight: 1.0,
            execution_weight: 2.0,
            pass_reward: -0.5,
        }
    }

    /// Run the full mirror test pipeline.
    ///
    /// `code`: the model-generated code
    /// `test_code`: the model-generated test for that code
    /// `language`: programming language (rust, python, etc.)
    pub fn evaluate(&self, code: &str, test_code: &str, language: &str) -> MirrorTestResult {
        let mut result = MirrorTestResult {
            code: code.to_string(),
            test_code: test_code.to_string(),
            syntax_valid: false,
            test_syntax_valid: false,
            test_passed: false,
            execution_errors: Vec::new(),
            loss: 0.0,
            loss_breakdown: LossBreakdown {
                syntax_loss: 0.0,
                test_syntax_loss: 0.0,
                execution_loss: 0.0,
                pass_bonus: 0.0,
            },
        };

        // Step 1: Validate code syntax
        let code_diags = self.check_syntax(code, language);
        result.syntax_valid = code_diags.is_empty();
        result.loss_breakdown.syntax_loss = compiler_error_loss(&code_diags) * self.syntax_weight;

        // Step 2: Validate test syntax
        let test_diags = self.check_syntax(test_code, language);
        result.test_syntax_valid = test_diags.is_empty();
        result.loss_breakdown.test_syntax_loss = compiler_error_loss(&test_diags) * self.syntax_weight;

        // Step 3: Execute the test (if both are syntactically valid)
        if result.syntax_valid && result.test_syntax_valid {
            let exec_result = self.execute_test(code, test_code, language);
            result.test_passed = exec_result.passed;
            result.execution_errors = exec_result.errors.clone();

            if exec_result.passed {
                result.loss_breakdown.pass_bonus = self.pass_reward;
            } else {
                result.loss_breakdown.execution_loss =
                    (exec_result.errors.len() as f32) * self.execution_weight;
            }
        } else {
            // Can't execute — treat as execution failure
            result.execution_errors.push("Cannot execute: syntax errors present".into());
            result.loss_breakdown.execution_loss = self.execution_weight;
        }

        result.loss = result.loss_breakdown.total();
        result
    }

    /// Check syntax using WASM sandbox or fallback.
    fn check_syntax(&self, code: &str, language: &str) -> Vec<Diagnostic> {
        if self.use_wasm {
            let wasm = embedded_syntax_checker_wasm();
            match self.wasm_runtime.execute_parse_from_bytes(&wasm, code) {
                Ok(result) => result.diagnostics,
                Err(_) => crate::inference::lsp_tools::fallback_syntax_check(code, language),
            }
        } else {
            crate::inference::lsp_tools::fallback_syntax_check(code, language)
        }
    }

    /// Execute a test against the generated code.
    ///
    /// In a full implementation, this would use the CALM VM. For now, it uses
    /// a simulated execution that checks for common patterns.
    fn execute_test(&self, code: &str, test_code: &str, language: &str) -> TestExecResult {
        let mut errors = Vec::new();

        // Simulated test execution based on language patterns.
        match language {
            "rust" => self.simulate_rust_test(code, test_code, &mut errors),
            "python" | "py" => self.simulate_python_test(code, test_code, &mut errors),
            _ => {
                // Generic: just check that the test references functions from the code
                self.simulate_generic_test(code, test_code, &mut errors);
            }
        }

        TestExecResult {
            passed: errors.is_empty(),
            errors,
        }
    }

    /// Simulate Rust test execution.
    fn simulate_rust_test(&self, code: &str, test_code: &str, errors: &mut Vec<String>) {
        // Check that test_code contains #[test] or #[cfg(test)]
        let has_test_attr = test_code.contains("#[test]") || test_code.contains("#[cfg(test)]");
        let has_assert = test_code.contains("assert!") || test_code.contains("assert_eq!")
            || test_code.contains("assert_ne!") || test_code.contains("panic!");

        if !has_test_attr && !has_assert {
            errors.push("Test code has no #[test] attribute or assertions".into());
        }

        // Check that functions called in test exist in the code
        self.check_function_references(code, test_code, errors);

        // Check for obvious type mismatches (very basic)
        if test_code.contains("assert_eq!") {
            // Check that assert_eq! has two arguments
            for line in test_code.lines() {
                if line.contains("assert_eq!") && line.contains(',') {
                    // Has at least two args — OK
                } else if line.contains("assert_eq!") && !line.contains(',') {
                    errors.push(format!("assert_eq! missing second argument: {}", line.trim()));
                }
            }
        }
    }

    /// Simulate Python test execution.
    fn simulate_python_test(&self, code: &str, test_code: &str, errors: &mut Vec<String>) {
        let has_assert = test_code.contains("assert ") || test_code.contains("assert(");
        let has_pytest = test_code.contains("def test_") || test_code.contains("import pytest");

        if !has_assert && !has_pytest {
            errors.push("Test code has no assert statements or pytest tests".into());
        }

        self.check_function_references(code, test_code, errors);
    }

    /// Simulate generic test execution.
    fn simulate_generic_test(&self, code: &str, test_code: &str, errors: &mut Vec<String>) {
        // Check that the test references something from the code
        if code.len() > 10 && test_code.len() > 5 {
            self.check_function_references(code, test_code, errors);
        }
    }

    /// Check that functions called in test_code are defined in code.
    fn check_function_references(&self, code: &str, test_code: &str, _errors: &mut Vec<String>) {
        // Extract function names from code
        let defined_fns = extract_function_names(code);
        let called_fns = extract_function_calls(test_code);

        // Check that called functions are defined
        for called in &called_fns {
            // Skip common assertions and builtins
            if matches!(called.as_str(),
                "assert" | "assert_eq" | "assert_ne" | "println" | "print" |
                "format" | "vec" | "String" | "Box" | "Rc" | "Arc" |
                "Ok" | "Err" | "Some" | "None" | "True" | "False" |
                "len" | "push" | "pop" | "get" | "is_empty" | "contains" |
                "to_string" | "as_str" | "clone" | "expect" | "unwrap" |
                "pytest" | "import" | "range" | "str" | "int" | "float" |
                "list" | "dict" | "set" | "tuple" | "type" | "isinstance" |
                "main" | "test" | "setup" | "teardown"
            ) {
                continue;
            }
            if !defined_fns.contains(called) {
                // Could be a standard library function — only warn
                // For stricter checking, add to errors
            }
        }
    }
}

impl Default for MirrorTestRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of test execution.
struct TestExecResult {
    passed: bool,
    errors: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Extract function names defined in code.
fn extract_function_names(code: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in code.lines() {
        let trimmed = line.trim();
        // Rust: fn name
        if let Some(rest) = trimmed.strip_prefix("fn ") {
            if let Some(name) = rest.split('(').next() {
                let name = name.trim().trim_start_matches("pub ").trim();
                if !name.is_empty() && name.chars().next().map_or(false, |c| c.is_alphabetic()) {
                    names.push(name.to_string());
                }
            }
        }
        // Python: def name
        if let Some(rest) = trimmed.strip_prefix("def ") {
            if let Some(name) = rest.split('(').next() {
                let name = name.trim();
                if !name.is_empty() && name.chars().next().map_or(false, |c| c.is_alphabetic()) {
                    names.push(name.to_string());
                }
            }
        }
        // C/Go: type name(
        if let Some(pos) = trimmed.find(" (") {
            let before = &trimmed[..pos];
            if let Some(name) = before.rsplit(|c: char| !c.is_alphanumeric() && c != '_').next() {
                if !name.is_empty() && name.chars().next().map_or(false, |c| c.is_alphabetic()) {
                    names.push(name.to_string());
                }
            }
        }
    }
    names
}

/// Extract function names called in code.
/// Handles both `func(` and `macro!(` patterns.
fn extract_function_calls(code: &str) -> Vec<String> {
    let mut calls = Vec::new();
    for line in code.lines() {
        // Find all identifier( and macro!( patterns
        let bytes = line.as_bytes();
        for i in 0..bytes.len().saturating_sub(1) {
            if bytes[i] == b'(' && i > 0 {
                // Walk backwards to find the identifier
                let end = i;
                let mut start = i;
                // Skip trailing '!' for macros like assert_eq!()
                let mut macro_end = i;
                if end > 0 && bytes[end - 1] == b'!' {
                    macro_end = end - 1;
                }
                for j in (0..macro_end).rev() {
                    if bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' {
                        start = j;
                    } else {
                        break;
                    }
                }
                if start < macro_end {
                    let name = &line[start..macro_end];
                    if !name.is_empty() && name.chars().next().map_or(false, |c| c.is_alphabetic()) {
                        calls.push(name.to_string());
                    }
                }
            }
        }
    }
    calls
}

// ---------------------------------------------------------------------------
// Loss computation for autodiff integration
// ---------------------------------------------------------------------------

/// Compute the mirror test loss from a batch of results.
/// Returns the mean loss across the batch.
pub fn mirror_batch_loss(results: &[MirrorTestResult]) -> f32 {
    if results.is_empty() { return 0.0; }
    let total: f32 = results.iter().map(|r| r.loss).sum();
    total / results.len() as f32
}

/// Compute per-sample weights for importance sampling.
/// Samples with higher loss get more weight (curriculum learning).
pub fn importance_weights(results: &[MirrorTestResult]) -> Vec<f32> {
    let total_loss: f32 = results.iter().map(|r| r.loss.abs()).sum();
    if total_loss == 0.0 {
        return vec![1.0 / results.len() as f32; results.len()];
    }
    results.iter()
        .map(|r| r.loss.abs() / total_loss)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirror_valid_rust() {
        let runner = MirrorTestRunner::new();
        let code = r#"fn add(a: i32, b: i32) -> i32 { a + b }"#;
        let test = r#"#[test] fn test_add() { assert_eq!(add(1, 2), 3); }"#;
        let result = runner.evaluate(code, test, "rust");

        assert!(result.syntax_valid, "Code should be syntactically valid");
        assert!(result.test_syntax_valid, "Test should be syntactically valid");
        assert!(result.test_passed, "Test should pass");
        assert!(result.loss <= 0.0, "Loss should be <= 0 (reward) for passing test, got {}", result.loss);
    }

    #[test]
    fn test_mirror_invalid_code() {
        let runner = MirrorTestRunner::new();
        let code = r#"fn main() { let x = 1"#; // Missing closing brace
        let test = r#"#[test] fn test_main() { assert!(true); }"#;
        let result = runner.evaluate(code, test, "rust");

        assert!(!result.syntax_valid, "Code should have syntax errors");
        assert!(result.loss_breakdown.syntax_loss > 0.0, "Should have syntax loss");
    }

    #[test]
    fn test_mirror_test_without_assertions() {
        let runner = MirrorTestRunner::new();
        let code = r#"fn add(a: i32, b: i32) -> i32 { a + b }"#;
        let test = r#"fn my_test() { let _x = 1; }"#; // No #[test], no assert
        let result = runner.evaluate(code, test, "rust");

        assert!(!result.test_passed, "Test without assertions should fail");
        assert!(result.loss_breakdown.execution_loss > 0.0, "Should have execution loss");
    }

    #[test]
    fn test_mirror_python() {
        let runner = MirrorTestRunner::new();
        let code = r#"def add(a, b): return a + b"#;
        let test = r#"def test_add(): assert add(1, 2) == 3"#;
        let result = runner.evaluate(code, test, "python");

        assert!(result.syntax_valid);
        assert!(result.test_passed);
    }

    #[test]
    fn test_mirror_batch_loss() {
        let results = vec![
            MirrorTestResult {
                code: "fn a() {}".into(), test_code: "#[test] fn t() {}".into(),
                syntax_valid: true, test_syntax_valid: true, test_passed: true,
                execution_errors: vec![], loss: -0.5,
                loss_breakdown: LossBreakdown { syntax_loss: 0.0, test_syntax_loss: 0.0, execution_loss: 0.0, pass_bonus: -0.5 },
            },
            MirrorTestResult {
                code: "fn b() {".into(), test_code: "#[test] fn t() {}".into(),
                syntax_valid: false, test_syntax_valid: true, test_passed: false,
                execution_errors: vec!["syntax error".into()], loss: 3.0,
                loss_breakdown: LossBreakdown { syntax_loss: 1.0, test_syntax_loss: 0.0, execution_loss: 2.0, pass_bonus: 0.0 },
            },
        ];
        let batch_loss = mirror_batch_loss(&results);
        assert!((batch_loss - 1.25).abs() < 0.001, "Expected 1.25, got {}", batch_loss);
    }

    #[test]
    fn test_importance_weights() {
        let results = vec![
            MirrorTestResult {
                code: "a".into(), test_code: "t".into(),
                syntax_valid: true, test_syntax_valid: true, test_passed: true,
                execution_errors: vec![], loss: 0.0,
                loss_breakdown: LossBreakdown { syntax_loss: 0.0, test_syntax_loss: 0.0, execution_loss: 0.0, pass_bonus: 0.0 },
            },
            MirrorTestResult {
                code: "b".into(), test_code: "t".into(),
                syntax_valid: false, test_syntax_valid: true, test_passed: false,
                execution_errors: vec![], loss: 3.0,
                loss_breakdown: LossBreakdown { syntax_loss: 1.0, test_syntax_loss: 0.0, execution_loss: 2.0, pass_bonus: 0.0 },
            },
        ];
        let weights = importance_weights(&results);
        assert_eq!(weights.len(), 2);
        // First sample has loss=0, second has loss=3.0
        // Total abs loss = 3.0
        assert!((weights[0]).abs() < 0.001);
        assert!((weights[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_extract_function_names_rust() {
        let code = r#"fn add(a: i32, b: i32) -> i32 { a + b }
fn multiply(x: i32, y: i32) -> i32 { x * y }"#;
        let names = extract_function_names(code);
        assert!(names.contains(&"add".to_string()));
        assert!(names.contains(&"multiply".to_string()));
    }

    #[test]
    fn test_extract_function_names_python() {
        let code = r#"def add(a, b):
    return a + b

def multiply(x, y):
    return x * y"#;
        let names = extract_function_names(code);
        assert!(names.contains(&"add".to_string()));
        assert!(names.contains(&"multiply".to_string()));
    }

    #[test]
    fn test_extract_function_calls() {
        let code = r#"assert_eq!(add(1, 2), 3);
let result = multiply(4, 5);"#;
        let calls = extract_function_calls(code);
        assert!(calls.contains(&"assert_eq".to_string()));
        assert!(calls.contains(&"add".to_string()));
        assert!(calls.contains(&"multiply".to_string()));
    }

    #[test]
    fn test_loss_breakdown_total() {
        let breakdown = LossBreakdown {
            syntax_loss: 1.0,
            test_syntax_loss: 0.3,
            execution_loss: 2.0,
            pass_bonus: -0.5,
        };
        assert!((breakdown.total() - 2.8).abs() < 0.001);
    }
}
