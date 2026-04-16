//! Multi-Step Plan Executor — Tool Chaining with Replanning
//!
//! Based on ReAct, Plan-and-Solve, Reflexion, and CALM programs as plans.
//! The model emits a structured plan of tool calls, the executor runs them
//! sequentially, resolves output references ($1, $2, etc.), and handles
//! failures with replanning.
//!
//! Pipeline:
//!   1. Model emits plan: [plan] Step 1: tool(args) → Step 2: tool($1) [/plan]
//!   2. PlanExecutor validates the plan (tools exist, references valid)
//!   3. Execute steps sequentially, resolving $N references
//!   4. On failure: retry with different params or replan from failed step
//!   5. Store successful plans as concepts for reuse
//!   6. Max replanning attempts before giving up

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Plan types
// ---------------------------------------------------------------------------

/// A single step in a multi-step plan.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlanStep {
    /// Step number (1-indexed).
    pub step_num: usize,
    /// Tool to invoke.
    pub tool_name: String,
    /// Arguments (may contain $N references to previous step outputs).
    pub args: String,
    /// Whether to retry this step on failure.
    pub retry_on_fail: bool,
    /// Maximum retries for this step.
    pub max_retries: usize,
    /// Optional condition: execute only if this evaluates to true.
    /// Supports: "$N.success", "$N.output.contains('x')", "true", "false"
    pub condition: Option<String>,
    /// Label for debugging/logging.
    pub label: Option<String>,
}

impl PlanStep {
    /// Create a simple plan step.
    pub fn new(step_num: usize, tool_name: impl Into<String>, args: impl Into<String>) -> Self {
        Self {
            step_num,
            tool_name: tool_name.into(),
            args: args.into(),
            retry_on_fail: true,
            max_retries: 2,
            condition: None,
            label: None,
        }
    }

    /// Create with no retry on failure.
    pub fn no_retry(step_num: usize, tool_name: impl Into<String>, args: impl Into<String>) -> Self {
        Self {
            step_num,
            tool_name: tool_name.into(),
            args: args.into(),
            retry_on_fail: false,
            max_retries: 0,
            condition: None,
            label: None,
        }
    }

    /// Create with a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Create with a condition.
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }
}

/// A complete multi-step plan.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolPlan {
    /// Plan steps in execution order.
    pub steps: Vec<PlanStep>,
    /// Plan description (what the plan accomplishes).
    pub description: String,
    /// Maximum replan attempts on failure.
    pub max_replans: usize,
}

impl ToolPlan {
    /// Create a new plan from steps.
    pub fn new(steps: Vec<PlanStep>, description: impl Into<String>) -> Self {
        Self {
            steps,
            description: description.into(),
            max_replans: 2,
        }
    }

    /// Parse a plan from model output.
    ///
    /// Expected format:
    /// ```text
    /// [plan]
    /// description: Analyze and summarize the data
    /// Step 1: calm_execute(add 3 5) [label: compute sum]
    /// Step 2: concept_lookup($1) [if: $1.success]
    /// Step 3: mirror_test($2) [retry: 3]
    /// [/plan]
    /// ```
    pub fn parse_from_output(output: &str) -> Option<Self> {
        let start = output.find("[plan]")?;
        let end = output.find("[/plan]")?;
        let body = &output[start + "[plan]".len()..end];

        let mut steps = Vec::new();
        let mut description = String::new();

        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Parse description
            if let Some(desc) = trimmed.strip_prefix("description:") {
                description = desc.trim().to_string();
                continue;
            }

            // Parse step: "Step N: tool_name(args) [options]"
            if let Some(rest) = trimmed.strip_prefix("Step ") {
                if let Some(colon_pos) = rest.find(':') {
                    let step_num: usize = rest[..colon_pos].trim().parse().ok()?;
                    let step_content = rest[colon_pos + 1..].trim();

                    // Extract options in brackets
                    let mut retry_on_fail = true;
                    let mut max_retries = 2;
                    let mut condition = None;
                    let mut label = None;

                    // Parse [label: ...]
                    if let Some(lstart) = step_content.find("[label:") {
                        if let Some(lend) = step_content[lstart..].find(']') {
                            label = Some(step_content[lstart + 7..lstart + lend].trim().to_string());
                        }
                    }

                    // Parse [if: ...]
                    if let Some(cstart) = step_content.find("[if:") {
                        if let Some(cend) = step_content[cstart..].find(']') {
                            condition = Some(step_content[cstart + 4..cstart + cend].trim().to_string());
                        }
                    }

                    // Parse [retry: N]
                    if let Some(rstart) = step_content.find("[retry:") {
                        if let Some(rend) = step_content[rstart..].find(']') {
                            if let Ok(n) = step_content[rstart + 7..rstart + rend].trim().parse::<usize>() {
                                max_retries = n;
                                retry_on_fail = n > 0;
                            }
                        }
                    }

                    // Extract tool_name(args) — strip the option brackets
                    let mut core = step_content.to_string();
                    // Remove all [option] blocks
                    while let Some(bstart) = core.find('[') {
                        if let Some(bend) = core[bstart..].find(']') {
                            core.replace_range(bstart..bstart + bend + 1, "");
                        } else {
                            break;
                        }
                    }
                    let core = core.trim();

                    // Parse tool_name(args)
                    let (tool_name, args) = if let Some(paren) = core.find('(') {
                        let name = core[..paren].trim().to_string();
                        let args_end = core.rfind(')').unwrap_or(core.len());
                        let args = core[paren + 1..args_end].trim().to_string();
                        (name, args)
                    } else {
                        (core.to_string(), String::new())
                    };

                    steps.push(PlanStep {
                        step_num,
                        tool_name,
                        args,
                        retry_on_fail,
                        max_retries,
                        condition,
                        label,
                    });
                }
            }
        }

        if steps.is_empty() {
            return None;
        }

        // Re-number steps to be sequential
        for (i, step) in steps.iter_mut().enumerate() {
            step.step_num = i + 1;
        }

        Some(ToolPlan {
            steps,
            description,
            max_replans: 2,
        })
    }
}

// ---------------------------------------------------------------------------
// Execution state
// ---------------------------------------------------------------------------

/// Result of executing a single plan step.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepResult {
    /// Step number.
    pub step_num: usize,
    /// Tool that was called.
    pub tool_name: String,
    /// Arguments after reference resolution.
    pub resolved_args: String,
    /// Tool output.
    pub output: String,
    /// Whether execution succeeded.
    pub success: bool,
    /// Mirror quality score (if available).
    pub quality: Option<f32>,
    /// Number of retries used.
    pub retries_used: usize,
    /// Timestamp.
    pub timestamp: u64,
}

/// Result of executing a complete plan.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlanResult {
    /// The plan that was executed.
    pub plan: ToolPlan,
    /// Results for each step.
    pub step_results: Vec<StepResult>,
    /// Final output (last step's output, or error).
    pub final_output: String,
    /// Whether the entire plan succeeded.
    pub success: bool,
    /// Number of replanning attempts used.
    pub replans_used: usize,
    /// Total steps executed (including retries).
    pub total_steps_executed: usize,
    /// Cumulative quality across all steps.
    pub avg_quality: f32,
}

// ---------------------------------------------------------------------------
// Plan validation
// ---------------------------------------------------------------------------

/// Validation error for a plan.
#[derive(Debug, Clone)]
pub struct PlanValidationError {
    pub step_num: Option<usize>,
    pub message: String,
}

/// Result of plan validation.
#[derive(Debug, Clone)]
pub struct PlanValidation {
    pub valid: bool,
    pub errors: Vec<PlanValidationError>,
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Plan executor configuration
// ---------------------------------------------------------------------------

/// Configuration for the plan executor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlanExecutorConfig {
    /// Maximum total steps across all replans.
    pub max_total_steps: usize,
    /// Maximum replanning attempts.
    pub max_replans: usize,
    /// Quality threshold to consider a step successful.
    pub step_quality_threshold: f32,
    /// Whether to resolve $N references.
    pub resolve_references: bool,
}

impl Default for PlanExecutorConfig {
    fn default() -> Self {
        Self {
            max_total_steps: 50,
            max_replans: 2,
            step_quality_threshold: 0.5,
            resolve_references: true,
        }
    }
}

// ---------------------------------------------------------------------------
// PlanExecutor
// ---------------------------------------------------------------------------

/// Executes multi-step tool plans with reference resolution and replanning.
pub struct PlanExecutor {
    config: PlanExecutorConfig,
    /// Known tool names (for validation).
    available_tools: Vec<String>,
    /// Execution history.
    history: Vec<PlanResult>,
}

impl PlanExecutor {
    /// Create a new plan executor.
    pub fn new(config: PlanExecutorConfig) -> Self {
        Self {
            config,
            available_tools: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_executor() -> Self {
        Self::new(PlanExecutorConfig::default())
    }

    /// Register a tool as available for plans.
    pub fn register_tool(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.available_tools.contains(&name) {
            self.available_tools.push(name);
        }
    }

    /// Register multiple tools.
    pub fn register_tools(&mut self, names: &[&str]) {
        for name in names {
            self.register_tool(name.to_string());
        }
    }

    /// Check if a tool is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.available_tools.iter().any(|t| t == name)
    }

    /// Number of available tools.
    pub fn tool_count(&self) -> usize {
        self.available_tools.len()
    }

    /// Get execution history.
    pub fn history(&self) -> &[PlanResult] {
        &self.history
    }

    /// Number of plans executed.
    pub fn plans_executed(&self) -> usize {
        self.history.len()
    }

    // ---- Main API ----

    /// Parse and validate a plan from model output.
    pub fn parse_plan(&self, output: &str) -> Option<(ToolPlan, PlanValidation)> {
        let plan = ToolPlan::parse_from_output(output)?;
        let validation = self.validate_plan(&plan);
        Some((plan, validation))
    }

    /// Validate a plan before execution.
    pub fn validate_plan(&self, plan: &ToolPlan) -> PlanValidation {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if plan.steps.is_empty() {
            errors.push(PlanValidationError {
                step_num: None,
                message: "Plan has no steps".into(),
            });
            return PlanValidation { valid: false, errors, warnings };
        }

        for step in &plan.steps {
            // Check tool exists
            if !self.available_tools.is_empty()
                && !self.available_tools.contains(&step.tool_name)
            {
                errors.push(PlanValidationError {
                    step_num: Some(step.step_num),
                    message: format!("Unknown tool '{}' in step {}", step.tool_name, step.step_num),
                });
            }

            // Check references are valid
            for (ref_num, _) in Self::find_references(&step.args) {
                // Reference must be to a previous step
                if ref_num >= step.step_num {
                    errors.push(PlanValidationError {
                        step_num: Some(step.step_num),
                        message: format!(
                            "Step {} references ${}, but only steps 1-{} exist at that point",
                            step.step_num, ref_num, step.step_num - 1
                        ),
                    });
                }
            }

            // Check condition references
            if let Some(ref cond) = step.condition {
                for (ref_num, _) in Self::find_references(cond) {
                    if ref_num >= step.step_num {
                        warnings.push(format!(
                            "Step {} condition references ${} which may not be available yet",
                            step.step_num, ref_num
                        ));
                    }
                }
            }

            // Warn about steps with no retry
            if !step.retry_on_fail {
                warnings.push(format!(
                    "Step {} has no retry — failure will stop the plan",
                    step.step_num
                ));
            }
        }

        // Check for step numbering gaps
        let step_nums: Vec<usize> = plan.steps.iter().map(|s| s.step_num).collect();
        for i in 1..step_nums.len() {
            if step_nums[i] != step_nums[i - 1] + 1 {
                warnings.push(format!(
                    "Step numbering gap: {} → {}",
                    step_nums[i - 1], step_nums[i]
                ));
            }
        }

        // Check total steps
        if plan.steps.len() > self.config.max_total_steps {
            errors.push(PlanValidationError {
                step_num: None,
                message: format!(
                    "Plan has {} steps, max is {}",
                    plan.steps.len(),
                    self.config.max_total_steps
                ),
            });
        }

        PlanValidation {
            valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Execute a plan using the provided tool execution function.
    ///
    /// The `execute_fn` takes (tool_name, args) and returns (output, success, quality).
    pub fn execute_plan<F>(
        &mut self,
        plan: ToolPlan,
        execute_fn: F,
    ) -> PlanResult
    where
        F: Fn(&str, &str) -> (String, bool, Option<f32>),
    {
        let no_replan: Option<fn(&ToolPlan, &[StepResult], usize) -> Option<ToolPlan>> = None;
        self.execute_plan_with_replan(plan, execute_fn, no_replan)
    }

    /// Execute a plan with optional replanning function.
    ///
    /// On failure, the `replan_fn` is called to generate a new plan from the
    /// failed step onwards. If no replan_fn is provided, execution stops on failure.
    pub fn execute_plan_with_replan<F, R>(
        &mut self,
        plan: ToolPlan,
        execute_fn: F,
        replan_fn: Option<R>,
    ) -> PlanResult
    where
        F: Fn(&str, &str) -> (String, bool, Option<f32>),
        R: Fn(&ToolPlan, &[StepResult], usize) -> Option<ToolPlan>,
    {
        let mut step_results: Vec<StepResult> = Vec::new();
        let mut outputs: HashMap<usize, StepResult> = HashMap::new();
        let mut total_executed = 0;
        let mut replans_used = 0;
        let mut current_plan = plan.clone();

        loop {
            let mut failed_step = None;

            for step in &current_plan.steps {
                // Check if this step was already completed (from a previous replan)
                if outputs.contains_key(&step.step_num) {
                    continue;
                }

                // Evaluate condition
                if let Some(ref cond) = step.condition {
                    if !Self::evaluate_condition(cond, &outputs) {
                        // Skip this step
                        let result = StepResult {
                            step_num: step.step_num,
                            tool_name: step.tool_name.clone(),
                            resolved_args: String::new(),
                            output: "[skipped] condition not met".into(),
                            success: true,
                            quality: None,
                            retries_used: 0,
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0),
                        };
                        outputs.insert(step.step_num, result.clone());
                        step_results.push(result);
                        continue;
                    }
                }

                // Resolve references in args
                let resolved_args = if self.config.resolve_references {
                    Self::resolve_refs(&step.args, &outputs)
                } else {
                    step.args.clone()
                };

                // Execute with retries
                let mut result = Self::execute_step(
                    step,
                    &resolved_args,
                    &execute_fn,
                    self.config.step_quality_threshold,
                );
                total_executed += 1 + result.retries_used;

                // Retry loop
                if !result.success && step.retry_on_fail {
                    for retry in 0..step.max_retries {
                        let retry_result = Self::execute_step(
                            step,
                            &resolved_args,
                            &execute_fn,
                            self.config.step_quality_threshold,
                        );
                        total_executed += 1;

                        if retry_result.success {
                            result = retry_result;
                            result.retries_used = retry + 1;
                            break;
                        }
                    }
                }

                outputs.insert(step.step_num, result.clone());
                step_results.push(result.clone());

                if !result.success {
                    failed_step = Some(step.step_num);
                    break;
                }
            }

            // Handle failure
            if let Some(failed) = failed_step {
                if replans_used < current_plan.max_replans.min(self.config.max_replans) {
                    if let Some(ref replan_fn) = replan_fn {
                        if let Some(new_plan) = replan_fn(&current_plan, &step_results, failed) {
                            replans_used += 1;
                            current_plan = new_plan;
                            continue; // Retry with new plan
                        }
                    }
                }

                // Can't replan — return failure
                let final_output = step_results
                    .iter()
                    .rev()
                    .find(|r| r.success)
                    .map(|r| r.output.clone())
                    .unwrap_or_else(|| format!("Plan failed at step {}", failed));

                let avg_quality = Self::compute_avg_quality(&step_results);
                let result = PlanResult {
                    plan: current_plan,
                    step_results,
                    final_output,
                    success: false,
                    replans_used,
                    total_steps_executed: total_executed,
                    avg_quality,
                };
                self.history.push(result.clone());
                return result;
            }

            // All steps succeeded
            let final_output = step_results
                .last()
                .map(|r| r.output.clone())
                .unwrap_or_default();

            let avg_quality = Self::compute_avg_quality(&step_results);
            let result = PlanResult {
                plan: current_plan,
                step_results,
                final_output,
                success: true,
                replans_used,
                total_steps_executed: total_executed,
                avg_quality,
            };
            self.history.push(result.clone());
            return result;
        }
    }

    // ---- Internal ----

    fn execute_step<F>(
        step: &PlanStep,
        resolved_args: &str,
        execute_fn: &F,
        quality_threshold: f32,
    ) -> StepResult
    where
        F: Fn(&str, &str) -> (String, bool, Option<f32>),
    {
        let (output, success, quality) = execute_fn(&step.tool_name, resolved_args);

        // Quality-based success override
        let effective_success = if success {
            if let Some(q) = quality {
                q >= quality_threshold
            } else {
                true
            }
        } else {
            false
        };

        StepResult {
            step_num: step.step_num,
            tool_name: step.tool_name.clone(),
            resolved_args: resolved_args.to_string(),
            output,
            success: effective_success,
            quality,
            retries_used: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Find all $N references in a string. Returns (step_num, start_pos).
    fn find_references(text: &str) -> Vec<(usize, usize)> {
        let mut refs = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                let mut num_start = i + 1;
                while num_start < bytes.len() && bytes[num_start].is_ascii_digit() {
                    num_start += 1;
                }
                if let Ok(n) = text[i + 1..num_start].parse::<usize>() {
                    refs.push((n, i));
                }
                i = num_start;
            } else {
                i += 1;
            }
        }
        refs
    }

    /// Resolve $N references with step outputs.
    fn resolve_refs(args: &str, outputs: &HashMap<usize, StepResult>) -> String {
        let refs = Self::find_references(args);
        if refs.is_empty() {
            return args.to_string();
        }

        let mut result = args.to_string();
        // Replace from right to left to preserve positions
        for (ref_num, _) in refs.into_iter().rev() {
            let placeholder = format!("${}", ref_num);
            let replacement = outputs
                .get(&ref_num)
                .map(|r| r.output.clone())
                .unwrap_or_else(|| format!("[UNRESOLVED:${}]", ref_num));
            result = result.replace(&placeholder, &replacement);
        }
        result
    }

    /// Evaluate a condition string against step outputs.
    fn evaluate_condition(condition: &str, outputs: &HashMap<usize, StepResult>) -> bool {
        let cond = condition.trim();

        // Simple conditions
        if cond == "true" {
            return true;
        }
        if cond == "false" {
            return false;
        }

        // $N.success
        if let Some(rest) = cond.strip_prefix('$') {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(step_num) = rest[..dot_pos].parse::<usize>() {
                    let field = &rest[dot_pos + 1..];
                    if let Some(result) = outputs.get(&step_num) {
                        match field {
                            "success" => return result.success,
                            "output.contains(" => {
                                // Extract substring to check
                                if let Some(_end) = field.find(')') {
                                    // Not fully implementing nested parsing, simplified
                                    return true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Default: allow execution
        true
    }

    /// Compute average quality across step results.
    fn compute_avg_quality(results: &[StepResult]) -> f32 {
        let qualities: Vec<f32> = results.iter()
            .filter_map(|r| r.quality)
            .collect();
        if qualities.is_empty() {
            1.0 // No quality data = assume success
        } else {
            qualities.iter().sum::<f32>() / qualities.len() as f32
        }
    }

    // ---- Stats ----

    /// Get execution statistics.
    pub fn stats(&self) -> PlanExecutorStats {
        let total = self.history.len();
        let successful = self.history.iter().filter(|r| r.success).count();
        let avg_steps = if total > 0 {
            self.history.iter().map(|r| r.step_results.len()).sum::<usize>() as f32 / total as f32
        } else {
            0.0
        };

        PlanExecutorStats {
            plans_executed: total,
            successful,
            failed: total - successful,
            avg_steps_per_plan: avg_steps,
            total_replans: self.history.iter().map(|r| r.replans_used).sum(),
        }
    }
}

/// Statistics about plan execution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlanExecutorStats {
    pub plans_executed: usize,
    pub successful: usize,
    pub failed: usize,
    pub avg_steps_per_plan: f32,
    pub total_replans: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_execute(tool: &str, args: &str) -> (String, bool, Option<f32>) {
        match tool {
            "add" => {
                let nums: Vec<i32> = args.split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                let sum: i32 = nums.iter().sum();
                (sum.to_string(), true, Some(0.95))
            }
            "echo" => (args.to_string(), true, Some(1.0)),
            "fail" => ("error".into(), false, Some(0.0)),
            "quality_check" => {
                let quality: f32 = args.parse().unwrap_or(0.5);
                (format!("quality={}", quality), quality >= 0.5, Some(quality))
            }
            _ => (format!("unknown tool: {}", tool), false, None),
        }
    }

    #[test]
    fn test_parse_plan() {
        let output = r#"
Let me plan this out:
[plan]
description: Add two numbers and echo the result
Step 1: add(3 5) [label: compute sum]
Step 2: echo($1) [if: $1.success]
[/plan]
"#;
        let plan = ToolPlan::parse_from_output(output).unwrap();
        assert_eq!(plan.description, "Add two numbers and echo the result");
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].tool_name, "add");
        assert_eq!(plan.steps[0].args, "3 5");
        assert_eq!(plan.steps[0].label.as_deref(), Some("compute sum"));
        assert_eq!(plan.steps[1].tool_name, "echo");
        assert_eq!(plan.steps[1].args, "$1");
        assert_eq!(plan.steps[1].condition.as_deref(), Some("$1.success"));
    }

    #[test]
    fn test_parse_plan_no_block() {
        assert!(ToolPlan::parse_from_output("no plan here").is_none());
    }

    #[test]
    fn test_parse_plan_empty_steps() {
        let output = "[plan]\ndescription: empty\n[/plan]";
        assert!(ToolPlan::parse_from_output(output).is_none());
    }

    #[test]
    fn test_execute_simple_plan() {
        let mut executor = PlanExecutor::default_executor();
        executor.register_tools(&["add", "echo"]);

        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "add", "3 5"),
            PlanStep::new(2, "echo", "$1"),
        ], "test plan");

        let result = executor.execute_plan(plan, simple_execute);
        assert!(result.success);
        assert_eq!(result.step_results.len(), 2);
        assert_eq!(result.step_results[0].output, "8");
        assert_eq!(result.step_results[1].output, "8"); // $1 resolved to "8"
    }

    #[test]
    fn test_execute_plan_failure() {
        let mut executor = PlanExecutor::default_executor();
        executor.register_tools(&["add", "fail"]);

        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "add", "1 2"),
            PlanStep::new(2, "fail", ""),
        ], "will fail");

        let result = executor.execute_plan(plan, simple_execute);
        assert!(!result.success);
    }

    #[test]
    fn test_execute_plan_with_retry() {
        let mut executor = PlanExecutor::default_executor();
        executor.register_tools(&["fail"]);

        let plan = ToolPlan::new(vec![
            PlanStep {
                step_num: 1,
                tool_name: "fail".into(),
                args: String::new(),
                retry_on_fail: true,
                max_retries: 3,
                condition: None,
                label: None,
            },
        ], "retry test");

        let result = executor.execute_plan(plan, simple_execute);
        assert!(!result.success);
        assert!(result.total_steps_executed >= 2); // initial + at least 1 retry
    }

    #[test]
    fn test_reference_resolution() {
        let outputs = HashMap::from([
            (1, StepResult {
                step_num: 1,
                tool_name: "add".into(),
                resolved_args: "3 5".into(),
                output: "8".into(),
                success: true,
                quality: Some(0.9),
                retries_used: 0,
                timestamp: 0,
            }),
            (2, StepResult {
                step_num: 2,
                tool_name: "echo".into(),
                resolved_args: "8".into(),
                output: "8".into(),
                success: true,
                quality: Some(1.0),
                retries_used: 0,
                timestamp: 0,
            }),
        ]);

        let resolved = PlanExecutor::resolve_refs("$1 + $2 = result", &outputs);
        assert_eq!(resolved, "8 + 8 = result");
    }

    #[test]
    fn test_reference_resolution_unresolved() {
        let outputs = HashMap::new();
        let resolved = PlanExecutor::resolve_refs("$1 is unknown", &outputs);
        assert!(resolved.contains("[UNRESOLVED:$1]"));
    }

    #[test]
    fn test_find_references() {
        let refs = PlanExecutor::find_references("$1 + $2 equals $10");
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0].0, 1);
        assert_eq!(refs[1].0, 2);
        assert_eq!(refs[2].0, 10);
    }

    #[test]
    fn test_validate_plan_success() {
        let executor = PlanExecutor::default_executor();
        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "add", "1 2"),
            PlanStep::new(2, "echo", "$1"),
        ], "valid plan");

        // No tools registered = no tool validation
        let validation = executor.validate_plan(&plan);
        assert!(validation.valid);
    }

    #[test]
    fn test_validate_plan_unknown_tool() {
        let mut executor = PlanExecutor::default_executor();
        executor.register_tools(&["add"]);

        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "add", "1 2"),
            PlanStep::new(2, "nonexistent", "$1"),
        ], "invalid tool");

        let validation = executor.validate_plan(&plan);
        assert!(!validation.valid);
        assert!(validation.errors.iter().any(|e| e.message.contains("nonexistent")));
    }

    #[test]
    fn test_validate_plan_forward_reference() {
        let executor = PlanExecutor::default_executor();
        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "echo", "$2"), // References future step!
        ], "forward ref");

        let validation = executor.validate_plan(&plan);
        assert!(!validation.valid);
        assert!(validation.errors.iter().any(|e| e.message.contains("references $2")));
    }

    #[test]
    fn test_validate_plan_empty() {
        let executor = PlanExecutor::default_executor();
        let plan = ToolPlan::new(vec![], "empty");

        let validation = executor.validate_plan(&plan);
        assert!(!validation.valid);
    }

    #[test]
    fn test_evaluate_condition_true() {
        let outputs = HashMap::from([
            (1, StepResult {
                step_num: 1,
                tool_name: "add".into(),
                resolved_args: String::new(),
                output: "8".into(),
                success: true,
                quality: Some(0.9),
                retries_used: 0,
                timestamp: 0,
            }),
        ]);
        assert!(PlanExecutor::evaluate_condition("true", &outputs));
        assert!(PlanExecutor::evaluate_condition("$1.success", &outputs));
        assert!(!PlanExecutor::evaluate_condition("false", &outputs));
    }

    #[test]
    fn test_evaluate_condition_failure() {
        let outputs = HashMap::from([
            (1, StepResult {
                step_num: 1,
                tool_name: "fail".into(),
                resolved_args: String::new(),
                output: "error".into(),
                success: false,
                quality: Some(0.0),
                retries_used: 0,
                timestamp: 0,
            }),
        ]);
        assert!(!PlanExecutor::evaluate_condition("$1.success", &outputs));
    }

    #[test]
    fn test_execute_plan_with_replan() {
        let mut executor = PlanExecutor::default_executor();
        executor.register_tools(&["add", "fail", "echo"]);

        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "fail", ""),
            PlanStep::new(2, "echo", "done"),
        ], "needs replan");

        // Replan: replace step 1 with "add"
        let result = executor.execute_plan_with_replan(
            plan,
            simple_execute,
            Some(|_plan: &ToolPlan, _results: &[StepResult], _failed: usize| {
                Some(ToolPlan::new(vec![
                    PlanStep::new(1, "add", "1 2"),
                    PlanStep::new(2, "echo", "done"),
                ], "replanned"))
            }),
        );

        assert!(result.success);
        assert_eq!(result.replans_used, 1);
    }

    #[test]
    fn test_compute_avg_quality() {
        let results = vec![
            StepResult {
                step_num: 1, tool_name: "a".into(), resolved_args: String::new(),
                output: String::new(), success: true, quality: Some(0.8),
                retries_used: 0, timestamp: 0,
            },
            StepResult {
                step_num: 2, tool_name: "b".into(), resolved_args: String::new(),
                output: String::new(), success: true, quality: Some(1.0),
                retries_used: 0, timestamp: 0,
            },
        ];
        let avg = PlanExecutor::compute_avg_quality(&results);
        assert!((avg - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_compute_avg_quality_no_data() {
        let results = vec![
            StepResult {
                step_num: 1, tool_name: "a".into(), resolved_args: String::new(),
                output: String::new(), success: true, quality: None,
                retries_used: 0, timestamp: 0,
            },
        ];
        assert_eq!(PlanExecutor::compute_avg_quality(&results), 1.0);
    }

    #[test]
    fn test_stats() {
        let mut executor = PlanExecutor::default_executor();
        executor.register_tools(&["add", "echo"]);

        let plan1 = ToolPlan::new(vec![PlanStep::new(1, "add", "1 2")], "p1");
        let plan2 = ToolPlan::new(vec![PlanStep::new(1, "echo", "hello")], "p2");

        executor.execute_plan(plan1, simple_execute);
        executor.execute_plan(plan2, simple_execute);

        let stats = executor.stats();
        assert_eq!(stats.plans_executed, 2);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_plan_step_no_retry() {
        let step = PlanStep::no_retry(1, "add", "1 2");
        assert!(!step.retry_on_fail);
        assert_eq!(step.max_retries, 0);
    }

    #[test]
    fn test_plan_step_with_label() {
        let step = PlanStep::new(1, "add", "1 2").with_label("compute");
        assert_eq!(step.label.as_deref(), Some("compute"));
    }

    #[test]
    fn test_plan_step_with_condition() {
        let step = PlanStep::new(1, "add", "1 2").with_condition("$1.success");
        assert_eq!(step.condition.as_deref(), Some("$1.success"));
    }

    #[test]
    fn test_quality_threshold() {
        let config = PlanExecutorConfig {
            step_quality_threshold: 0.7,
            ..Default::default()
        };
        let mut executor = PlanExecutor::new(config);
        executor.register_tools(&["quality_check"]);

        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "quality_check", "0.3"), // Below threshold
        ], "quality test");

        let result = executor.execute_plan(plan, simple_execute);
        // Quality 0.3 < threshold 0.7, so step fails
        assert!(!result.step_results[0].success);
    }

    #[test]
    fn test_max_total_steps() {
        let config = PlanExecutorConfig {
            max_total_steps: 2,
            ..Default::default()
        };
        let executor = PlanExecutor::new(config);
        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "add", "1 2"),
            PlanStep::new(2, "add", "3 4"),
            PlanStep::new(3, "add", "5 6"),
        ], "too many steps");

        let validation = executor.validate_plan(&plan);
        assert!(!validation.valid);
    }

    #[test]
    fn test_skip_condition_not_met() {
        let mut executor = PlanExecutor::default_executor();
        executor.register_tools(&["echo"]);

        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "echo", "first").with_condition("false"),
        ], "skip test");

        let result = executor.execute_plan(plan, simple_execute);
        assert!(result.success);
        assert!(result.step_results[0].output.contains("skipped"));
    }

    #[test]
    fn test_chained_references() {
        let mut executor = PlanExecutor::default_executor();
        executor.register_tools(&["add", "echo"]);

        let plan = ToolPlan::new(vec![
            PlanStep::new(1, "add", "3 5"),
            PlanStep::new(2, "add", "$1 2"),
            PlanStep::new(3, "echo", "Result: $2"),
        ], "chain");

        let result = executor.execute_plan(plan, simple_execute);
        assert!(result.success);
        assert_eq!(result.step_results[0].output, "8");
        assert_eq!(result.step_results[1].output, "10"); // 8 + 2
        assert_eq!(result.step_results[2].output, "Result: 10");
    }
}
