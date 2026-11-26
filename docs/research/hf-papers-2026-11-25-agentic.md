# HF papers — 2026-11-25, agentic

Scope: the 25 November daily page, filtered to agentic/long-horizon papers. Four
were read and assessed; the rest are named at the end. No code is copied.

| paper | why it matters here |
|---|---|
| **RECAP: Training Long-horizon LLM Agents with Context Compaction** (`2611.31258`) | trains compaction, not just prompts it — our long-running plan |
| **Lazy Agents Answer Against Their Own Beliefs** (`2611.32884`) | evidence-gated commitment without a judge — our "no claim without source" |
| **InstructReplay** (`2611.32918`) | factual fine-tuning erodes general ability; replay the initial model |
| **CoT-Proxy** (`2611.33148`) | the long-horizon policy distils from **actions alone** |

## 1. RECAP — compaction as a trained behaviour

**Claim.** A summary should preserve the **behaviour** induced by the original
history. Measure that as the token-level KL between the agent's actions under the
compressed and the full context; the full-context policy is a self-teacher. The
gradient splits into a distortion-based reward for the summary and self-distillation
for post-compaction actions. No judge, no extra rollouts.

**Numbers.** Qwen3.5-4B, 32K working context, up to 3 compactions: **+4.6%** on
BrowseComp-Plus and **+4.3%** on SWE-bench Verified over task-reward-only RL, with
fewer tool calls and tokens. The gain grows at a 16K window, and it closes most of
the gap to a 128K context at a quarter of the peak length.

**For us.** This is the training objective our notebook/compaction design was
missing. It also gives an eval we can implement cheaply: run the same task with
full history and with the compacted notebook, and compare the next action. A
summary that changes the action is a bad summary, whatever it reads like.

**Action.** Add *behaviour distortion* to the long-horizon eval now (no training
needed), and put the RECAP objective on the post-SFT list. The notebook's typed
items map directly: decisions/facts/evidence are what the summary must preserve.

**Caveat.** RL on a 4B model, two benchmarks, 150 steps. The *metric* transfers;
the training recipe needs our own budget.

## 2. RLVB — acting on what you believe

**Claim.** Agents commit before they have enough evidence ("laziness"), even when
their own belief line names several possible answers. The harness cannot detect
this because only the agent sees the observations. In a simulator the exact
posterior over the hidden answer is computable, so the process reward can score
*the decision to continue or commit* against the evidence — no judge, critic or
reward model.

**Numbers.** +17.1% average across three VLMs and three benchmarks; base models
answer with sufficient evidence in 8%/0.4%/36% of episodes; the trained agent
51%/12%/74%. The gain comes from answering at high evidence, not from better
answers at equal evidence. It commits at 0.1–2.2% of turns where its belief line
still names several answers, against 3–68% for the base model.

**For us.** This is the behavioural form of "no claim without a source", applied
to *stopping*. Our notebook already has `open_question` and `resolve`; the
trained behaviour is: while an item is open, act; when nothing is open and the
evidence supports it, answer. The eval metric is "commits with insufficient
evidence", which we do not currently measure.

**Action.** Add the metric to `eval-plan.md` §11 and make the behaviour a
training target for long-horizon work. The exact posterior needs a simulator; in
the harness, the evidence tools (`evidence_query`, `memory_unresolved`) are the
closest verifiable substitute.

## 3. InstructReplay — keep the model you started with

**Claim.** Fine-tuning on new facts lowers instruction-following, reasoning and
coding by **35 points on average**, and the losses track a **collapse in answer
length**. Replaying the *initial model's own answers* to general instructions
during fine-tuning, with a KL term toward them, reduces the average decline from
**28.1 to 2.4** points while learning the new facts as well as standard SFT.

**For us.** Every SFT/KD run we plan is exactly this failure mode. A replay set
of the base model's answers to general instructions, mixed ~15%, is cheap and
should be mandatory. The answer-length collapse is also a usable early warning.

**Action.** Add to `data-plan.md`/`training-runbook.md`: a replay corpus from the
initial checkpoint, out-of-domain, with a KL term; and a **general-ability
regression gate** (instruction-following, reasoning, coding, answer length) that
every fine-tune must pass. That gate does not exist in our eval plan.

## 4. CoT-Proxy — actions are the policy

**Claim.** Distil a long-horizon policy from the teacher's **actions alone**,
with an empty reasoning block in every training target and a KL replay term
toward the initial student. The student still reasons in its own style at
inference ("cross-format transfer"). **CoT-proxies** — short summaries placed
*outside* the think block — add signal without imitating the teacher's CoT. The
recipe works with closed teachers' visible output and API summaries.

**Numbers.** Qwen3-4B on AppWorld 7.3% → 17.9% action-only, 34.5% with proxies;
Qwen3-32B 11.5% → 25.4% → 29.8%; GPT-5.6 Luna teacher: 7.3% → 35.1% → 48.2%.
Replay is required: without it AIME drops 74.7 → 68.9 and KL divergence rises 14×.

**For us.** Our distillation was planned as text-aligned logit KD, which teaches
*language*, not *agent behaviour*. This says the behaviour can come from
trajectory SFT on actions, with the student's own reasoning, plus short proxies.
Keep text-aligned KD for language; use action-only/CoT-proxy for tool behaviour.
It also means a teacher's hidden CoT is not a blocker.

**Caveat.** Terms of service for closed teachers' outputs must be checked; we use
open-weight teachers, so the issue is the datasets' terms, not a hidden CoT.

## The pattern across all four

Three of the four use **a KL/replay term toward the initial or full-context
policy** to prevent drift. That is the cheapest safeguard in the set and it is
missing from our training plan. The fourth (RLVB) uses a verifiable belief for
the same reason: don't let the model drift from what the evidence supports.

## Changes to make

1. **Replay + KL term in every SFT/KD run**, out-of-domain, ~15%, plus a
   general-ability gate and an answer-length check.
2. **Behaviour-distortion eval** for compaction (full vs notebook context).
3. **"Commits with insufficient evidence"** metric and a trained stopping policy.
4. **Distillation split**: text-aligned KD for language, action-only/CoT-proxy
   trajectory SFT for tool behaviour; keep the teacher's CoT out of the targets.
5. **RECAP-style compaction training** after SFT, not before.

## Not assessed

- `Distill to Loop` (`2611.31977`) — turning a pretrained model into a looped one;
  relevant to the loop/MoLT line.
- `Model Carving` (`2611.27962`) — topology self-optimisation.
- `Steerable by Design` (`2611.32199`) — conditional pretraining for control;
  relevant to modes.
- `SpecEngineeringBench` (`2611.32973`) — coding agents as spec engineers.
- `Don't Sleep, Keep Watch` (`2611.28947`) — proactive perception.
- `Self-Evolving Evaluation Environments` (`2611.30962`) — agent eval environments.
- `OWE-Bench` (`2611.30017`) — long-horizon engineering benchmark.
