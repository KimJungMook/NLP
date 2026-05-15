"""
Conductor-style orchestration exploration pilot on MATH-500 subset.

목적
----
- Recursive topology 구현 (논문 Section 3.2 / Figure 12, 14)
- 실험 관찰 지표:
    1) Agent selection entropy
    2) Topology diversity
    3) Workflow length variance
    4) Success / failure trajectory contrast
    5) Recursion statistics (trigger rate, accuracy delta, agent redistribution)

- Conductor : GPT-5-mini (프롬프트 기반, GRPO 학습 없음)
- Workers   : Model 0/1/2 = anonymous language model workers; roles are assigned dynamically by subtasks
- Topology  : 논문 Appendix F.1 5가지 + paper-aligned role taxonomy

Install
-------
pip install openai datasets

Usage
-----
python conductor_math500_anonymous_workers.py
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ============================================================
# 0. Configuration
# ============================================================

OPENAI_API_KEY = "여기에_본인_API_KEY"

CONDUCTOR_MODEL = "gpt-5-mini"
WORKER_MODELS   = ["gpt-5-mini", "gpt-4.1-mini", "gpt-5-nano"]
JUDGE_MODEL     = "gpt-5-mini"

N_QUESTIONS  = 10
N_ROLLOUTS   = 5
RANDOM_SEED  = 40

CONDUCTOR_TEMPERATURE = 1.0
WORKER_TEMPERATURE    = 0.2
JUDGE_TEMPERATURE     = 0.0

MAX_WORKFLOW_STEPS             = 5
MAX_CONDUCTOR_OUTPUT_TOKENS    = 2048
MAX_WORKER_OUTPUT_TOKENS       = 4096
MAX_JUDGE_OUTPUT_TOKENS        = 700
MAX_FORMAT_VALIDATOR_OUTPUT_TOKENS = 250

USE_LLM_JUDGE       = True
USE_FORMAT_VALIDATOR = False
USE_FEW_SHOT_EXAMPLES = True
FEW_SHOT_MODE         = "ood"   # "ood" | "in_domain" | "none"

# ── Recursion (논문 Section 3.2) ─────────────────────────────
ENABLE_RECURSION       = True
MAX_RECURSIVE_CALLS    = 2      # 논문: "less than 2x original agentic calls"
RECURSION_DISCOUNT     = 0.25   # 논문 학습용 discount factor (여기선 참고값)

DEMO_FALLBACK_IF_HF_FAILS = False
OUTPUT_DIR           = Path("conductor_math500_outputs")
SLEEP_BETWEEN_CALLS  = 0.15

PRICE_PER_1M = {
    "gpt-5-mini":   {"input": 0.25, "output": 2.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-5-nano":   {"input": 0.05, "output": 0.40},
}


# ============================================================
# 1. Data structures
# ============================================================

@dataclass
class Usage:
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens  += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens  += other.total_tokens


# ============================================================
# 2. OpenAI client
# ============================================================

class OpenAITextClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        model: str,
        instructions: str,
        user_input: str,
        temperature: Optional[float],
        max_output_tokens: int,
    ) -> Tuple[str, Usage]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": user_input,
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if model.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": "minimal"}
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            resp = self.client.responses.create(**kwargs)
        except Exception:
            kwargs.pop("temperature", None)
            try:
                resp = self.client.responses.create(**kwargs)
            except Exception:
                kwargs.pop("reasoning", None)
                combined = f"SYSTEM INSTRUCTIONS:\n{instructions}\n\nUSER INPUT:\n{user_input}"
                resp = self.client.responses.create(
                    model=model, input=combined,
                    max_output_tokens=max_output_tokens, store=False,
                )

        text  = getattr(resp, "output_text", "") or ""
        usage = self._extract_usage(resp)
        return text, usage

    @staticmethod
    def _extract_usage(resp: Any) -> Usage:
        u = getattr(resp, "usage", None)
        if u is None:
            return Usage()
        inp  = getattr(u, "input_tokens",  0) or 0
        out  = getattr(u, "output_tokens", 0) or 0
        tot  = getattr(u, "total_tokens",  inp + out) or 0
        return Usage(inp, out, tot)


# ============================================================
# 3. Data loading
# ============================================================

def extract_boxed_answer(solution: str) -> Optional[str]:
    if not solution:
        return None
    starts = [m.start() for m in re.finditer(r"\\boxed\{", solution)]
    if not starts:
        return None
    start = starts[-1] + len(r"\boxed{")
    depth, i = 1, start
    while i < len(solution):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i].strip()
        i += 1
    return None


def load_math500_subset(n_questions: int, seed: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception as e:
        if not DEMO_FALLBACK_IF_HF_FAILS:
            raise RuntimeError(
                "Could not load HuggingFaceH4/MATH-500.\n"
                "pip install datasets 후 인터넷 연결을 확인하세요.\n"
                "코드 흐름만 테스트하려면 DEMO_FALLBACK_IF_HF_FAILS = True 설정.\n"
                f"Original error: {repr(e)}"
            )
        return _demo_fallback(n_questions)

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    tasks: List[Dict[str, Any]] = []
    for idx in indices[:n_questions]:
        item    = ds[idx]
        problem = item.get("problem") or item.get("question") or ""
        answer  = item.get("answer") or extract_boxed_answer(item.get("solution", "")) or ""
        tasks.append({
            "id":            f"math500_{item.get('unique_id', item.get('id', idx))}",
            "dataset_index": idx,
            "type":          "math",
            "question":      problem,
            "gold":          str(answer),
            "solution":      item.get("solution", ""),
            "subject":       item.get("subject", "MATH"),
            "level":         item.get("level", ""),
        })
    return tasks


def _demo_fallback(n: int) -> List[Dict[str, Any]]:
    ex = [
        ("If 3x + 7 = 22, find x.", "5"),
        ("Compute 12^2 - 5^2.", "119"),
        ("Triangle angles 40 and 65 degrees. Third angle?", "75"),
        ("Car travels 180 miles in 3 hours. Average speed?", "60"),
        ("Simplify 2/3 + 1/6.", "5/6"),
        ("Area of circle radius 3 in terms of pi.", "9pi"),
        ("Solve 2y - 4 = 10.", "7"),
        ("Next: 2, 4, 8, 16, ?", "32"),
        ("15% of 200?", "30"),
        ("f(x)=x^2+1, f(4)=?", "17"),
    ]
    return [{"id": f"demo_{i}", "type": "math", "question": q, "gold": a, "solution": ""}
            for i, (q, a) in enumerate(ex[:n])]


# ============================================================
# 4. Conductor prompts
# ============================================================

def few_shot_text() -> str:
    if not USE_FEW_SHOT_EXAMPLES or FEW_SHOT_MODE == "none":
        return ""

    if FEW_SHOT_MODE == "ood":
        return """
FEW-SHOT EXAMPLE 1, OOD biomedical QA (2 steps, SINGLE-SHOT→format):
Question: Does brain-derived neurotrophic factor enhance intestinal muscle contraction induced by SP and CGRP? Answer A for Yes or B for No.
Workflow:
{"model_id":[1,2],"subtasks":["Answer the biomedical yes/no question independently with the required option letter.","Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."],"access_list":[[],["all"]]}

FEW-SHOT EXAMPLE 2, OOD limit problem (3 steps, PARALLEL INDEPENDENT):
Question: Evaluate lim_{t->0}(1/ln(1+t)+1/ln(1-t)). Return final answer in LaTeX.
Workflow:
{"model_id":[0,1,2],"subtasks":["Solve the limit independently using Taylor expansion and provide a candidate answer.","Solve the limit independently using another method and provide a candidate answer.","Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."],"access_list":[[],[],["all"]]}

FEW-SHOT EXAMPLE 3, OOD combinatorics (4 steps, MIXED):
Question: In how many ways can 8 non-attacking rooks be placed on an 8x8 chessboard?
Workflow:
{"model_id":[0,1,0,2],"subtasks":["Solve using a direct counting argument and give a candidate integer.","Solve independently using a permutation/factorial approach and give a candidate integer.","Compare the two candidates. If they agree confirm; if not, identify the error and give a corrected answer.","Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."],"access_list":[[],[],["all"],["all"]]}

FEW-SHOT EXAMPLE 4, OOD multi-step integral (5 steps, SEQUENTIAL CHAIN):
Question: Compute the definite integral of x^3*ln(x) from 1 to e, verify with integration by parts twice.
Workflow:
{"model_id":[0,1,2,0,1],"subtasks":["Apply integration by parts once and record the intermediate result.","Apply integration by parts a second time and compute the full definite value.","Independently compute the same integral using the tabular method and give a candidate value.","Compare all three candidates from previous steps. Resolve any discrepancy and confirm.","Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."],"access_list":[[],["all"],[],["all"],["all"]]}
""".strip()

    if FEW_SHOT_MODE == "in_domain":
        return """
FEW-SHOT EXAMPLE 1, contest math (2 steps, SINGLE-SHOT→format):
Question: Solve 2x + 5 = 17, report x.
Workflow:
{"model_id":[1,2],"subtasks":["Solve independently with concise algebra and give the candidate value of x.","Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."],"access_list":[[],["all"]]}

FEW-SHOT EXAMPLE 2, contest math (3 steps, PARALLEL INDEPENDENT):
Question: Count positive integer pairs (a,b) with a+b=20 and ab divisible by 12.
Workflow:
{"model_id":[0,1,2],"subtasks":["Solve the counting problem independently and give a candidate integer.","Solve independently using a modular/case-based check and give a candidate integer.","Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."],"access_list":[[],[],["all"]]}

FEW-SHOT EXAMPLE 3, contest number theory (4 steps, MIXED):
Question: Find the number of integers n with 1<=n<=1000 such that n^2-n is divisible by 5.
Workflow:
{"model_id":[0,1,2,1],"subtasks":["Factor n^2-n=n(n-1) and determine residue classes mod 5 for which 5|n(n-1).","Independently count integers in [1,1000] belonging to the valid residue classes.","Compare the two counts. If they match confirm; if not, identify discrepancy and correct.","Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."],"access_list":[[],[],["all"],["all"]]}

FEW-SHOT EXAMPLE 4, contest geometry+algebra (5 steps, SEQUENTIAL CHAIN):
Question: In triangle ABC, AB=13, BC=14, CA=15. Find the altitude from A to BC.
Workflow:
{"model_id":[0,1,0,2,1],"subtasks":["Compute area using Heron's formula. Show semi-perimeter and each factor.","Independently compute the same area using coordinate geometry.","Use area from Step 1 to compute altitude h=2*Area/BC. State exact fraction.","Cross-check the altitude value using the result from Step 2. Confirm or correct.","Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."],"access_list":[[],[],["all"],["all"],["all"]]}
""".strip()

    return ""


def build_conductor_instructions() -> str:
    fs = few_shot_text()
    return f"""
You are the Conductor in a multi-agent language model system.

Your job is NOT to solve the problem directly.
Your job is to design a workflow of worker-model calls by choosing the most effective coordination topology.

════════════════════════════════════════════
TOPOLOGY CATALOGUE  (논문 Appendix F.1 기준)
════════════════════════════════════════════

access_list[i] = []      → worker i sees ONLY the original question and its subtask.
access_list[i] = ["all"] → worker i sees ALL previous subtasks and responses.

──────────────────────────────────────────
TOPOLOGY 1 · SINGLE-SHOT
  When: trivially simple one-step problems.
  access_list = [[]]
  Flow: User → Worker A → Output

──────────────────────────────────────────
TOPOLOGY 2 · PARALLEL INDEPENDENT (Best-of-N)
  When: factual recall — models answer independently, last one aggregates.
  access_list = [[], [], ["all"]]
  Flow: Worker A ──┐
        Worker B ──┴→ Aggregator

──────────────────────────────────────────
TOPOLOGY 3 · SEQUENTIAL CHAIN (Planner→Executor→Refiner→Checker)
  When: multi-step derivation, code generation, hard contest math.
  access_list = [[], ["all"], ["all"], ["all"]]
  Flow: Planner → Executor → Refiner → Checker
  Note: MATH-500 problems often benefit from 4–5 chain steps.

──────────────────────────────────────────
TOPOLOGY 4 · MIXED (Arbitrary Tree)
  When: hard problems needing BOTH independent parallel attempts
        AND verification/refinement.
  Example A (parallel→cross-check→format):
    access_list = [[], [], ["all"], ["all"]]
  Example B (chain + independent re-solve + aggregate):
    access_list = [[], ["all"], [], ["all"], ["all"]]
  Flow: freely mix [] and ["all"] to express the exact information flow.

──────────────────────────────────────────
TOPOLOGY 5 · CONDUCTOR ABDICATION
  When: extremely complex problems — delegate meta-orchestration to a
        frontier worker.
  subtasks[0] = "Analyze the problem and propose a step-by-step plan
                 for the subsequent models to follow."
  access_list = [[], ["all"], ["all"]]
  Flow: Meta-Orchestrator → Worker B → Worker C

════════════════════════════════════════════
ROLE TAXONOMY  (논문 Appendix F.1 기준)
════════════════════════════════════════════

  planner         — analyzes the problem and devises a strategy
  executor        — implements a plan from the previous step
  solver          — solves the problem independently from scratch
  verifier        — checks/validates a previous candidate answer
  refiner         — improves or corrects a previous answer
  aggregator      — combines/resolves multiple independent attempts
  formatter       — boxes and returns the final answer
  meta_orchestrator — (abdication only) directs other models

════════════════════════════════════════════
TOPOLOGY SELECTION GUIDE
════════════════════════════════════════════

  Problem type                         → Topology
  ─────────────────────────────────────────────────
  Single-step / trivially simple       → SINGLE-SHOT
  Factual recall, multiple-choice      → PARALLEL INDEPENDENT
  Multi-step derivation / code gen     → SEQUENTIAL CHAIN (3–5 steps)
  Hard math + dual verification        → MIXED
  Extremely complex / open-ended       → ABDICATION or MIXED

════════════════════════════════════════════
OUTPUT RULES
════════════════════════════════════════════

1. 1 to {MAX_WORKFLOW_STEPS} workflow steps.
2. All three lists must have the same length.
3. Each model_id must be 0, 1, or 2.
4. Each access_list entry must be [] or ["all"].
5. The same worker model may appear multiple times.
6. Do NOT always default to a 3-step parallel pattern.
   Choose adaptively based on the problem's difficulty and type.
7. The final subtask must say exactly:
   "Return only FINAL_ANSWER: \\boxed{{<exact LaTeX answer>}}."
8. Output compact minified JSON only. No markdown. No commentary.

{fs}
""".strip()


def build_conductor_input(question: str) -> str:
    # Paper-aligned setting: expose only anonymous worker ids, not fixed roles.
    # The Conductor must assign roles dynamically through the natural-language subtasks.
    workers = [
        "Model 0: language model worker.",
        "Model 1: language model worker.",
        "Model 2: language model worker.",
    ]
    return f"""USER QUESTION:
{question}

AVAILABLE WORKER MODELS:
{chr(10).join(workers)}

Return the JSON workflow now."""


# ── Recursive conductor prompts (논문 Figure 14) ──────────────

def build_recursive_conductor_instructions(remaining_steps: int) -> str:
    return f"""
You are the Conductor in a multi-agent language model system.

You have just received the final response from your previous coordination strategy.
Now decide whether to pass it through or launch a new verification/correction round.

════════════════════════════════════════════
YOUR OPTIONS
════════════════════════════════════════════

OPTION A — PASS THROUGH (previous answer looks correct)
  Output three empty lists:
  {{"model_id":[],"subtasks":[],"access_list":[]}}

OPTION B — NEW WORKFLOW (previous answer looks wrong or needs verification)
  Design a new sequence of up to {remaining_steps} steps.
  access_list[i] = []      → worker sees only the question + subtask.
  access_list[i] = ["all"] → worker sees the PREVIOUS ROUND'S FINAL RESPONSE
                              plus all earlier steps in the current round.
  Focus on: correction, alternative approach, independent re-verification.

Use the same topology catalogue as the initial round.
The final subtask must say: "Return only FINAL_ANSWER: \\boxed{{<exact LaTeX answer>}}."
Output compact minified JSON only. No markdown. No commentary.
""".strip()


def build_recursive_conductor_input(question: str, previous_final_response: str) -> str:
    # Keep worker identities anonymous in recursive rounds as well.
    workers = [
        "Model 0: language model worker.",
        "Model 1: language model worker.",
        "Model 2: language model worker.",
    ]
    return f"""ORIGINAL QUESTION:
{question}

FINAL RESPONSE FROM YOUR PREVIOUS COORDINATION STRATEGY:
{previous_final_response}

AVAILABLE WORKER MODELS:
{chr(10).join(workers)}

Pass through (empty lists) or design a new verification workflow."""


# ── Plan parsing ──────────────────────────────────────────────

def parse_plan(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    for pat in (r"^```json\s*", r"^```\s*", r"\s*```$"):
        text = re.sub(pat, "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group(0)

    plan       = json.loads(text)
    model_id   = plan.get("model_id")
    subtasks   = plan.get("subtasks")
    access_list = plan.get("access_list")

    if not (isinstance(model_id, list) and isinstance(subtasks, list) and isinstance(access_list, list)):
        raise ValueError("Plan must include list fields: model_id, subtasks, access_list")
    if len(model_id) == 0 or len(model_id) > MAX_WORKFLOW_STEPS:
        raise ValueError(f"Invalid workflow length: {len(model_id)}")
    if not (len(model_id) == len(subtasks) == len(access_list)):
        raise ValueError("model_id, subtasks, access_list lengths differ")

    subtasks = [str(s).strip() for s in subtasks]
    subtasks[-1] = "Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."

    for mid in model_id:
        if not isinstance(mid, int) or mid < 0 or mid >= len(WORKER_MODELS):
            raise ValueError(f"Invalid model id: {mid}")

    cleaned = []
    for a in access_list:
        if a == []:
            cleaned.append([])
        elif a == ["all"] or a == "all":
            cleaned.append(["all"])
        else:
            raise ValueError(f"Unsupported access list value: {a}")

    return {"model_id": model_id, "subtasks": subtasks, "access_list": cleaned}


def parse_recursive_plan(raw: str) -> Optional[Dict[str, Any]]:
    """Returns None = pass-through. Returns plan dict = execute new workflow."""
    text = raw.strip()
    for pat in (r"^```json\s*", r"^```\s*", r"\s*```$"):
        text = re.sub(pat, "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group(0)

    try:
        obj = json.loads(text)
    except Exception:
        raise ValueError(f"JSON parse failed for recursive plan: {raw[:200]}")

    mid = obj.get("model_id", [])
    sub = obj.get("subtasks", [])
    acc = obj.get("access_list", [])

    if (isinstance(mid, list) and len(mid) == 0
            and isinstance(sub, list) and len(sub) == 0
            and isinstance(acc, list) and len(acc) == 0):
        return None  # pass-through

    return parse_plan(raw)


# ============================================================
# 5. Worker execution
# ============================================================

def worker_description(model_id: int) -> str:
    # Paper-aligned setting: no fixed per-worker role.
    # The current role should come from the assigned subtask, not from model_id.
    return "language model worker"


def build_worker_instructions(model_id: int, is_final_step: bool) -> str:
    final_rule = (
        "You are the final worker.\n"
        "End with exactly one line: FINAL_ANSWER: \\boxed{<latex_answer>}\n"
        "Inside \\boxed{} put ONLY the mathematical expression (exact LaTeX form). "
        "No prose, no units, no approximations unless the problem asks for a decimal."
        if is_final_step
        else (
            "You are not the final worker. "
            "Solve the assigned subtask directly with concise reasoning. "
            "Give an explicit candidate answer for later workers to check."
        )
    )
    return (
        f"You are a worker in a multi-agent math-solving workflow.\n"
        f"Worker profile: {worker_description(model_id)}.\n\n"
        f"Follow your assigned subtask. Include key equations/cases for auditability.\n"
        f"{final_rule}"
    )


def build_worker_input(
    question: str,
    subtask: str,
    history: List[Dict[str, Any]],
    access: List[str],
    parent_response: Optional[str] = None,
) -> str:
    """Build worker context.

    parent_response is set in recursive rounds.  When access == ["all"],
    the worker sees the parent round's final response first, then all
    steps in the current round (논문 Figure 14 동일 구조).
    """
    if access == ["all"]:
        parts: List[str] = []
        if parent_response and parent_response.strip():
            parts.append(
                "[Previous round final response]\n"
                f"{parent_response}"
            )
        for h in history:
            parts.append(
                f"[Current round, step {h['step']}]\n"
                f"Worker: Model {h['model_id']}\n"
                f"Subtask: {h['subtask']}\n"
                f"Response:\n{h['response']}"
            )
        history_text = "\n\n".join(parts) if parts else "No previous outputs visible."
    else:
        history_text = "No previous worker outputs are visible."

    return (
        f"ORIGINAL MATH QUESTION:\n{question}\n\n"
        f"VISIBLE PREVIOUS WORK:\n{history_text}\n\n"
        f"YOUR CURRENT SUBTASK:\n{subtask}"
    )


def execute_workflow(
    client: OpenAITextClient,
    question: str,
    plan: Dict[str, Any],
    parent_response: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Usage]]:
    history: List[Dict[str, Any]] = []
    usage_by_model: Dict[str, Usage] = defaultdict(Usage)

    for step, (mid, subtask, access) in enumerate(
        zip(plan["model_id"], plan["subtasks"], plan["access_list"])
    ):
        model    = WORKER_MODELS[mid]
        is_final = (step == len(plan["model_id"]) - 1)
        instr    = build_worker_instructions(mid, is_final_step=is_final)
        usr_in   = build_worker_input(question, subtask, history, access,
                                      parent_response=parent_response)

        response, usage = client.generate(
            model=model, instructions=instr, user_input=usr_in,
            temperature=WORKER_TEMPERATURE, max_output_tokens=MAX_WORKER_OUTPUT_TOKENS,
        )
        usage_by_model[model].add(usage)
        history.append({
            "step": step, "model_id": mid, "api_model": model,
            "subtask": subtask, "access": access,
            "response": response, "usage": usage.__dict__,
        })
        time.sleep(SLEEP_BETWEEN_CALLS)

    final_answer = history[-1]["response"] if history else ""
    return final_answer, history, usage_by_model


def execute_with_recursion(
    client: OpenAITextClient,
    question: str,
    initial_plan: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Usage], List[Dict[str, Any]]]:
    """
    논문 Section 3.2 / Figure 12 구현.

    Returns
    -------
    final_answer      : 최종 답변 문자열
    final_trajectory  : 최종 실행 라운드의 trajectory
    total_usage       : 전체 API usage 집계
    rounds            : 라운드별 상세 정보 (로깅용)
    """
    total_usage: Dict[str, Usage] = defaultdict(Usage)
    rounds: List[Dict[str, Any]] = []

    # ── Round 0: initial workflow ────────────────────────────
    final_answer, trajectory, round_usage = execute_workflow(
        client, question, initial_plan
    )
    for model, u in round_usage.items():
        total_usage[model].add(u)

    rounds.append({
        "round":           0,
        "is_recursive":    False,
        "recursion_triggered": False,
        "passed_through":  False,
        "plan":            initial_plan,
        "trajectory":      trajectory,
        "raw_final_answer": final_answer,
        "topology":        None,   # filled below after classify_topology defined
        "n_steps":         len(initial_plan["model_id"]),
    })

    if not ENABLE_RECURSION:
        return final_answer, trajectory, total_usage, rounds

    current_answer   = final_answer
    final_trajectory = trajectory

    for rec_round in range(1, MAX_RECURSIVE_CALLS + 1):
        remaining = MAX_WORKFLOW_STEPS

        # ── Conductor decides: pass-through or new workflow ──
        try:
            raw_rec, rec_usage = client.generate(
                model=CONDUCTOR_MODEL,
                instructions=build_recursive_conductor_instructions(remaining),
                user_input=build_recursive_conductor_input(question, current_answer),
                temperature=CONDUCTOR_TEMPERATURE,
                max_output_tokens=MAX_CONDUCTOR_OUTPUT_TOKENS,
            )
            total_usage[CONDUCTOR_MODEL].add(rec_usage)
            time.sleep(SLEEP_BETWEEN_CALLS)
        except Exception as e:
            rounds.append({
                "round": rec_round, "is_recursive": True,
                "conductor_error": repr(e),
                "recursion_triggered": False, "passed_through": False,
            })
            break

        try:
            rec_plan = parse_recursive_plan(raw_rec)
        except Exception as e:
            rounds.append({
                "round": rec_round, "is_recursive": True,
                "parse_error": repr(e), "raw_plan": raw_rec,
                "recursion_triggered": False, "passed_through": False,
            })
            break

        if rec_plan is None:
            # Conductor chose pass-through
            rounds.append({
                "round": rec_round, "is_recursive": True,
                "recursion_triggered": False, "passed_through": True,
                "raw_plan": raw_rec,
            })
            break

        # ── Execute recursive workflow ───────────────────────
        try:
            new_answer, new_traj, new_usage = execute_workflow(
                client, question, rec_plan,
                parent_response=current_answer,
            )
            for model, u in new_usage.items():
                total_usage[model].add(u)

            rounds.append({
                "round":           rec_round,
                "is_recursive":    True,
                "recursion_triggered": True,
                "passed_through":  False,
                "plan":            rec_plan,
                "trajectory":      new_traj,
                "raw_final_answer": new_answer,
                "n_steps":         len(rec_plan["model_id"]),
            })
            current_answer   = new_answer
            final_trajectory = new_traj
        except Exception as e:
            rounds.append({
                "round": rec_round, "is_recursive": True,
                "recursion_triggered": True, "execution_error": repr(e),
            })
            break

    return current_answer, final_trajectory, total_usage, rounds


# ============================================================
# 6. Answer extraction and judging
# ============================================================

def extract_boxed_answers_from_output(text: str) -> List[str]:
    if not text:
        return []
    answers, marker = [], r"\boxed{"
    start = 0
    while True:
        pos = text.find(marker, start)
        if pos == -1:
            break
        i, depth = pos + len(marker), 1
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[pos + len(marker):i].strip())
                    start = i + 1
                    break
            i += 1
        else:
            answers.append("")
            start = pos + len(marker)
    return answers


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    boxed = extract_boxed_answers_from_output(text)
    if len(boxed) == 1:
        return boxed[0].strip().strip("$ .")
    for pat in (r"FINAL_ANSWER\s*:\s*(.*)", r"Final answer\s*:\s*(.*)", r"Answer\s*:\s*(.*)"):
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip().splitlines()[0].strip().strip("$ .")
    return text.strip().splitlines()[-1].strip().strip("$ .")


def clean_latex_answer(answer: str) -> str:
    ans = str(answer or "").strip()
    ans = re.sub(r"^FINAL_ANSWER\s*:\s*", "", ans, flags=re.IGNORECASE).strip()
    ans = ans.strip("$ .")
    ans = re.sub(r"^\\\((.*)\\\)$", r"\1", ans).strip()
    ans = re.sub(r"^\\\[(.*)\\\]$", r"\1", ans, flags=re.DOTALL).strip()
    ans = re.sub(r"^(?:verified|answer|final answer)\s*:\s*", "", ans, flags=re.IGNORECASE).strip()
    ans = re.sub(r"^(?:a|A|A\s*\+\s*B|tan\s*A)\s*=\s*", "", ans).strip()
    ans = re.split(r"\s*(?:;|,\s*because\b|\bbecause\b|\bwhere\b)\s*", ans, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    ans = re.sub(r"\bunits?\b", "", ans, flags=re.IGNORECASE).strip()
    for old, new in [("π",r"\pi"),("√",r"\sqrt"),("∛",r"\sqrt[3]"),("≤",r"\le"),("≥",r"\ge")]:
        ans = ans.replace(old, new)
    for fn in ("cot","sec","sin","cos","tan"):
        ans = re.sub(rf"\b{fn}\s+([A-Za-z])\b", rf"\\{fn} \1", ans)
    ans = re.sub(r"(?<!\\)\bpi\b", r"\\pi", ans)
    for cmd in ("cot","sec","sin","cos","tan","pi","sqrt","frac"):
        ans = ans.replace(f"\\\\{cmd}", f"\\{cmd}")
    ans = re.sub(r"^([+-]?)\\pi/(\d+)$",    r"\1\\frac{\\pi}{\2}", ans)
    ans = re.sub(r"^([+-]?)pi/(\d+)$",      r"\1\\frac{\\pi}{\2}", ans)
    ans = re.sub(r"\s+", " ", ans).strip().strip(" .;:,")
    return ans


def latex_final_answer(answer: str) -> str:
    c = clean_latex_answer(answer)
    return f"FINAL_ANSWER: \\boxed{{{c}}}" if c else ""


def recover_final_answer(text: str, trajectory: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    for src_idx, t in enumerate([text] + [s.get("response","") for s in reversed(trajectory or [])]):
        ext = extract_final_answer(t)
        cln = clean_latex_answer(ext)
        if cln:
            return latex_final_answer(cln), {
                "recovered": src_idx != 0 or t != text,
                "source": "final_answer" if src_idx == 0 else f"traj_rev_{src_idx-1}",
                "raw_extracted": ext, "latex_answer": cln,
            }
    return text, {"recovered": False, "source": "none", "raw_extracted": "", "latex_answer": ""}


def is_parseable_math(answer: str) -> bool:
    if not answer or not answer.strip():
        return False
    if not re.compile(r"^[0-9a-zA-Z\\{}\[\]()., _+\-*/^=<>|:!%&]+$").match(answer):
        return False
    stack, pairs = [], {"}":"{", ")":"(", "]":"["}
    for ch in answer:
        if ch in "{([":
            stack.append(ch)
        elif ch in "})]":
            if not stack or stack.pop() != pairs[ch]:
                return False
    if stack or re.search(r"[+\-*/^=]{2,}", answer.replace("--","")):
        return False
    return bool(re.search(r"[0-9a-zA-Z\\]", answer))


def local_format_validation(model_answer: str) -> Dict[str, Any]:
    boxed = extract_boxed_answers_from_output(model_answer)
    if not boxed:
        return {"format_valid": False, "error_type": "missing_boxed_answer",
                "extracted_answer": "", "reason": "No \\boxed{...} found."}
    if len(boxed) > 1:
        return {"format_valid": False, "error_type": "multiple_boxed_answers",
                "extracted_answer": boxed[-1], "reason": "More than one \\boxed{...} found."}
    ext = boxed[0].strip()
    if not ext:
        return {"format_valid": False, "error_type": "empty_boxed_answer",
                "extracted_answer": "", "reason": "Boxed answer is empty."}
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)\.\.\.", ext):
        return {"format_valid": False, "error_type": "unparseable_answer",
                "extracted_answer": ext, "reason": "Decimal approximation with ellipsis."}
    if re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", ext) and "\\" not in ext:
        return {"format_valid": False, "error_type": "non_math_final_answer",
                "extracted_answer": ext, "reason": "Looks like prose not math."}
    if not is_parseable_math(ext):
        return {"format_valid": False, "error_type": "unparseable_answer",
                "extracted_answer": ext, "reason": "Not parseable as math expression."}
    return {"format_valid": True, "error_type": "no_error",
            "extracted_answer": ext, "reason": "Valid single boxed answer."}


def validate_answer_format(client: OpenAITextClient, model_answer: str) -> Tuple[Dict[str, Any], Usage]:
    local = local_format_validation(model_answer)
    return {"method": "local_format_validator", **local}, Usage()


def normalize_math(x: str) -> str:
    x = clean_latex_answer(str(x)).strip().lower()
    x = x.replace("π","\\pi").replace("\\left","").replace("\\right","")
    x = re.sub(r"^x\s*[\\in∈]\s*","",x)
    x = re.sub(r"^\\boxed\{(.*)\}$",r"\1",x)
    x = re.sub(r"^\\frac\{([^{}]+)\}\{([^{}]+)\}$",r"\1/\2",x)
    x = re.sub(r"^([-+]?)\\frac\{\\pi\}\{([^{}]+)\}$",r"\1\\pi/\2",x)
    x = re.sub(r"^([-+]?)\\frac\{([^{}]+)\}\{([^{}]+)\}$",r"\1\2/\3",x)
    return x.replace(" ","").strip("$.")


def exact_match(pred: str, gold: str) -> bool:
    return normalize_math(pred) == normalize_math(gold)


def is_decimal(x: str) -> bool:
    return bool(re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:\.\.\.)?", str(x).strip()))


def needs_symbolic(gold: str) -> bool:
    n = normalize_math(gold)
    return any(k in n for k in ["\\sqrt","sqrt","\\frac","/","\\pi","pi","^","!","\\binom"])


def judge_answer(
    client: OpenAITextClient,
    question: str,
    gold: str,
    model_answer: str,
) -> Tuple[Optional[bool], Dict[str, Any], Usage]:
    ext = extract_final_answer(model_answer)
    if exact_match(ext, gold):
        return True, {"method":"exact","extracted_answer":ext,"reason":"exact normalized match"}, Usage()
    if needs_symbolic(gold) and is_decimal(ext):
        return False, {"method":"strict_symbolic_mismatch","extracted_answer":ext,
                       "reason":"Gold is symbolic; model returned decimal."}, Usage()
    if not USE_LLM_JUDGE:
        return False, {"method":"exact","extracted_answer":ext,"reason":"exact mismatch"}, Usage()

    instr = (
        "You are a strict math answer equivalence judge.\n"
        "Return JSON: {\"correct\":true/false,\"extracted_answer\":\"...\",\"reason\":\"short\"}\n"
        "Be strict. Do not accept decimal for symbolic gold unless problem says so."
    )
    usr = (f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\n"
           f"MODEL RESPONSE:\n{model_answer}\n\nEXTRACTED GUESS:\n{ext}")
    raw, usage = client.generate(JUDGE_MODEL, instr, usr, JUDGE_TEMPERATURE, MAX_JUDGE_OUTPUT_TOKENS)
    try:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        obj = json.loads(m.group(0) if m else raw)
        return bool(obj.get("correct")), {"method":"llm_judge",**obj}, usage
    except Exception:
        return False, {"method":"llm_judge_parse_failed","raw":raw,"extracted_answer":ext}, usage


# ============================================================
# 6.5. Logging helpers
# ============================================================

def error_details(phase: str, error: Exception) -> Dict[str, str]:
    return {"phase": phase, "error_type": type(error).__name__,
            "message": str(error), "repr": repr(error)}


def last_worker_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    traj = row.get("trajectory") or []
    if traj:
        last = traj[-1]
        return {"model_id": last.get("model_id"), "api_model": last.get("api_model"),
                "step": last.get("step"), "role": classify_role(last.get("subtask","")),
                "subtask": last.get("subtask","")}
    plan = row.get("plan") or {}
    mids, subs = plan.get("model_id",[]), plan.get("subtasks",[])
    if not mids:
        return None
    i = len(mids) - 1
    return {"model_id": mids[i],
            "api_model": WORKER_MODELS[mids[i]] if 0 <= mids[i] < len(WORKER_MODELS) else None,
            "step": i,
            "role": classify_role(subs[i]) if i < len(subs) else None,
            "subtask": subs[i] if i < len(subs) else ""}


def rollout_answer_summary(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "question_id":   r.get("question_id"),
        "rollout_id":    r.get("rollout_id"),
        "parse_ok":      r.get("parse_ok"),
        "executed_ok":   r.get("executed_ok"),
        "format_valid":  r.get("format_valid"),
        "is_correct":    r.get("is_correct"),
        "gold_answer":   r.get("gold_answer", r.get("gold")),
        "predicted_answer": r.get("predicted_answer",""),
        "topology":      r.get("topology"),
        "n_steps":       r.get("n_steps"),
        "is_recursive":  r.get("is_recursive", False),
        "n_recursive_rounds_triggered": r.get("n_recursive_rounds_triggered", 0),
        "judge_method":  (r.get("judge") or {}).get("method"),
        "judge_reason":  (r.get("judge") or {}).get("reason"),
        "last_worker":   last_worker_from_row(r),
        "error":         r.get("error"),
    } for r in logs]


# ============================================================
# 7. Exploration metrics
# ============================================================

def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((c/total) * math.log(c/total + 1e-12, 2) for c in counts.values())


# ── 논문 Appendix F.1 기반 키워드 상수 ────────────────────────

_ABDICATION_KEYWORDS = [
    "propose subtasks", "assign work", "direct the other",
    "plan for the subsequent", "subtasks for the other",
    "orchestrate", "tell the other models",
]

_VERIFIER_KW  = ["verify","check","validate","review","ensure","confirm"]
_REFINER_KW   = ["refine","revise","improve","correct","fix"]
_PLANNER_KW   = ["plan","strategy","approach","analyze","analyse","outline","understand","develop"]
_EXECUTOR_KW  = ["implement","execute","code","write the","apply"]
_AGGREGATOR_KW= ["compare","aggregate","combine","resolve"]
_FORMATTER_KW = ["return only final_answer","return only","format","final_answer"]


def classify_topology(
    access_list: List[List[str]],
    subtasks: Optional[List[str]] = None,
) -> str:
    """논문 Appendix F.1 5가지 + independent."""
    if len(access_list) == 1:
        return "single"
    if all(a == [] for a in access_list):
        return "independent"
    if (len(access_list) >= 3
            and access_list[0] == []
            and access_list[1] == []
            and access_list[-1] == ["all"]):
        return "parallel_independent"
    if all(a == ["all"] for a in access_list[1:]):
        if subtasks and any(k in subtasks[0].lower() for k in _ABDICATION_KEYWORDS):
            return "abdication"
        return "chain"
    return "mixed"


def classify_role(subtask: str) -> str:
    """논문 Appendix F.1 role taxonomy."""
    s = subtask.lower()
    if any(k in s for k in _ABDICATION_KEYWORDS):  return "meta_orchestrator"
    if any(k in s for k in _VERIFIER_KW):           return "verifier"
    if any(k in s for k in _REFINER_KW):            return "refiner"
    if any(k in s for k in _PLANNER_KW):            return "planner"
    if any(k in s for k in _EXECUTOR_KW):           return "executor"
    if any(k in s for k in _AGGREGATOR_KW):         return "aggregator"
    if any(k in s for k in _FORMATTER_KW):          return "formatter"
    return "solver"


def compute_cost(usage_by_model: Dict[str, Usage]) -> Dict[str, Any]:
    total, details = 0.0, {}
    for model, u in usage_by_model.items():
        prices = PRICE_PER_1M.get(model)
        if not prices:
            details[model] = {**u.__dict__, "estimated_cost_usd": None}
            continue
        cost = (u.input_tokens/1e6)*prices["input"] + (u.output_tokens/1e6)*prices["output"]
        total += cost
        details[model] = {**u.__dict__, "estimated_cost_usd": cost}
    return {"estimated_total_cost_usd": total, "by_model": details}


def analyze_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """기존 metrics 유지 (backward compat)."""
    valid = [r for r in logs if r.get("parse_ok") and r.get("executed_ok")]
    by_q  = defaultdict(list)
    for r in valid:
        by_q[r["question_id"]].append(r)

    g_model  = Counter()
    g_topo   = Counter()
    g_lens   = []
    corr_all = []
    q_summ   = []

    for qid, rows in by_q.items():
        m_calls = Counter(); m_seqs = Counter(); topos = Counter()
        role_seqs = Counter(); lens = []; answers = Counter(); corr = []
        for r in rows:
            plan = r["plan"]
            mids = tuple(plan["model_id"])
            subs = plan["subtasks"]
            topo = classify_topology(plan["access_list"], subtasks=subs)
            pred = normalize_math(extract_final_answer(r.get("final_answer","")))
            for mid in mids:
                m_calls[mid] += 1; g_model[mid] += 1
            m_seqs[mids] += 1; topos[topo] += 1; g_topo[topo] += 1
            role_seqs[tuple(classify_role(s) for s in subs)] += 1
            lens.append(len(mids)); g_lens.append(len(mids))
            answers[pred] += 1
            if r.get("is_correct") is not None:
                corr.append(bool(r["is_correct"])); corr_all.append(bool(r["is_correct"]))
        q_summ.append({
            "question_id": qid, "n_rollouts": len(rows),
            "accuracy": statistics.mean(corr) if corr else None,
            "agent_selection_entropy": entropy(m_calls),
            "unique_model_sequences":  len(m_seqs),
            "model_sequence_counts":   {str(k): v for k,v in m_seqs.items()},
            "topology_entropy":        entropy(topos),
            "topology_counts":         dict(topos),
            "role_sequence_counts":    {str(k): v for k,v in role_seqs.items()},
            "workflow_length_mean":    statistics.mean(lens),
            "workflow_length_variance":statistics.pvariance(lens) if len(lens)>1 else 0.0,
            "answer_entropy":          entropy(answers),
            "answer_counts":           dict(answers),
        })

    parse_failed = [r for r in logs if r.get("parse_ok") is False]
    exec_failed  = [r for r in logs if r.get("parse_ok") and r.get("executed_ok") is False]
    parse_ok_cnt = len([r for r in logs if r.get("parse_ok")])

    return {
        "n_total_logs":       len(logs),
        "n_valid_logs":       len(valid),
        "n_parse_failed":     len(parse_failed),
        "parse_failure_rate": len(parse_failed)/len(logs) if logs else None,
        "n_execution_failed": len(exec_failed),
        "exec_failure_rate":  len(exec_failed)/max(1,parse_ok_cnt) if logs else None,
        "n_questions":        len(by_q),
        "global_accuracy":    statistics.mean(corr_all) if corr_all else None,
        "global_agent_selection_entropy": entropy(g_model),
        "global_model_call_counts":       dict(g_model),
        "global_topology_entropy":        entropy(g_topo),
        "global_topology_counts":         dict(g_topo),
        "global_workflow_length_mean":    statistics.mean(g_lens) if g_lens else None,
        "global_workflow_length_variance":statistics.pvariance(g_lens) if len(g_lens)>1 else None,
        "question_summaries": q_summ,
    }


# ── 실험 관찰용 exploration report ──────────────────────────────

def compute_exploration_report(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    실험 관찰 지표 전용 리포트.

    1. Agent Selection Entropy
    2. Topology Diversity
    3. Workflow Length Variance
    4. Answer Consistency (per question)
    5. Success / Failure Trajectory Contrast
    6. Recursion Statistics
    7. Role Distribution
    """
    valid = [r for r in logs if r.get("parse_ok") and r.get("executed_ok")]

    # ── 1. Agent Selection Entropy ──────────────────────────────
    g_agents: Counter = Counter()
    pq_agents: Dict[str, Counter] = defaultdict(Counter)
    for r in valid:
        qid = r["question_id"]
        for mid in r["plan"]["model_id"]:
            g_agents[mid] += 1
            pq_agents[qid][mid] += 1

    agent_entropy_report = {
        "global_entropy":       entropy(g_agents),
        "global_counts":        dict(g_agents),
        "max_possible_bits":    round(math.log2(len(WORKER_MODELS)), 4),
        "per_question": {
            qid: {"entropy": round(entropy(c), 4), "counts": dict(c)}
            for qid, c in pq_agents.items()
        },
        "interpretation": (
            "Global entropy measures model-selection diversity across ALL rollouts. "
            f"Max for {len(WORKER_MODELS)} models = {math.log2(len(WORKER_MODELS)):.3f} bits. "
            "High entropy → Conductor uses all workers; low → biased toward one."
        ),
    }

    # ── 2. Topology Diversity ───────────────────────────────────
    g_topo: Counter = Counter()
    pq_topo: Dict[str, Counter] = defaultdict(Counter)
    for r in valid:
        qid  = r["question_id"]
        topo = classify_topology(r["plan"]["access_list"], subtasks=r["plan"].get("subtasks"))
        g_topo[topo] += 1
        pq_topo[qid][topo] += 1

    topology_diversity_report = {
        "global_entropy": round(entropy(g_topo), 4),
        "global_counts":  dict(g_topo),
        "per_question": {
            qid: {"entropy": round(entropy(c), 4), "counts": dict(c)}
            for qid, c in pq_topo.items()
        },
        "interpretation": (
            "High global topology entropy → Conductor adapts structure to problem type. "
            "Low entropy → always uses same topology regardless of problem."
        ),
    }

    # ── 3. Workflow Length Variance ─────────────────────────────
    g_lens = [len(r["plan"]["model_id"]) for r in valid]
    pq_lens: Dict[str, List[int]] = defaultdict(list)
    for r in valid:
        pq_lens[r["question_id"]].append(len(r["plan"]["model_id"]))

    workflow_variance_report = {
        "global_mean":     round(statistics.mean(g_lens), 4) if g_lens else None,
        "global_variance": round(statistics.pvariance(g_lens), 4) if len(g_lens) > 1 else 0.0,
        "global_stdev":    round(statistics.pstdev(g_lens), 4) if len(g_lens) > 1 else 0.0,
        "global_min":      min(g_lens) if g_lens else None,
        "global_max":      max(g_lens) if g_lens else None,
        "length_distribution": dict(Counter(g_lens)),
        "per_question": {
            qid: {
                "mean":     round(statistics.mean(ls), 4),
                "variance": round(statistics.pvariance(ls), 4) if len(ls) > 1 else 0.0,
                "lengths":  ls,
            }
            for qid, ls in pq_lens.items()
        },
        "interpretation": (
            "High variance → Conductor adapts workflow complexity to problem difficulty "
            "(논문 Figure 8: MMLU=2 steps, LiveCodeBench=4-5 steps). "
            "Low variance → fixed strategy."
        ),
    }

    # ── 4. Answer Consistency ───────────────────────────────────
    pq_ans: Dict[str, List[str]] = defaultdict(list)
    for r in valid:
        pred = normalize_math(extract_final_answer(r.get("final_answer", "")))
        pq_ans[r["question_id"]].append(pred)

    answer_consistency_report = {
        "per_question": {
            qid: {
                "entropy":        round(entropy(Counter(ans)), 4),
                "unique_answers": len(set(ans)),
                "n_rollouts":     len(ans),
                "answer_counts":  dict(Counter(ans)),
            }
            for qid, ans in pq_ans.items()
        },
        "interpretation": (
            "Low entropy → consistent answers across rollouts (confident system). "
            "High entropy → inconsistent answers (system is uncertain or exploring)."
        ),
    }

    # ── 5. Success / Failure Trajectory Contrast ────────────────
    judged  = [r for r in valid if r.get("is_correct") is not None]
    success = [r for r in judged if r["is_correct"]]
    failure = [r for r in judged if not r["is_correct"]]

    def _traj_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0}
        lens  = [len(r["plan"]["model_id"]) for r in rows]
        topos = Counter(
            classify_topology(r["plan"]["access_list"], subtasks=r["plan"].get("subtasks"))
            for r in rows
        )
        roles: Counter = Counter()
        a_sel: Counter = Counter()
        rec_triggered = 0
        rec_rounds_list = []
        for r in rows:
            for s in r["plan"]["subtasks"]:
                roles[classify_role(s)] += 1
            for mid in r["plan"]["model_id"]:
                a_sel[mid] += 1
            if r.get("is_recursive"):
                rec_triggered += 1
            rec_rounds_list.append(r.get("n_recursive_rounds_triggered", 0))
        return {
            "n": len(rows),
            "workflow_length": {
                "mean":     round(statistics.mean(lens), 4),
                "variance": round(statistics.pvariance(lens), 4) if len(lens) > 1 else 0.0,
                "distribution": dict(Counter(lens)),
            },
            "topology_distribution": dict(topos),
            "topology_entropy":      round(entropy(topos), 4),
            "role_distribution":     dict(roles),
            "agent_selection":       dict(a_sel),
            "agent_entropy":         round(entropy(a_sel), 4),
            "recursion_trigger_rate": rec_triggered / len(rows),
            "mean_recursive_rounds":  round(statistics.mean(rec_rounds_list), 4) if rec_rounds_list else 0,
        }

    success_failure_contrast = {
        "success": _traj_stats(success),
        "failure": _traj_stats(failure),
        "diff_notes": {
            "workflow_length": "Do successful rollouts use more steps?",
            "topology":        "Do certain topologies (chain/mixed) correlate with success?",
            "verifier_role":   "Does having a verifier step correlate with success?",
            "recursion":       "Does recursion help when triggered?",
            "agent_diversity": "Do successful rollouts use more diverse model selection?",
        },
    }

    # ── 6. Recursion Statistics ─────────────────────────────────
    all_valid       = valid
    rec_triggered   = [r for r in all_valid if r.get("is_recursive", False) and
                       r.get("n_recursive_rounds_triggered", 0) > 0]
    no_rec          = [r for r in all_valid if not r.get("is_recursive", False) or
                       r.get("n_recursive_rounds_triggered", 0) == 0]

    def _acc(rows):
        judged = [r for r in rows if r.get("is_correct") is not None]
        return round(statistics.mean([bool(r["is_correct"]) for r in judged]), 4) if judged else None

    # Agent redistribution in recursive rounds (논문 Figure 10 유사)
    rec_agent_rounds: Dict[int, Counter] = defaultdict(Counter)   # round → model_id counter
    for r in all_valid:
        for rnd in (r.get("rounds") or []):
            if rnd.get("recursion_triggered") and "plan" in rnd:
                for mid in rnd["plan"].get("model_id", []):
                    rec_agent_rounds[rnd["round"]][mid] += 1

    recursion_stats = {
        "enabled":               ENABLE_RECURSION,
        "max_recursive_calls":   MAX_RECURSIVE_CALLS,
        "n_total_valid_rollouts": len(all_valid),
        "n_recursion_triggered": len(rec_triggered),
        "recursion_trigger_rate": round(len(rec_triggered)/len(all_valid), 4) if all_valid else 0,
        "accuracy_with_recursion":    _acc(rec_triggered),
        "accuracy_without_recursion": _acc(no_rec),
        "round_distribution": dict(Counter(r.get("n_recursive_rounds_triggered", 0) for r in all_valid)),
        "agent_redistribution_by_round": {
            f"round_{k}": dict(v) for k, v in sorted(rec_agent_rounds.items())
        },
        "interpretation": (
            "recursion_trigger_rate: fraction of rollouts where Conductor chose to refine. "
            "agent_redistribution_by_round: which models are selected in each recursive round "
            "(analogous to 논문 Figure 10 — Conductor redistributes away from underperforming models)."
        ),
    }

    # ── 7. Role Distribution ────────────────────────────────────
    g_roles: Counter = Counter()
    for r in valid:
        for s in r["plan"]["subtasks"]:
            g_roles[classify_role(s)] += 1

    role_distribution = {
        "global_counts":  dict(g_roles),
        "global_entropy": round(entropy(g_roles), 4),
        "interpretation": (
            "High planner+verifier ratio → Conductor uses structured check strategies. "
            "High solver ratio → mostly direct attempts."
        ),
    }

    return {
        "agent_selection_entropy":   agent_entropy_report,
        "topology_diversity":        topology_diversity_report,
        "workflow_length_variance":  workflow_variance_report,
        "answer_consistency":        answer_consistency_report,
        "success_failure_contrast":  success_failure_contrast,
        "recursion_stats":           recursion_stats,
        "role_distribution":         role_distribution,
    }


# ============================================================
# 8. Main experiment
# ============================================================

def main() -> None:
    api_key = OPENAI_API_KEY
    if api_key == "여기에_본인_API_KEY":
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY를 스크립트 또는 환경 변수로 설정하세요.")

    random.seed(RANDOM_SEED)
    client = OpenAITextClient(api_key=api_key)
    tasks  = load_math500_subset(N_QUESTIONS, RANDOM_SEED)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "conductor_model":        CONDUCTOR_MODEL,
        "dataset":                "HuggingFaceH4/MATH-500",
        "worker_models":          WORKER_MODELS,
        "judge_model":            JUDGE_MODEL,
        "n_questions":            N_QUESTIONS,
        "n_rollouts":             N_ROLLOUTS,
        "random_seed":            RANDOM_SEED,
        "conductor_temperature":  CONDUCTOR_TEMPERATURE,
        "worker_temperature":     WORKER_TEMPERATURE,
        "use_llm_judge":          USE_LLM_JUDGE,
        "few_shot_mode":          FEW_SHOT_MODE if USE_FEW_SHOT_EXAMPLES else "none",
        "enable_recursion":       ENABLE_RECURSION,
        "max_recursive_calls":    MAX_RECURSIVE_CALLS,
        "recursion_discount":     RECURSION_DISCOUNT,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "tasks.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    logs: List[Dict[str, Any]] = []
    total_usage: Dict[str, Usage] = defaultdict(Usage)

    print(f"Loaded {len(tasks)} MATH-500 tasks.")
    print(f"Rollouts: {N_ROLLOUTS} × {len(tasks)} = {N_ROLLOUTS*len(tasks)} trajectories.")
    print(f"Recursion: {'ON (max '+str(MAX_RECURSIVE_CALLS)+' calls)' if ENABLE_RECURSION else 'OFF'}")
    print(f"Output: {run_dir}\n")

    def _save_logs():
        with (run_dir / "logs.jsonl").open("w", encoding="utf-8") as f:
            for item in logs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for task_i, task in enumerate(tasks, start=1):
        print(f"[{task_i}/{len(tasks)}] {task['id']}")
        for rollout_id in range(N_ROLLOUTS):
            row: Dict[str, Any] = {
                "question_id":   task["id"],
                "dataset_index": task.get("dataset_index"),
                "question":      task["question"],
                "gold":          task["gold"],
                "gold_answer":   task["gold"],
                "rollout_id":    rollout_id,
            }

            # ── 1) Conductor: initial plan ───────────────────
            try:
                raw_plan, u = client.generate(
                    model=CONDUCTOR_MODEL,
                    instructions=build_conductor_instructions(),
                    user_input=build_conductor_input(task["question"]),
                    temperature=CONDUCTOR_TEMPERATURE,
                    max_output_tokens=MAX_CONDUCTOR_OUTPUT_TOKENS,
                )
                total_usage[CONDUCTOR_MODEL].add(u)
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                det = error_details("conductor_generate", e)
                row.update({"parse_ok": False, "executed_ok": False,
                            "parse_error": det, "error": f"{det['phase']}: {det['repr']}"})
                logs.append(row); _save_logs(); continue

            try:
                plan = parse_plan(raw_plan)
                row.update({"raw_plan": raw_plan, "plan": plan,
                            "parse_ok": True, "fallback_used": False})
            except Exception as e:
                det = error_details("plan_parse", e)
                row.update({"raw_plan": raw_plan, "parse_ok": False, "executed_ok": False,
                            "parse_error": det, "error": f"{det['phase']}: {det['repr']}"})
                logs.append(row); _save_logs()
                print(f"  [{rollout_id}] parse FAILED: {det['message']}")
                continue

            # ── 2) Execute (with optional recursion) ────────
            try:
                final_answer, final_traj, worker_usage, rounds = execute_with_recursion(
                    client, task["question"], plan
                )
                for model, u in worker_usage.items():
                    total_usage[model].add(u)

                # Determine recursion outcome
                n_rec_triggered = sum(
                    1 for rnd in rounds
                    if rnd.get("is_recursive") and rnd.get("recursion_triggered")
                )
                is_recursive = n_rec_triggered > 0

                # topology & n_steps from initial plan
                init_topo  = classify_topology(plan["access_list"], subtasks=plan["subtasks"])
                init_steps = len(plan["model_id"])

                # Repair final answer
                final_answer, repair_info = recover_final_answer(final_answer, final_traj)
                predicted   = extract_final_answer(final_answer)

                row.update({
                    "executed_ok":      True,
                    "trajectory":       final_traj,
                    "rounds":           rounds,
                    "final_answer":     final_answer,
                    "final_answer_repair": repair_info,
                    "predicted_answer": predicted,
                    "topology":         init_topo,
                    "n_steps":          init_steps,
                    "is_recursive":     is_recursive,
                    "n_recursive_rounds_triggered": n_rec_triggered,
                    "last_worker":      last_worker_from_row({"trajectory": final_traj}),
                })
            except Exception as e:
                det = error_details("worker_execution", e)
                row.update({"executed_ok": False, "execution_error": det,
                            "error": f"{det['phase']}: {det['repr']}"})
                logs.append(row); _save_logs()
                print(f"  [{rollout_id}] exec FAILED: {det['message']}")
                continue

            # ── 3) Format check ──────────────────────────────
            try:
                fmt, fu = validate_answer_format(client, final_answer)
                total_usage[JUDGE_MODEL].add(fu)
                row.update({"format_valid": fmt.get("format_valid") is True, "format_check": fmt})
                if fmt.get("extracted_answer"):
                    row["predicted_answer"] = str(fmt["extracted_answer"])
            except Exception as e:
                det = error_details("format_validation", e)
                row.update({"format_valid": False, "format_error": det})

            if not row.get("format_valid"):
                row.update({"is_correct": False, "judge": {
                    "method": "format_validation",
                    "extracted_answer": row.get("predicted_answer",""),
                    "reason": f"Invalid format: {(row.get('format_check') or row.get('format_error') or {}).get('reason','')}",
                }})
                print(f"  [{rollout_id}] fmt_FAIL  topo={row.get('topology')} "
                      f"steps={row.get('n_steps')} rec={row.get('is_recursive')} "
                      f"gold={task['gold']} pred={row.get('predicted_answer','')}")
                logs.append(row); _save_logs(); continue

            # ── 4) Judge correctness ─────────────────────────
            try:
                correct, judge_info, ju = judge_answer(
                    client, task["question"], task["gold"], final_answer)
                total_usage[JUDGE_MODEL].add(ju)
                row.update({"is_correct": correct, "judge": judge_info})
                print(
                    f"  [{rollout_id}] steps={row['n_steps']} "
                    f"topo={row['topology']} "
                    f"rec={'Y('+str(n_rec_triggered)+')' if is_recursive else 'N'} "
                    f"correct={correct} "
                    f"gold={task['gold']} pred={row.get('predicted_answer','')}"
                )
            except Exception as e:
                det = error_details("judge", e)
                row.update({"is_correct": None, "judge_error": det})

            logs.append(row)
            _save_logs()

        print()  # blank line between questions

    # ── Final outputs ─────────────────────────────────────────
    metrics    = analyze_logs(logs)
    exp_report = compute_exploration_report(logs)
    cost       = compute_cost(total_usage)

    # contrast (kept for backward compat)
    judged  = [r for r in logs if r.get("parse_ok") and r.get("executed_ok")
               and r.get("is_correct") is not None]
    success = [r for r in judged if r["is_correct"]]
    failure = [r for r in judged if not r["is_correct"]]

    def _grp(rows):
        if not rows:
            return {"n": 0}
        lens  = [len(r["plan"]["model_id"]) for r in rows]
        topos = Counter(
            classify_topology(r["plan"]["access_list"], subtasks=r["plan"].get("subtasks"))
            for r in rows
        )
        roles: Counter = Counter()
        for r in rows:
            for s in r["plan"]["subtasks"]:
                roles[classify_role(s)] += 1
        return {
            "n": len(rows),
            "workflow_length_mean":     statistics.mean(lens),
            "workflow_length_variance": statistics.pvariance(lens) if len(lens)>1 else 0.0,
            "topology_counts":  dict(topos),
            "role_counts":      dict(roles),
            "verifier_ratio":   sum(1 for r in rows if "verifier" in [classify_role(s) for s in r["plan"]["subtasks"]]) / len(rows),
        }
    contrast = {"success": _grp(success), "failure": _grp(failure)}

    _save_logs()
    (run_dir/"metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir/"exploration_report.json").write_text(
        json.dumps(exp_report, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir/"success_failure_contrast.json").write_text(
        json.dumps(contrast, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir/"rollout_answer_summary.json").write_text(
        json.dumps(rollout_answer_summary(logs), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir/"usage_cost_estimate.json").write_text(
        json.dumps(cost, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Console summary ───────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT DONE")
    print("="*60)
    print(f"  logs.jsonl              → {run_dir/'logs.jsonl'}")
    print(f"  exploration_report.json → {run_dir/'exploration_report.json'}")
    print(f"  metrics.json            → {run_dir/'metrics.json'}")
    print(f"  success_failure_contrast.json")
    print(f"  rollout_answer_summary.json")
    print(f"  usage_cost_estimate.json")

    print("\n── Key Exploration Metrics ──────────────────────────────")
    ae  = exp_report["agent_selection_entropy"]
    td  = exp_report["topology_diversity"]
    wv  = exp_report["workflow_length_variance"]
    rec = exp_report["recursion_stats"]
    print(json.dumps({
        "global_accuracy":             metrics.get("global_accuracy"),
        "agent_selection_entropy":     round(ae["global_entropy"], 4),
        "agent_max_possible_bits":     ae["max_possible_bits"],
        "topology_entropy":            td["global_entropy"],
        "topology_counts":             td["global_counts"],
        "workflow_length_mean":        wv["global_mean"],
        "workflow_length_variance":    wv["global_variance"],
        "recursion_trigger_rate":      rec["recursion_trigger_rate"],
        "accuracy_with_recursion":     rec["accuracy_with_recursion"],
        "accuracy_without_recursion":  rec["accuracy_without_recursion"],
        "estimated_total_cost_usd":    cost["estimated_total_cost_usd"],
    }, indent=2, ensure_ascii=False))

    print("\n── Success / Failure Contrast ───────────────────────────")
    print(json.dumps(contrast, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
