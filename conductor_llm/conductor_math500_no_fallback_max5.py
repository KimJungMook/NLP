"""
Conductor-style orchestration exploration pilot on AIME25 subset.

Goal
----
This is NOT a full reproduction of the Conductor paper's GRPO training.
It is a minimal behavioral reproduction for observing agent-level exploration / inconsistency.

Setup
-----
- Conductor: GPT-5-mini
- Workers:
    Model 0: GPT-5-mini
    Model 1: GPT-4.1-mini
    Model 2: GPT-5-nano
- Dataset: 10 examples from math-ai/aime25 by default
- Output:
    1) JSONL trajectory logs
    2) metrics JSON
    3) success/failure contrast JSON

Install
-------
pip install openai datasets

Usage
-----
1. Put your OpenAI API key below, or set OPENAI_API_KEY as an environment variable.
2. Run:
   python conductor_math500_no_fallback_max5.py

To reduce cost:
   set N_ROLLOUTS = 3

To use exact matching only without LLM judge:
   set USE_LLM_JUDGE = False
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
# 0. User configuration
# ============================================================

# Option A: paste your key here.
OPENAI_API_KEY = "여기에_본인_API_KEY"
# Option B: leave the above placeholder and set an environment variable:
# export OPENAI_API_KEY="sk-..."

CONDUCTOR_MODEL = "gpt-5-mini"
WORKER_MODELS = ["gpt-5-mini", "gpt-4.1-mini", "gpt-5-nano"]
JUDGE_MODEL = "gpt-5-mini"

N_QUESTIONS = 5
N_ROLLOUTS = 5
RANDOM_SEED = 42

CONDUCTOR_TEMPERATURE = 1.0
WORKER_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.0

MAX_WORKFLOW_STEPS = 5
MAX_CONDUCTOR_OUTPUT_TOKENS = 2048
MAX_WORKER_OUTPUT_TOKENS = 4096
MAX_JUDGE_OUTPUT_TOKENS = 700
MAX_FORMAT_VALIDATOR_OUTPUT_TOKENS = 250

USE_LLM_JUDGE = True
USE_FORMAT_VALIDATOR = False
USE_FEW_SHOT_EXAMPLES = True
FEW_SHOT_MODE = "ood"  # choices: "ood", "in_domain", "none"

# Default is strict AIME25. If HF loading fails and you only want to test the code,
# set this to True. The fallback examples are NOT AIME25.
DEMO_FALLBACK_IF_HF_FAILS = False

OUTPUT_DIR = Path("conductor_aime25_outputs")
SLEEP_BETWEEN_CALLS = 0.15

# Approximate OpenAI prices per 1M tokens. Update if your account has different rates.
# These are only for rough local cost estimation.
PRICE_PER_1M = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
}


# ============================================================
# 1. Data structures
# ============================================================

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens


# ============================================================
# 2. OpenAI model wrapper
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
        """Generate text with a robust fallback for model-specific unsupported params."""
        kwargs = {
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
            # Some models/settings may reject temperature or instructions.
            # Fall back to a single combined input without temperature.
            kwargs.pop("temperature", None)
            try:
                resp = self.client.responses.create(**kwargs)
            except Exception:
                kwargs.pop("reasoning", None)
                combined = f"SYSTEM INSTRUCTIONS:\n{instructions}\n\nUSER INPUT:\n{user_input}"
                resp = self.client.responses.create(
                    model=model,
                    input=combined,
                    max_output_tokens=max_output_tokens,
                    store=False,
                )

        text = getattr(resp, "output_text", "") or ""
        usage = self._extract_usage(resp)
        return text, usage

    @staticmethod
    def _extract_usage(resp: Any) -> Usage:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return Usage()

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) or 0
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


# ============================================================
# 3. AIME25 loading
# ============================================================

def extract_boxed_answer(solution: str) -> Optional[str]:
    """Extract the last \boxed{...} content from a MATH-style solution."""
    if not solution:
        return None

    starts = [m.start() for m in re.finditer(r"\\boxed\{", solution)]
    if not starts:
        return None

    start = starts[-1] + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(solution):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i].strip()
        i += 1
    return None


def load_aime25_subset(n_questions: int, seed: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("math-ai/aime25", split="test")
    except Exception as e:
        if not DEMO_FALLBACK_IF_HF_FAILS:
            raise RuntimeError(
                "Could not load math-ai/aime25.\n"
                "Please run `pip install datasets` and make sure internet access is available.\n"
                "If you only want to test the code flow without AIME25, set "
                "DEMO_FALLBACK_IF_HF_FAILS = True.\n"
                f"Original error: {repr(e)}"
            )
        return demo_fallback_math_questions(n_questions)

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    tasks: List[Dict[str, Any]] = []
    for idx in indices[:n_questions]:
        item = ds[idx]
        problem = item.get("problem") or item.get("question") or ""
        answer = item.get("answer") or extract_boxed_answer(item.get("solution", "")) or ""
        solution = item.get("solution", "")
        subject = item.get("subject", "AIME")
        level = item.get("level", "2025")
        source_id = item.get("id", idx)

        tasks.append(
            {
                "id": f"aime25_{source_id}",
                "dataset_index": idx,
                "type": "math",
                "question": problem,
                "gold": str(answer),
                "solution": solution,
                "subject": subject,
                "level": level,
            }
        )
    return tasks


def demo_fallback_math_questions(n_questions: int) -> List[Dict[str, Any]]:
    examples = [
        ("If 3x + 7 = 22, find x.", "5"),
        ("Compute 12^2 - 5^2.", "119"),
        ("A triangle has angles 40 and 65 degrees. What is the third angle?", "75"),
        ("If a car travels 180 miles in 3 hours, what is its average speed?", "60"),
        ("Simplify 2/3 + 1/6.", "5/6"),
        ("What is the area of a circle with radius 3? Give answer in terms of pi.", "9pi"),
        ("Solve for y: 2y - 4 = 10.", "7"),
        ("Find the next number in the sequence: 2, 4, 8, 16, ?", "32"),
        ("What is 15 percent of 200?", "30"),
        ("If f(x)=x^2+1, compute f(4).", "17"),
    ]
    return [
        {"id": f"demo_{i}", "type": "math", "question": q, "gold": a, "solution": ""}
        for i, (q, a) in enumerate(examples[:n_questions])
    ]


# ============================================================
# 4. Conductor prompt and parsing
# ============================================================

def few_shot_text() -> str:
    if not USE_FEW_SHOT_EXAMPLES or FEW_SHOT_MODE == "none":
        return ""

    if FEW_SHOT_MODE == "ood":
        return """
FEW-SHOT EXAMPLE 1, OOD biomedical QA:
Question: Does brain-derived neurotrophic factor enhance intestinal muscle contraction induced by SP and CGRP? Answer A for Yes or B for No.
Workflow:
{
  "model_id": [1, 0, 2],
  "subtasks": [
    "Answer the biomedical yes/no question independently with the required option letter.",
    "Answer the same biomedical question independently with the required option letter.",
    "Check the two previous answers and provide the correct final option if necessary."
  ],
  "access_list": [[], [], ["all"]]
}

FEW-SHOT EXAMPLE 2, OOD limit problem:
Question: Evaluate lim_{t->0} (1/ln(1+t) + 1/ln(1-t)). Return the final answer in LaTeX.
Workflow:
{
  "model_id": [0, 1, 2],
  "subtasks": [
    "Solve the limit independently, for example using Taylor expansion, and provide a candidate answer.",
    "Solve the limit independently using another valid method and provide a candidate answer.",
    "Check both solutions, resolve discrepancies, and provide the final answer in the requested format."
  ],
  "access_list": [[], [], ["all"]]
}
""".strip()

    if FEW_SHOT_MODE == "in_domain":
        return """
FEW-SHOT EXAMPLE 1, contest math task:
Question: Solve 2x + 5 = 17, then report x.
Workflow:
{
  "model_id": [1, 2],
  "subtasks": [
    "Solve independently with concise algebra and give the candidate value of x.",
    "Verify the candidate against the equation, then return only the final answer."
  ],
  "access_list": [[], ["all"]]
}

FEW-SHOT EXAMPLE 2, contest math task:
Question: Count positive integer pairs (a,b) with a+b=20 and ab divisible by 12.
Workflow:
{
  "model_id": [0, 1, 2],
  "subtasks": [
    "Solve the counting problem independently and give a candidate integer.",
    "Solve independently using a modular or case-based check and give a candidate integer.",
    "Compare the candidates, resolve any discrepancy, then return only the final answer."
  ],
  "access_list": [[], [], ["all"]]
}
""".strip()

    return ""


def build_conductor_instructions() -> str:
    fs = few_shot_text()
    return f"""
You are the Conductor in a multi-agent language model system.

Your job is NOT to solve the problem directly.
Your job is to design a workflow of worker-model calls.

For each user question, output a JSON object equivalent to the paper's three Python lists, with exactly these keys:
- "model_id": list[int]
- "subtasks": list[str]
- "access_list": list[list[str]]

Meaning:
- model_id[i] selects the worker model for step i.
- subtasks[i] is the natural-language instruction for that worker.
- access_list[i] controls what previous worker outputs are visible.

Rules:
1. Use 1 to {MAX_WORKFLOW_STEPS} workflow steps.
2. All three lists must have the same length.
3. Each model_id must be one of the available worker ids.
4. Each access_list item must be either [] or ["all"].
5. [] means the worker sees only the original question and its current subtask.
6. ["all"] means the worker also sees all previous subtasks and responses.
7. A subtask may ask a model to solve from scratch, verify or refine previous work, aggregate independent attempts, or handle final formatting.
8. Choose the number of steps adaptively: simple questions may use one step; harder AIME-style problems often benefit from verification or independent attempts.
9. Encourage useful collaboration by exposing previous work with ["all"] when a later worker should check, refine, or aggregate earlier responses.
10. Keep subtasks focused and targeted to the worker's role.
11. The final subtask must say exactly: "Return only FINAL_ANSWER: \\boxed{{<exact LaTeX answer>}}."
12. Output compact minified JSON only. No markdown. No commentary.

{fs}
""".strip()


def build_conductor_input(question: str) -> str:
    # The conductor sees anonymous model ids and capability descriptions, not actual model names.
    workers = [
        "Model 0: strong general math solver; good for first-pass algebra, geometry, and number theory.",
        "Model 1: careful step-by-step contest math solver; good for independent derivations and edge cases.",
        "Model 2: concise verifier and final-answer formatter; good for checking candidates and boxing the answer.",
    ]
    return f"""
USER QUESTION:
{question}

AVAILABLE WORKER MODELS:
{chr(10).join(workers)}

Return the JSON workflow now.
""".strip()


def parse_plan(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group(0)

    plan = json.loads(text)
    model_id = plan.get("model_id")
    subtasks = plan.get("subtasks")
    access_list = plan.get("access_list")

    if not isinstance(model_id, list) or not isinstance(subtasks, list) or not isinstance(access_list, list):
        raise ValueError("Plan must include list fields: model_id, subtasks, access_list")
    if len(model_id) == 0 or len(model_id) > MAX_WORKFLOW_STEPS:
        raise ValueError("Invalid workflow length")
    if not (len(model_id) == len(subtasks) == len(access_list)):
        raise ValueError("model_id, subtasks, access_list lengths differ")

    subtasks = [
        str(s).strip()
        for s in subtasks
    ]
    subtasks[-1] = "Return only FINAL_ANSWER: \\boxed{<exact LaTeX answer>}."

    cleaned_access = []
    for mid in model_id:
        if not isinstance(mid, int) or mid < 0 or mid >= len(WORKER_MODELS):
            raise ValueError(f"Invalid model id: {mid}")
    for a in access_list:
        if a == []:
            cleaned_access.append([])
        elif a == ["all"] or a == "all":
            cleaned_access.append(["all"])
        else:
            raise ValueError(f"Unsupported access list value: {a}")

    return {"model_id": model_id, "subtasks": subtasks, "access_list": cleaned_access}


# No fallback plan is used: if the Conductor output cannot be parsed,
# the rollout is recorded as a parse failure and skipped, matching the
# paper-style treatment of malformed workflows as failed trajectories.


# ============================================================
# 5. Worker execution
# ============================================================

def worker_description(model_id: int) -> str:
    descriptions = {
        0: "strong general reasoning worker",
        1: "careful step-by-step math worker",
        2: "low-cost verifier and final-answer formatter",
    }
    return descriptions.get(model_id, "language model worker")


def build_worker_instructions(model_id: int, is_final_step: bool) -> str:
    final_rule = (
        "You are the final worker. You must end with exactly one final answer line formatted exactly as:\n"
        "FINAL_ANSWER: \\boxed{<latex_answer>}\n"
        "Before that final line, briefly verify the visible candidate answer or correct it if needed. "
        "The content inside \\boxed{...} must be a LaTeX math expression only. "
        "Use exact LaTeX forms such as \\frac{a}{b}, \\sqrt{3}, x^2, \\pi, or sets/tuples in LaTeX when appropriate. "
        "Do not use decimal approximations for exact values unless the problem explicitly asks for a decimal. "
        "Do not put prose, units, explanations, or more than one \\boxed{...} answer in the final line."
        if is_final_step
        else (
            "You are not the final worker. Solve the assigned contest problem directly. "
            "Give concise reasoning and an explicit candidate answer for later workers to check."
        )
    )
    return f"""
You are a worker in a multi-agent math-solving workflow.
Worker profile: {worker_description(model_id)}.

Follow your assigned subtask. Be concise, but include the key equations, cases, or invariants needed to audit your answer.
{final_rule}
""".strip()


def build_worker_input(
    question: str,
    subtask: str,
    history: List[Dict[str, Any]],
    access: List[str],
) -> str:
    if access == ["all"] and history:
        visible = []
        for h in history:
            visible.append(
                f"[Previous step {h['step']}]\n"
                f"Worker: Model {h['model_id']}\n"
                f"Subtask: {h['subtask']}\n"
                f"Response:\n{h['response']}"
            )
        history_text = "\n\n".join(visible)
    else:
        history_text = "No previous worker outputs are visible."

    return f"""
ORIGINAL MATH QUESTION:
{question}

VISIBLE PREVIOUS WORK:
{history_text}

YOUR CURRENT SUBTASK:
{subtask}
""".strip()


def execute_workflow(
    client: OpenAITextClient,
    question: str,
    plan: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Usage]]:
    history: List[Dict[str, Any]] = []
    usage_by_model: Dict[str, Usage] = defaultdict(Usage)

    for step, (mid, subtask, access) in enumerate(zip(plan["model_id"], plan["subtasks"], plan["access_list"])):
        model = WORKER_MODELS[mid]
        is_final = step == len(plan["model_id"]) - 1
        instructions = build_worker_instructions(mid, is_final_step=is_final)
        user_input = build_worker_input(question, subtask, history, access)

        response, usage = client.generate(
            model=model,
            instructions=instructions,
            user_input=user_input,
            temperature=WORKER_TEMPERATURE,
            max_output_tokens=MAX_WORKER_OUTPUT_TOKENS,
        )
        usage_by_model[model].add(usage)

        history.append(
            {
                "step": step,
                "model_id": mid,
                "api_model": model,
                "subtask": subtask,
                "access": access,
                "response": response,
                "usage": usage.__dict__,
            }
        )
        time.sleep(SLEEP_BETWEEN_CALLS)

    final_answer = history[-1]["response"] if history else ""
    return final_answer, history, usage_by_model


# ============================================================
# 6. Answer extraction and judging
# ============================================================

def extract_boxed_answers_from_output(text: str) -> List[str]:
    if not text:
        return []

    answers = []
    marker = r"\boxed{"
    start_search = 0
    while True:
        marker_pos = text.find(marker, start_search)
        if marker_pos == -1:
            break

        start = marker_pos + len(marker)
        depth = 1
        i = start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[start:i].strip())
                    start_search = i + 1
                    break
            i += 1
        else:
            answers.append("")
            start_search = marker_pos + len(marker)

    return answers


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    boxed_answers = extract_boxed_answers_from_output(text)
    if len(boxed_answers) == 1:
        return boxed_answers[0].strip().strip("$ .")

    patterns = [
        r"FINAL_ANSWER\s*:\s*(.*)",
        r"Final answer\s*:\s*(.*)",
        r"Answer\s*:\s*(.*)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            return ans.splitlines()[0].strip().strip("$ .")
    return text.strip().splitlines()[-1].strip().strip("$ .")


def clean_latex_answer(answer: str) -> str:
    ans = str(answer or "").strip()
    ans = re.sub(r"^FINAL_ANSWER\s*:\s*", "", ans, flags=re.IGNORECASE).strip()
    ans = ans.strip().strip("$ .")
    ans = re.sub(r"^\\\((.*)\\\)$", r"\1", ans).strip()
    ans = re.sub(r"^\\\[(.*)\\\]$", r"\1", ans, flags=re.DOTALL).strip()
    ans = re.sub(r"^(?:verified\s*:|answer\s*:|final answer\s*:)\s*", "", ans, flags=re.IGNORECASE).strip()
    ans = re.sub(r"^(?:a|A|A\s*\+\s*B|tan\s*A)\s*=\s*", "", ans).strip()
    ans = re.split(r"\s*(?:;|,\s*because\b|,\s*check\b|\bbecause\b|\bwhere\b)\s*", ans, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    ans = re.sub(r"\bunits?\b", "", ans, flags=re.IGNORECASE).strip()
    ans = ans.replace("π", r"\pi")
    ans = ans.replace("√", r"\sqrt")
    ans = ans.replace("∛", r"\sqrt[3]")
    ans = ans.replace("≤", r"\le")
    ans = ans.replace("≥", r"\ge")
    ans = re.sub(r"\bcot\s+([A-Za-z])\b", r"\\cot \1", ans)
    ans = re.sub(r"\bsec\s+([A-Za-z])\b", r"\\sec \1", ans)
    ans = re.sub(r"\bsin\s+([A-Za-z])\b", r"\\sin \1", ans)
    ans = re.sub(r"\bcos\s+([A-Za-z])\b", r"\\cos \1", ans)
    ans = re.sub(r"\btan\s+([A-Za-z])\b", r"\\tan \1", ans)
    ans = re.sub(r"(?<!\\)\bpi\b", r"\\pi", ans)
    for cmd in ("cot", "sec", "sin", "cos", "tan", "pi", "sqrt", "frac"):
        ans = ans.replace("\\\\" + cmd, "\\" + cmd)
    ans = re.sub(r"^([+-]?)\\pi/(\d+)$", r"\1\\frac{\\pi}{\2}", ans)
    ans = re.sub(r"^([+-]?)pi/(\d+)$", r"\1\\frac{\\pi}{\2}", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    ans = ans.strip(" .;:,")
    return ans


def latex_final_answer(answer: str) -> str:
    cleaned = clean_latex_answer(answer)
    return f"FINAL_ANSWER: \\boxed{{{cleaned}}}" if cleaned else ""


def recover_final_answer_from_trajectory(final_answer: str, trajectory: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    candidates = [final_answer]
    candidates.extend((step.get("response") or "") for step in reversed(trajectory or []))

    for source_index, text in enumerate(candidates):
        extracted = extract_final_answer(text)
        cleaned = clean_latex_answer(extracted)
        if cleaned:
            return latex_final_answer(cleaned), {
                "recovered": source_index != 0 or text != final_answer,
                "source": "final_answer" if source_index == 0 else f"trajectory_reverse_index_{source_index - 1}",
                "raw_extracted": extracted,
                "latex_answer": cleaned,
            }

    return final_answer, {
        "recovered": False,
        "source": "none",
        "raw_extracted": "",
        "latex_answer": "",
    }


def is_parseable_math_expression(answer: str) -> bool:
    if not answer or not answer.strip():
        return False

    allowed = re.compile(r"^[0-9a-zA-Z\\{}\[\]()., _+\-*/^=<>|:!%&]+$")
    if not allowed.match(answer):
        return False

    stack = []
    pairs = {"}": "{", ")": "(", "]": "["}
    for ch in answer:
        if ch in "{([":
            stack.append(ch)
        elif ch in "})]":
            if not stack or stack.pop() != pairs[ch]:
                return False

    if stack:
        return False

    if re.search(r"[+\-*/^=]{2,}", answer.replace("--", "")):
        return False

    return bool(re.search(r"[0-9a-zA-Z\\]", answer))


def local_format_validation(model_answer: str) -> Dict[str, Any]:
    boxed_answers = extract_boxed_answers_from_output(model_answer)
    if not boxed_answers:
        return {
            "format_valid": False,
            "error_type": "missing_boxed_answer",
            "extracted_answer": "",
            "reason": "No \\boxed{...} final answer was found.",
        }

    if len(boxed_answers) > 1:
        return {
            "format_valid": False,
            "error_type": "multiple_boxed_answers",
            "extracted_answer": boxed_answers[-1],
            "reason": "More than one \\boxed{...} answer was found.",
        }

    extracted = boxed_answers[0].strip()
    if not extracted:
        return {
            "format_valid": False,
            "error_type": "empty_boxed_answer",
            "extracted_answer": "",
            "reason": "The boxed answer is empty.",
        }

    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)\.\.\.", extracted):
        return {
            "format_valid": False,
            "error_type": "unparseable_answer",
            "extracted_answer": extracted,
            "reason": "The boxed answer is a decimal approximation with ellipsis, not an exact LaTeX answer.",
        }

    if re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", extracted) and "\\" not in extracted:
        return {
            "format_valid": False,
            "error_type": "non_math_final_answer",
            "extracted_answer": extracted,
            "reason": "The boxed answer appears to be prose rather than a mathematical expression.",
        }

    if not is_parseable_math_expression(extracted):
        return {
            "format_valid": False,
            "error_type": "unparseable_answer",
            "extracted_answer": extracted,
            "reason": "The boxed answer is not parseable as a mathematical expression.",
        }

    return {
        "format_valid": True,
        "error_type": "no_error",
        "extracted_answer": extracted,
        "reason": "Exactly one non-empty parseable \\boxed{...} answer was found.",
    }


def validate_answer_format(
    client: OpenAITextClient,
    model_answer: str,
) -> Tuple[Dict[str, Any], Usage]:
    local = local_format_validation(model_answer)
    if not USE_FORMAT_VALIDATOR:
        return {"method": "local_format_validator", **local}, Usage()

    instructions = """
You are a strict format validator for MATH benchmark outputs.

Check only whether the model output follows the required answer format.
Do not judge mathematical correctness.

Required format:
- The output must contain exactly one final answer.
- The final answer must be inside \\boxed{...}.
- The boxed answer must not be empty.
- The boxed answer must be parseable as a LaTeX mathematical expression.
- The boxed answer must contain only the mathematical answer, not prose or explanation.
- Exact values must be written in exact LaTeX form, not as decimal approximations, unless the problem explicitly asks for a decimal.

Return JSON only:
{
  "format_valid": true/false,
  "error_type": one of [
    "missing_boxed_answer",
    "empty_boxed_answer",
    "multiple_boxed_answers",
    "unparseable_answer",
    "non_math_final_answer",
    "no_error"
  ],
  "extracted_answer": "...",
  "reason": "brief explanation"
}
""".strip()

    raw, usage = client.generate(
        model=JUDGE_MODEL,
        instructions=instructions,
        user_input=f"MODEL OUTPUT:\n{model_answer}",
        temperature=JUDGE_TEMPERATURE,
        max_output_tokens=MAX_FORMAT_VALIDATOR_OUTPUT_TOKENS,
    )
    try:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        obj = json.loads(m.group(0) if m else raw)
        return {"method": "llm_format_validator", "local_check": local, **obj}, usage
    except Exception:
        return {
            "method": "llm_format_validator_parse_failed",
            "raw": raw,
            **local,
        }, usage


def normalize_math_text(x: str) -> str:
    x = clean_latex_answer(str(x)).strip().lower()
    x = x.replace("π", "\\pi")
    x = x.replace("\\left", "").replace("\\right", "")
    x = re.sub(r"^x\s*\\in\s*", "", x)
    x = re.sub(r"^x\s*∈\s*", "", x)
    x = re.sub(r"^\\boxed\{(.*)\}$", r"\1", x)
    x = re.sub(r"^\\frac\{([^{}]+)\}\{([^{}]+)\}$", r"\1/\2", x)
    x = re.sub(r"^([-+]?)\\frac\{\\pi\}\{([^{}]+)\}$", r"\1\\pi/\2", x)
    x = re.sub(r"^([-+]?)\\frac\{([^{}]+)\}\{([^{}]+)\}$", r"\1\2/\3", x)
    x = x.replace(" ", "")
    x = x.strip("$.")
    return x


def exact_match(pred: str, gold: str) -> bool:
    return normalize_math_text(pred) == normalize_math_text(gold)


def is_plain_decimal_number(x: str) -> bool:
    return bool(re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:\.\.\.)?", str(x).strip()))


def gold_requires_exact_symbolic_form(gold: str) -> bool:
    normalized = normalize_math_text(gold)
    symbolic_markers = ["\\sqrt", "sqrt", "\\frac", "/", "\\pi", "pi", "^", "!", "\\binom"]
    return any(marker in normalized for marker in symbolic_markers)


def judge_answer(
    client: OpenAITextClient,
    question: str,
    gold: str,
    model_answer: str,
) -> Tuple[Optional[bool], Dict[str, Any], Usage]:
    extracted = extract_final_answer(model_answer)
    if exact_match(extracted, gold):
        return True, {"method": "exact", "extracted_answer": extracted, "reason": "exact normalized match"}, Usage()

    if gold_requires_exact_symbolic_form(gold) and is_plain_decimal_number(extracted):
        return (
            False,
            {
                "method": "strict_symbolic_mismatch",
                "extracted_answer": extracted,
                "reason": "Gold answer requires an exact symbolic form, but the model returned a decimal approximation.",
            },
            Usage(),
        )

    if not USE_LLM_JUDGE:
        return False, {"method": "exact", "extracted_answer": extracted, "reason": "exact normalized mismatch"}, Usage()

    instructions = """
You are a strict math answer equivalence judge.
Determine whether the model answer is mathematically equivalent to the gold answer.
Return valid JSON only:
{"correct": true/false, "extracted_answer": "...", "reason": "short reason"}
Do not solve the full problem unless needed for equivalence. Be strict about non-equivalent answers.
The extracted_answer should preserve the model's LaTeX final answer exactly when present.
If the gold answer is an exact symbolic expression such as a radical, fraction, pi expression, power, or factorial,
do not accept a decimal approximation as correct unless the problem explicitly asks for a decimal approximation.
""".strip()

    user_input = f"""
QUESTION:
{question}

GOLD ANSWER:
{gold}

MODEL FINAL RESPONSE:
{model_answer}

EXTRACTED ANSWER GUESS:
{extracted}
""".strip()

    raw, usage = client.generate(
        model=JUDGE_MODEL,
        instructions=instructions,
        user_input=user_input,
        temperature=JUDGE_TEMPERATURE,
        max_output_tokens=MAX_JUDGE_OUTPUT_TOKENS,
    )
    try:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        obj = json.loads(m.group(0) if m else raw)
        return bool(obj.get("correct")), {"method": "llm_judge", **obj}, usage
    except Exception:
        return False, {"method": "llm_judge_parse_failed", "raw": raw, "extracted_answer": extracted}, usage


# ============================================================
# 6.5. Logging helpers
# ============================================================

def error_details(phase: str, error: Exception) -> Dict[str, str]:
    return {
        "phase": phase,
        "error_type": type(error).__name__,
        "message": str(error),
        "repr": repr(error),
    }


def last_worker_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    trajectory = row.get("trajectory") or []
    if trajectory:
        last = trajectory[-1]
        return {
            "model_id": last.get("model_id"),
            "api_model": last.get("api_model"),
            "step": last.get("step"),
            "role": classify_role(last.get("subtask", "")),
            "subtask": last.get("subtask", ""),
        }

    plan = row.get("plan") or {}
    model_ids = plan.get("model_id") or []
    subtasks = plan.get("subtasks") or []
    if not model_ids:
        return None
    last_idx = len(model_ids) - 1
    model_id = model_ids[last_idx]
    return {
        "model_id": model_id,
        "api_model": WORKER_MODELS[model_id] if isinstance(model_id, int) and 0 <= model_id < len(WORKER_MODELS) else None,
        "step": last_idx,
        "role": classify_role(subtasks[last_idx]) if last_idx < len(subtasks) else None,
        "subtask": subtasks[last_idx] if last_idx < len(subtasks) else "",
    }


def rollout_answer_summary(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for row in logs:
        summary.append(
            {
                "question_id": row.get("question_id"),
                "dataset_index": row.get("dataset_index"),
                "rollout_id": row.get("rollout_id"),
                "parse_ok": row.get("parse_ok"),
                "executed_ok": row.get("executed_ok"),
                "format_valid": row.get("format_valid"),
                "is_correct": row.get("is_correct"),
                "gold_answer": row.get("gold_answer", row.get("gold")),
                "predicted_answer": row.get("predicted_answer", ""),
                "raw_final_answer": row.get("raw_final_answer", row.get("final_answer", "")),
                "final_answer_repair": row.get("final_answer_repair"),
                "format_error_type": (row.get("format_check") or {}).get("error_type"),
                "format_reason": (row.get("format_check") or row.get("format_error") or {}).get("reason"),
                "judge_extracted_answer": (row.get("judge") or {}).get("extracted_answer"),
                "judge_method": (row.get("judge") or {}).get("method"),
                "judge_reason": (row.get("judge") or {}).get("reason"),
                "last_worker": last_worker_from_row(row),
                "parse_error": row.get("parse_error"),
                "error": row.get("error"),
            }
        )
    return summary


# ============================================================
# 7. Exploration metrics
# ============================================================

def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for c in counts.values():
        p = c / total
        out -= p * math.log(p + 1e-12, 2)
    return out


def classify_topology(access_list: List[List[str]]) -> str:
    if len(access_list) == 1:
        return "single"
    if all(a == [] for a in access_list):
        return "independent"
    if len(access_list) >= 3 and access_list[0] == [] and access_list[1] == [] and access_list[-1] == ["all"]:
        return "parallel_aggregate"
    if all(a == ["all"] for a in access_list[1:]):
        return "chain"
    return "mixed"


def classify_role(subtask: str) -> str:
    s = subtask.lower()
    if any(k in s for k in ["verify", "check", "validate", "review", "correct"]):
        return "verifier"
    if any(k in s for k in ["plan", "strategy", "approach", "analyze", "outline"]):
        return "planner"
    if any(k in s for k in ["compare", "aggregate", "combine", "resolve"]):
        return "aggregator"
    if any(k in s for k in ["format", "final"]):
        return "formatter"
    return "solver"


def compute_cost(usage_by_model: Dict[str, Usage]) -> Dict[str, Any]:
    total = 0.0
    details = {}
    for model, usage in usage_by_model.items():
        prices = PRICE_PER_1M.get(model)
        if not prices:
            details[model] = {**usage.__dict__, "estimated_cost_usd": None}
            continue
        cost = (usage.input_tokens / 1_000_000) * prices["input"] + (usage.output_tokens / 1_000_000) * prices["output"]
        total += cost
        details[model] = {**usage.__dict__, "estimated_cost_usd": cost}
    return {"estimated_total_cost_usd": total, "by_model": details}


def analyze_logs(logs: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    valid = [r for r in logs if r.get("parse_ok") and r.get("executed_ok")]
    by_q = defaultdict(list)
    for r in valid:
        by_q[r["question_id"]].append(r)

    global_model_calls = Counter()
    global_topologies = Counter()
    global_lengths = []
    correctness_values = []
    question_summaries = []

    for qid, rows in by_q.items():
        model_calls = Counter()
        model_sequences = Counter()
        topologies = Counter()
        role_sequences = Counter()
        lengths = []
        answers = Counter()
        corrects = []

        for r in rows:
            plan = r["plan"]
            mids = tuple(plan["model_id"])
            access = plan["access_list"]
            roles = tuple(classify_role(s) for s in plan["subtasks"])
            topology = classify_topology(access)
            pred = normalize_math_text(extract_final_answer(r.get("final_answer", "")))

            for mid in mids:
                model_calls[mid] += 1
                global_model_calls[mid] += 1

            model_sequences[mids] += 1
            topologies[topology] += 1
            global_topologies[topology] += 1
            role_sequences[roles] += 1
            lengths.append(len(mids))
            global_lengths.append(len(mids))
            answers[pred] += 1

            if r.get("is_correct") is not None:
                corrects.append(bool(r["is_correct"]))
                correctness_values.append(bool(r["is_correct"]))

        question_summaries.append(
            {
                "question_id": qid,
                "n_rollouts": len(rows),
                "accuracy": statistics.mean(corrects) if corrects else None,
                "agent_selection_entropy": entropy(model_calls),
                "unique_model_sequences": len(model_sequences),
                "model_sequence_counts": {str(k): v for k, v in model_sequences.items()},
                "topology_entropy": entropy(topologies),
                "topology_counts": dict(topologies),
                "role_sequence_counts": {str(k): v for k, v in role_sequences.items()},
                "workflow_length_mean": statistics.mean(lengths),
                "workflow_length_variance": statistics.pvariance(lengths) if len(lengths) > 1 else 0.0,
                "answer_entropy": entropy(answers),
                "answer_counts": dict(answers),
            }
        )

    parse_failed = [r for r in logs if r.get("parse_ok") is False]
    execution_failed = [r for r in logs if r.get("parse_ok") and r.get("executed_ok") is False]

    metrics = {
        "n_total_logs": len(logs),
        "n_valid_logs": len(valid),
        "n_parse_failed_logs": len(parse_failed),
        "parse_failure_rate": len(parse_failed) / len(logs) if logs else None,
        "n_execution_failed_logs": len(execution_failed),
        "execution_failure_rate_among_parse_ok": len(execution_failed) / max(1, len([r for r in logs if r.get("parse_ok")])) if logs else None,
        "n_questions": len(by_q),
        "global_accuracy": statistics.mean(correctness_values) if correctness_values else None,
        "global_agent_selection_entropy": entropy(global_model_calls),
        "global_model_call_counts": dict(global_model_calls),
        "global_topology_entropy": entropy(global_topologies),
        "global_topology_counts": dict(global_topologies),
        "global_workflow_length_mean": statistics.mean(global_lengths) if global_lengths else None,
        "global_workflow_length_variance": statistics.pvariance(global_lengths) if len(global_lengths) > 1 else None,
        "question_summaries": question_summaries,
    }

    # Success/failure contrast
    judged = [r for r in valid if r.get("is_correct") is not None]
    success = [r for r in judged if r["is_correct"]]
    failure = [r for r in judged if not r["is_correct"]]

    def group_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0}
        lengths = [len(r["plan"]["model_id"]) for r in rows]
        topologies = Counter(classify_topology(r["plan"]["access_list"]) for r in rows)
        role_counts = Counter()
        last_worker_counts = Counter()
        last_worker_role_counts = Counter()
        last_worker_details = Counter()
        verifier_count = 0
        for r in rows:
            roles = [classify_role(s) for s in r["plan"]["subtasks"]]
            role_counts.update(roles)
            if "verifier" in roles:
                verifier_count += 1
            last_worker = last_worker_from_row(r)
            if last_worker:
                model_id = last_worker.get("model_id")
                api_model = last_worker.get("api_model")
                role = last_worker.get("role")
                last_worker_counts[f"Model {model_id} ({api_model})"] += 1
                if role:
                    last_worker_role_counts[role] += 1
                last_worker_details[f"Model {model_id} ({api_model}) as {role}"] += 1
        return {
            "n": len(rows),
            "workflow_length_mean": statistics.mean(lengths),
            "workflow_length_variance": statistics.pvariance(lengths) if len(lengths) > 1 else 0.0,
            "topology_counts": dict(topologies),
            "role_counts": dict(role_counts),
            "verifier_ratio": verifier_count / len(rows),
            "last_worker_counts": dict(last_worker_counts),
            "last_worker_role_counts": dict(last_worker_role_counts),
            "last_worker_details": dict(last_worker_details),
        }

    contrast = {"success": group_stats(success), "failure": group_stats(failure)}
    return metrics, contrast


# ============================================================
# 8. Main experiment
# ============================================================

def main() -> None:
    api_key = OPENAI_API_KEY
    if api_key == "여기에_본인_API_KEY":
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in the script or as an environment variable.")

    random.seed(RANDOM_SEED)
    client = OpenAITextClient(api_key=api_key)
    tasks = load_aime25_subset(N_QUESTIONS, RANDOM_SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "conductor_model": CONDUCTOR_MODEL,
        "dataset": "math-ai/aime25",
        "worker_models": WORKER_MODELS,
        "judge_model": JUDGE_MODEL,
        "n_questions": N_QUESTIONS,
        "n_rollouts": N_ROLLOUTS,
        "random_seed": RANDOM_SEED,
        "conductor_temperature": CONDUCTOR_TEMPERATURE,
        "worker_temperature": WORKER_TEMPERATURE,
        "use_llm_judge": USE_LLM_JUDGE,
        "use_format_validator": USE_FORMAT_VALIDATOR,
        "few_shot_mode": FEW_SHOT_MODE if USE_FEW_SHOT_EXAMPLES else "none",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "tasks.json").write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    logs: List[Dict[str, Any]] = []
    total_usage_by_model: Dict[str, Usage] = defaultdict(Usage)

    print(f"Loaded {len(tasks)} AIME25 tasks.")
    print(f"Running {N_ROLLOUTS} rollouts per task = {len(tasks) * N_ROLLOUTS} trajectories.")
    print(f"Output directory: {run_dir}")

    for task_i, task in enumerate(tasks, start=1):
        print(f"\n[{task_i}/{len(tasks)}] {task['id']}")
        for rollout_id in range(N_ROLLOUTS):
            row: Dict[str, Any] = {
                "question_id": task["id"],
                "dataset_index": task.get("dataset_index"),
                "question": task["question"],
                "gold": task["gold"],
                "gold_answer": task["gold"],
                "rollout_id": rollout_id,
            }

            # 1) Conductor plan
            try:
                raw_plan, usage = client.generate(
                    model=CONDUCTOR_MODEL,
                    instructions=build_conductor_instructions(),
                    user_input=build_conductor_input(task["question"]),
                    temperature=CONDUCTOR_TEMPERATURE,
                    max_output_tokens=MAX_CONDUCTOR_OUTPUT_TOKENS,
                )
                total_usage_by_model[CONDUCTOR_MODEL].add(usage)
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                details = error_details("conductor_generate", e)
                row.update(
                    {
                        "parse_ok": False,
                        "executed_ok": False,
                        "parse_error": details,
                        "error": f"{details['phase']} error: {details['repr']}",
                    }
                )
                logs.append(row)
                print(f"  rollout {rollout_id}: conductor generation failed ({details['error_type']}: {details['message']})")
                continue

            try:
                plan = parse_plan(raw_plan)
                row.update(
                    {
                        "raw_plan": raw_plan,
                        "plan": plan,
                        "parse_ok": True,
                        "parse_recovered": False,
                        "fallback_used": False,
                    }
                )
            except Exception as e:
                details = error_details("conductor_plan_parse", e)
                row.update(
                    {
                        "raw_plan": raw_plan,
                        "parse_ok": False,
                        "executed_ok": False,
                        "parse_recovered": False,
                        "fallback_used": False,
                        "parse_error": details,
                        "error": f"{details['phase']} error: {details['repr']}",
                    }
                )
                logs.append(row)
                print(
                    f"  rollout {rollout_id}: plan parse failed "
                    f"({details['error_type']}: {details['message']}); no fallback used"
                )

                # Save incrementally to avoid losing progress.
                with (run_dir / "logs.jsonl").open("w", encoding="utf-8") as f:
                    for item in logs:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            # 2) Execute worker workflow
            try:
                final_answer, trajectory, worker_usage = execute_workflow(client, task["question"], plan)
                for model, usage in worker_usage.items():
                    total_usage_by_model[model].add(usage)
                raw_final_answer = final_answer
                final_answer, final_answer_repair = recover_final_answer_from_trajectory(final_answer, trajectory)
                predicted_answer = extract_final_answer(final_answer)
                row.update(
                    {
                        "executed_ok": True,
                        "trajectory": trajectory,
                        "raw_final_answer": raw_final_answer,
                        "final_answer": final_answer,
                        "final_answer_repair": final_answer_repair,
                        "predicted_answer": predicted_answer,
                        "last_worker": last_worker_from_row({"trajectory": trajectory}),
                    }
                )
            except Exception as e:
                details = error_details("worker_execution", e)
                row.update(
                    {
                        "executed_ok": False,
                        "execution_error": details,
                        "error": f"{details['phase']} error: {details['repr']}",
                    }
                )
                logs.append(row)
                print(f"  rollout {rollout_id}: execution failed ({details['error_type']}: {details['message']})")
                continue

            # 3) Validate required answer format
            try:
                format_check, format_usage = validate_answer_format(client, final_answer)
                total_usage_by_model[JUDGE_MODEL].add(format_usage)
                row.update({"format_valid": format_check.get("format_valid") is True, "format_check": format_check})
                if format_check.get("extracted_answer"):
                    row["predicted_answer"] = str(format_check["extracted_answer"])
            except Exception as e:
                details = error_details("format_validation", e)
                row.update({"format_valid": False, "format_error": details})
                print(f"  rollout {rollout_id}: format validation failed ({details['error_type']}: {details['message']})")

            if row.get("format_valid") is False:
                row.update(
                    {
                        "is_correct": False,
                        "judge": {
                            "method": "format_validation",
                            "extracted_answer": row.get("predicted_answer", ""),
                            "reason": f"Invalid final answer format: {(row.get('format_check') or row.get('format_error') or {}).get('reason', 'format validation failed')}",
                        },
                    }
                )
                print(
                    f"  rollout {rollout_id}: format_valid=False, "
                    f"error_type={(row.get('format_check') or {}).get('error_type', 'validator_error')}, "
                    f"gold={task['gold']}, pred={row.get('predicted_answer', '')}"
                )
                logs.append(row)

                # Save incrementally to avoid losing progress.
                with (run_dir / "logs.jsonl").open("w", encoding="utf-8") as f:
                    for item in logs:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            # 4) Judge correctness
            try:
                correct, judge_info, judge_usage = judge_answer(client, task["question"], task["gold"], final_answer)
                total_usage_by_model[JUDGE_MODEL].add(judge_usage)
                row.update({"is_correct": correct, "judge": judge_info})
                print(
                    f"  rollout {rollout_id}: len={len(plan['model_id'])}, "
                    f"topology={classify_topology(plan['access_list'])}, format_valid={row.get('format_valid')}, correct={correct}, "
                    f"gold={task['gold']}, pred={row.get('predicted_answer', '')}"
                )
            except Exception as e:
                details = error_details("judge", e)
                row.update({"is_correct": None, "judge_error": details})
                print(
                    f"  rollout {rollout_id}: judge failed ({details['error_type']}: {details['message']}), "
                    f"gold={task['gold']}, pred={row.get('predicted_answer', '')}"
                )

            logs.append(row)

            # Save incrementally to avoid losing progress.
            with (run_dir / "logs.jsonl").open("w", encoding="utf-8") as f:
                for item in logs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics, contrast = analyze_logs(logs)
    cost = compute_cost(total_usage_by_model)

    (run_dir / "logs.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in logs),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "success_failure_contrast.json").write_text(json.dumps(contrast, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "rollout_answer_summary.json").write_text(
        json.dumps(rollout_answer_summary(logs), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "usage_cost_estimate.json").write_text(json.dumps(cost, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Logs: {run_dir / 'logs.jsonl'}")
    print(f"Metrics: {run_dir / 'metrics.json'}")
    print(f"Success/failure contrast: {run_dir / 'success_failure_contrast.json'}")
    print(f"Rollout answer summary: {run_dir / 'rollout_answer_summary.json'}")
    print(f"Usage/cost estimate: {run_dir / 'usage_cost_estimate.json'}")

    print("\n=== Summary ===")
    compact = {
        "global_accuracy": metrics.get("global_accuracy"),
        "global_agent_selection_entropy": metrics.get("global_agent_selection_entropy"),
        "global_topology_entropy": metrics.get("global_topology_entropy"),
        "global_topology_counts": metrics.get("global_topology_counts"),
        "global_workflow_length_mean": metrics.get("global_workflow_length_mean"),
        "global_workflow_length_variance": metrics.get("global_workflow_length_variance"),
        "estimated_total_cost_usd": cost.get("estimated_total_cost_usd"),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))

    print("\n=== Success / Failure Contrast ===")
    print(json.dumps(contrast, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
