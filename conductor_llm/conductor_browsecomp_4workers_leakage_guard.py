"""
Conductor-style orchestration exploration pilot on BrowseComp.

목적
----
- MATH-500용 Conductor-style workflow 실험을 BrowseComp web-browsing task로 전환
- 논문 방식의 topology 유지:
    1) single
    2) sequential chain
    3) parallel/tree independent search
    4) mixed topology
    5) recursive refinement
- Worker pool에 browser/search tool worker를 추가
- 관찰 지표:
    1) Agent selection entropy
    2) Topology diversity
    3) Workflow length variance
    4) Success / failure trajectory contrast
    5) Browser/tool usage statistics
    6) Evidence/candidate statistics
    7) Recursion statistics

Install
-------
pip install openai datasets

BrowseComp data
---------------
이 스크립트는 기본적으로 로컬 BrowseComp json/jsonl/csv 파일을 읽도록 설계했습니다.
BROWSECOMP_PATH 환경변수나 아래 BROWSECOMP_PATH 값을 지정하세요.

지원되는 필드명 예시:
- question / problem / prompt / input
- answer / gold / target / reference_answer
- id / question_id

Usage
-----
export OPENAI_API_KEY="..."
export BROWSECOMP_PATH="/path/to/browsecomp.jsonl"
python conductor_browsecomp.py
"""

from __future__ import annotations

import csv
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
# Four worker slots total. Model 3 is an additional gpt-5-mini used as the browser/search tool worker.
WORKER_MODELS = ["gpt-5-mini", "gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini"]
JUDGE_MODEL = "gpt-5-mini"

# Browser/search worker uses the 4th worker slot.
BROWSER_WORKER_ID = 3
BROWSER_MODEL = WORKER_MODELS[BROWSER_WORKER_ID]
WEB_SEARCH_TOOL_TYPE = "web_search"  # new Responses API hosted web search tool

# Dataset leakage guard: do not let the browser/verifier use benchmark mirrors
# or pages that expose gold/reference answers for BrowseComp items.
LEAKAGE_GUARD_ENABLED = True
LEAKAGE_REJECT_DOMAINS = [
    "huggingface.co/datasets",
    "kaggle.com/datasets",
    "github.com",
    "raw.githubusercontent.com",
]
LEAKAGE_REJECT_TERMS = [
    "browsecomp",
    "gold",
    "gold_answer",
    "reference_answer",
    "target",
    "answer field",
    "\"answer\"",
    "dataset viewer",
    "benchmark",
    "test_set",
    "test set",
]

# BrowseComp local file path. If empty, env BROWSECOMP_PATH is used.
BROWSECOMP_PATH = ""

N_QUESTIONS = 1
N_ROLLOUTS = 5
RANDOM_SEED = 40

CONDUCTOR_TEMPERATURE = 1.0
WORKER_TEMPERATURE = 0.2
BROWSER_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.0

MAX_WORKFLOW_STEPS = 5
MAX_CONDUCTOR_OUTPUT_TOKENS = 2048
MAX_WORKER_OUTPUT_TOKENS = 4096
MAX_BROWSER_OUTPUT_TOKENS = 4096
MAX_JUDGE_OUTPUT_TOKENS = 700

USE_LLM_JUDGE = True
USE_FEW_SHOT_EXAMPLES = True
FEW_SHOT_MODE = "browsecomp"  # "browsecomp" | "none"

ENABLE_RECURSION = True
MAX_RECURSIVE_CALLS = 2

DEMO_FALLBACK_IF_DATA_FAILS = False
OUTPUT_DIR = Path("conductor_browsecomp_outputs")
SLEEP_BETWEEN_CALLS = 0.15

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
        use_web_search: bool = False,
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
        if use_web_search:
            # Hosted web_search is not compatible with reasoning.effort="minimal".
            # Remove reasoning settings before the first web_search attempt.
            kwargs.pop("reasoning", None)
            kwargs["tools"] = [{"type": WEB_SEARCH_TOOL_TYPE}]

        try:
            resp = self.client.responses.create(**kwargs)
        except Exception as e1:
            # Some accounts / model combinations may still accept web_search_preview.
            if use_web_search:
                kwargs["tools"] = [{"type": "web_search_preview"}]
                try:
                    resp = self.client.responses.create(**kwargs)
                except Exception:
                    # Last fallback before disabling the web tool: remove temperature/reasoning while keeping web tool.
                    kwargs.pop("temperature", None)
                    kwargs.pop("reasoning", None)
                    try:
                        resp = self.client.responses.create(**kwargs)
                    except Exception:
                        # Final fallback: no web tool. This will usually hurt BrowseComp accuracy,
                        # but preserves logging and makes the failure mode visible.
                        kwargs.pop("tools", None)
                        kwargs.pop("reasoning", None)
                        combined = f"SYSTEM INSTRUCTIONS:\n{instructions}\n\nUSER INPUT:\n{user_input}\n\nNOTE: web_search tool failed with: {repr(e1)}"
                        resp = self.client.responses.create(
                            model=model,
                            input=combined,
                            max_output_tokens=max_output_tokens,
                            store=False,
                        )
            else:
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
        u = getattr(resp, "usage", None)
        if u is None:
            return Usage()
        inp = getattr(u, "input_tokens", 0) or 0
        out = getattr(u, "output_tokens", 0) or 0
        tot = getattr(u, "total_tokens", inp + out) or 0
        return Usage(inp, out, tot)


# ============================================================
# 3. BrowseComp loading
# ============================================================

QUESTION_KEYS = ["question", "problem", "prompt", "input", "query"]
ANSWER_KEYS = ["answer", "gold", "target", "reference_answer", "final_answer"]
ID_KEYS = ["id", "question_id", "uid", "uuid"]


def _first_key(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip():
            return str(d[k]).strip()
    return ""


def _normalize_task(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    q = _first_key(raw, QUESTION_KEYS)
    a = _first_key(raw, ANSWER_KEYS)
    if not q or not a:
        raise ValueError(f"Could not find question/answer fields in row {idx}: keys={list(raw.keys())}")
    qid = _first_key(raw, ID_KEYS) or f"browsecomp_{idx}"
    return {
        "id": qid,
        "dataset_index": idx,
        "type": "browsecomp",
        "question": q,
        "gold": a,
        "gold_answer": a,
        "raw": raw,
    }


def load_browsecomp_subset(n_questions: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    path = BROWSECOMP_PATH or os.getenv("BROWSECOMP_PATH", "")
    rows: List[Dict[str, Any]] = []

    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"BROWSECOMP_PATH does not exist: {p}")
        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        elif p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                rows = obj
            elif isinstance(obj, dict):
                for k in ["data", "examples", "samples", "questions"]:
                    if isinstance(obj.get(k), list):
                        rows = obj[k]
                        break
                if not rows:
                    raise ValueError("JSON file must be a list or contain data/examples/samples/questions list.")
            else:
                raise ValueError("Unsupported JSON structure.")
        elif p.suffix.lower() == ".csv":
            with p.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            raise ValueError("Supported BrowseComp file types: .jsonl, .json, .csv")
    else:
        # Best-effort HuggingFace loading. BrowseComp is officially in OpenAI simple-evals;
        # this fallback may work only if a mirror dataset is available in your environment.
        try:
            from datasets import load_dataset
            for name in ["openai/browsecomp", "OpenAI/BrowseComp", "browsecomp"]:
                try:
                    ds = load_dataset(name, split="test")
                    rows = [dict(x) for x in ds]
                    break
                except Exception:
                    continue
        except Exception:
            rows = []

    if not rows:
        if not DEMO_FALLBACK_IF_DATA_FAILS:
            raise RuntimeError(
                "Could not load BrowseComp data. Set BROWSECOMP_PATH to a local BrowseComp json/jsonl/csv file.\n"
                "Example: export BROWSECOMP_PATH=/path/to/browsecomp.jsonl\n"
                "For smoke testing only, set DEMO_FALLBACK_IF_DATA_FAILS=True."
            )
        rows = [
            {"id": "demo_browse_0", "question": "What is the official name of the OpenAI benchmark that measures agents locating hard-to-find information on the web?", "answer": "BrowseComp"},
            {"id": "demo_browse_1", "question": "Who wrote the novel Frankenstein?", "answer": "Mary Shelley"},
            {"id": "demo_browse_2", "question": "What chemical element has the symbol W?", "answer": "Tungsten"},
        ]

    tasks = []
    for idx, raw in enumerate(rows):
        try:
            tasks.append(_normalize_task(raw, idx))
        except Exception:
            continue

    indices = list(range(len(tasks)))
    if seed is None:
        random.shuffle(indices)
    else:
        random.Random(seed).shuffle(indices)
    return [tasks[i] for i in indices[:n_questions]]


# ============================================================
# 4. Conductor prompts
# ============================================================


def few_shot_text() -> str:
    if not USE_FEW_SHOT_EXAMPLES or FEW_SHOT_MODE == "none":
        return ""
    return """
FEW-SHOT EXAMPLE 1, SINGLE-SHOT direct lookup:
Question: What is the exact title of the webpage at a given URL?
Workflow:
{"model_id":[3],"subtasks":["Open or search the given URL and return the exact webpage title as FINAL_ANSWER: <exact answer string>. Do not explain."],"access_list":[[]]}

FEW-SHOT EXAMPLE 2, SHORT CHAIN for one clear search path:
Question: Identify the organization that published a page announcing a dataset named X.
Workflow:
{"model_id":[0,3,2],"subtasks":["Break the question into searchable clues and state the expected answer type.","Search the web for the source page and extract the exact organization name with evidence.","Return only FINAL_ANSWER: <exact answer string>. Return the exact organization name only."],"access_list":[[],["all"],["all"]]}

FEW-SHOT EXAMPLE 3, TRUE PARALLEL independent browsing:
Question: Find the actor described by one clue from an episode list and another clue from a biography page.
Workflow:
{"model_id":[3,3,1,2],"subtasks":["Search using only the episode-list clue. Extract candidate actors with source URLs.","Search using only the biography clue. Extract candidate actors with source URLs.","Compare the two independent result sets. Select the candidate that satisfies both clues and reject unsupported candidates.","Return only FINAL_ANSWER: <exact answer string>. Return the actor's name only."],"access_list":[[],[],["all"],["all"]]}

FEW-SHOT EXAMPLE 4, TRUE PARALLEL with three independent clue paths:
Question: Find the historical artifact described by date, location, and collection clues.
Workflow:
{"model_id":[3,3,3,1,2],"subtasks":["Search using only the date clue. Extract candidate artifacts with evidence.","Search using only the location clue. Extract candidate artifacts with evidence.","Search using only the collection or museum clue. Extract candidate artifacts with evidence.","Aggregate the three candidate sets and verify which artifact satisfies all clues.","Return only FINAL_ANSWER: <exact answer string>. Return the exact artifact name only."],"access_list":[[],[],[],["all"],["all"]]}

FEW-SHOT EXAMPLE 5, MIXED topology: initial search plus independent re-search:
Question: Find the book title from clues about a quote, a translator, and a publisher.
Workflow:
{"model_id":[0,3,3,1,2],"subtasks":["Break the task into clue groups and propose likely source types.","Search using the quote and translator clues. Extract candidate book titles with evidence.","Independently search using the publisher and date clues. Extract candidate book titles with evidence.","Compare both search paths and verify which title satisfies every clue.","Return only FINAL_ANSWER: <exact answer string>. Return the exact book title only."],"access_list":[[],["all"],[],["all"],["all"]]}

FEW-SHOT EXAMPLE 6, REFINED SEARCH after weak evidence:
Question: Find the full legal name of a person referred to only by initials in secondary sources.
Workflow:
{"model_id":[0,3,0,3,2],"subtasks":["Identify initials, aliases, date constraints, and likely official sources.","Search broad sources for candidates and note which constraints remain unresolved.","Generate refined queries using aliases, dates, official records, and archival sources.","Run the refined search and extract the exact full legal name with evidence.","Return only FINAL_ANSWER: <exact answer string>. Return the full name only."],"access_list":[[],["all"],["all"],["all"],["all"]]}

FEW-SHOT EXAMPLE 7, CONFLICTING SOURCES:
Question: Several sources disagree about the year an event occurred. Find the correct year using the original source.
Workflow:
{"model_id":[3,3,1,2],"subtasks":["Search broad web sources and collect candidate years with source type labels.","Search primary, official, or archival sources for the same event.","Compare the candidates. Prefer primary sources over mirrors, summaries, and unsourced pages.","Return only FINAL_ANSWER: <exact answer string>. Return only the year."],"access_list":[[],[],["all"],["all"]]}

FEW-SHOT EXAMPLE 8, EXACT STRING extraction:
Question: A repository release note contains a hidden token next to a changelog entry. Find the exact token.
Workflow:
{"model_id":[0,3,1,2],"subtasks":["Identify the repository, release note, changelog clue, and expected token format. Emphasize verbatim extraction.","Search the relevant source. Extract exact candidate tokens verbatim, preserving case, punctuation, symbols, and spacing.","Reject descriptions of the token type. Verify which exact token is directly supported by evidence.","Return only FINAL_ANSWER: <exact answer string>. Return the exact token only."],"access_list":[[],["all"],["all"],["all"]]}

FEW-SHOT EXAMPLE 9, MULTI-HOP entity resolution:
Question: Find the company founded by the person who wrote a particular obscure technical blog post.
Workflow:
{"model_id":[3,3,1,2],"subtasks":["Search for the blog post and extract the author with source evidence.","Independently search the author and founder/company records. Extract candidate companies with evidence.","Verify that the blog author and company founder are the same person, then select the matching company.","Return only FINAL_ANSWER: <exact answer string>. Return the company name only."],"access_list":[[],[],["all"],["all"]]}

FEW-SHOT EXAMPLE 10, ARCHIVED or removed content:
Question: A current page no longer contains the answer, but an archived version did. Find the exact phrase.
Workflow:
{"model_id":[0,3,0,3,2],"subtasks":["Identify the target page, time clues, and why an archived or mirrored source may be needed.","Search the live web, archives, and mirrors for the target page. Extract candidate phrases.","Analyze which candidate belongs to the correct historical version and generate refined archive queries.","Run refined archive or mirror searches and collect stronger evidence for the exact phrase.","Return only FINAL_ANSWER: <exact answer string>. Return the exact phrase only."],"access_list":[[],["all"],["all"],["all"],["all"]]}

FEW-SHOT EXAMPLE 11, TABLE or PDF extraction:
Question: A PDF table lists several codes. Find the code associated with a specific organization and year.
Workflow:
{"model_id":[0,3,1,2],"subtasks":["Identify the organization, year, likely document type, table fields, and expected code format.","Search for the PDF or table source. Extract candidate rows and exact codes with page or table evidence.","Verify that the row matches the organization and year. Reject nearby rows or similarly named organizations.","Return only FINAL_ANSWER: <exact answer string>. Return the exact code only."],"access_list":[[],["all"],["all"],["all"]]}

FEW-SHOT EXAMPLE 12, MULTILINGUAL clue:
Question: Find the original title of a work described in English but cataloged under another language.
Workflow:
{"model_id":[0,3,0,3,2],"subtasks":["Identify translated terms, likely original-language keywords, names, dates, and expected answer type.","Search using English clues and extract candidate translated titles or names.","Generate original-language query variants from the strongest candidates and remaining constraints.","Search original-language or catalog sources. Extract the exact original title with evidence.","Return only FINAL_ANSWER: <exact answer string>. Return the original title only."],"access_list":[[],["all"],["all"],["all"],["all"]]}

FEW-SHOT EXAMPLE 13, NO DIRECT HIT then alternative path:
Question: Find a person from a rare combination of school, award, and early-career role clues.
Workflow:
{"model_id":[0,3,0,3,2],"subtasks":["Create a broad search plan and list rare clue combinations. State likely answer type as a person name.","Search broad clue combinations and collect partial candidates with evidence.","If evidence is weak or incomplete, generate refined queries using aliases, abbreviations, date ranges, and site-specific searches.","Run refined searches and update the candidate list with stronger evidence linking all clues.","Return only FINAL_ANSWER: <exact answer string>. Return the person's name only."],"access_list":[[],["all"],["all"],["all"],["all"]]}

FEW-SHOT EXAMPLE 14, REJECT tempting candidate:
Question: Find the museum object whose description matches several clues, but a popular blog names a different object incorrectly.
Workflow:
{"model_id":[3,3,1,2],"subtasks":["Search broad web results and collect all candidate objects, including tempting false positives.","Independently search official museum or collection records for the same clues.","Compare popular-source candidates against official records. Reject candidates missing any required clue or relying only on secondary sources.","Return only FINAL_ANSWER: <exact answer string>. Return the official object title/name only."],"access_list":[[],[],["all"],["all"]]}

FEW-SHOT EXAMPLE 15, FINAL answer from multiple candidates:
Question: Two candidates satisfy most clues. Find which one also satisfies the final location clue.
Workflow:
{"model_id":[0,3,3,1,2],"subtasks":["Identify all constraints and split search into candidate discovery and final-location verification.","Search for candidates satisfying the main non-location clues. Extract candidate names and evidence.","Independently search the final location clue combined with each candidate. Extract supporting or contradicting evidence.","Compare candidates against every clue and select only the candidate satisfying the final location clue.","Return only FINAL_ANSWER: <exact answer string>. Return the exact name or title only."],"access_list":[[],["all"],[],["all"],["all"]]}

FEW-SHOT EXAMPLE 16, SOURCE QUALITY filtering:
Question: Find the exact number reported in a specific annual report table.
Workflow:
{"model_id":[0,3,1,2],"subtasks":["Identify the report, year, metric name, table title, and expected numeric format.","Search for the official report or table. Extract candidate numeric values with page/table evidence.","Verify the metric, year, units, and row/column. Reject values from wrong years, summaries, or different units.","Return only FINAL_ANSWER: <exact answer string>. Return only the exact number requested."],"access_list":[[],["all"],["all"],["all"]]}
""".strip()


def build_conductor_instructions() -> str:
    fs = few_shot_text()
    return f"""
You are the Conductor in a multi-agent web-browsing system for BrowseComp.

Your job is NOT to answer the question directly.
Your job is to design a workflow of worker calls that can locate hard-to-find information on the web.

BrowseComp questions require persistent web navigation, creative search, and evidence-backed short answers.

════════════════════════════════════════════
WORKER POOL
════════════════════════════════════════════
Model 0: language model worker specialized in search planning and query decomposition.
Model 1: language model worker specialized in evidence verification and candidate comparison.
Model 2: language model worker specialized in final answer formatting.
Model 3: browser/search tool worker. This worker can search the public web and extract evidence.

════════════════════════════════════════════
TOPOLOGY CATALOGUE  (paper-aligned)
════════════════════════════════════════════
access_list[i] = []      → worker i sees ONLY the original question and its subtask.
access_list[i] = ["all"] → worker i sees ALL previous subtasks and responses.

TOPOLOGY 1 · SINGLE-SHOT
  When: directly searchable question.
  Flow: Browser → Output

TOPOLOGY 2 · SEQUENTIAL CHAIN
  When: the question has one clear search path and only needs one evidence-gathering pass.
  Do not use chain by default.

TOPOLOGY 3 · PARALLEL / TREE
  When: question has separable clues, aliases, or multiple search paths.
  Flow: Planner → Browser A + Browser B → Verifier/Aggregator → Final

TOPOLOGY 4 · MIXED
  When: question needs search refinement after initial evidence.
  Flow: Planner → Browser → Query Refiner → Browser → Final

TOPOLOGY 5 · ABDICATION / META-ORCHESTRATION
  When: extremely complex question.
  First worker may propose a more detailed browsing strategy for later workers.

════════════════════════════════════════════
ROLE TAXONOMY
════════════════════════════════════════════
search_planner    — decomposes clues and writes focused queries
browser           — searches web, opens pages, extracts candidate answers and evidence
evidence_verifier — checks every candidate against every clue
aggregator        — compares independent search results
refiner           — creates refined queries or resolves missing evidence
formatter         — returns exactly one short final answer
meta_orchestrator — delegates a detailed strategy to other workers

════════════════════════════════════════════
OUTPUT RULES
════════════════════════════════════════════
1. Use 1 to {MAX_WORKFLOW_STEPS} workflow steps.
2. All three lists must have the same length.
3. Each model_id must be 0, 1, 2, or 3. These are the only four available worker slots.
4. Use model_id 3 when actual web search is needed. Model 3 is the added gpt-5-mini browser/search tool worker.
5. Each access_list entry must be [] or ["all"].
6. Do NOT always use the same topology. Allocate more steps only when needed.
7. For hard questions, prefer evidence-backed search and verification over guessing.
8. Dataset leakage is forbidden. Do not use benchmark dataset mirrors, Hugging Face dataset viewers, Kaggle dataset pages, GitHub dataset dumps, or any page that exposes fields like answer/gold/reference_answer for the same BrowseComp item.
9. The final subtask must say exactly:
   "Return only FINAL_ANSWER: <exact answer string>. Return the exact entity, title, name, number, code, token, identifier, or string requested. Do not describe the answer type."
9. Output compact minified JSON only. No markdown. No commentary.
10. Do not always use [0,3,1,2]. Use it only when a single sequential search path is clearly sufficient.
11. For questions with separable clues, aliases, titles, dates, locations, or independent search paths, prefer PARALLEL/TREE.
12. For weak or missing evidence, prefer MIXED or refined-search workflows.
13. Few-shot examples are topology demonstrations only. Do not infer answers from benchmark datasets, gold-answer pages, dataset mirrors, or benchmark viewers.
14. Do not copy long blobs or the full user question into subtasks. Refer to them as "the provided string" or "the provided blob".

{fs}
""".strip()


def build_conductor_input(question: str) -> str:
    return f"""USER QUESTION:
{question}

AVAILABLE WORKER MODELS:
Model 0: language model search planner.
Model 1: language model evidence verifier.
Model 2: final answer formatter.
Model 3: browser/search tool worker powered by gpt-5-mini.

Return the JSON workflow now."""


def build_recursive_conductor_instructions(remaining_steps: int) -> str:
    return f"""
You are the Conductor in a BrowseComp web-browsing system.

You have received the final response from your previous workflow. Decide whether to pass it through or launch a new evidence-search/refinement round.

OPTION A — PASS THROUGH:
Return {{"model_id":[],"subtasks":[],"access_list":[]}} if the previous answer is well-supported and satisfies every clue.

OPTION B — NEW WORKFLOW:
Design up to {remaining_steps} steps to search missing evidence, resolve conflicting candidates, verify unsupported clues, or correct the answer.

Rules:
- model_id must be a list containing only 0, 1, 2, or 3. Example: "model_id":[3], not "model_id":3.
- access_list must be a list of lists with the same length as model_id. Example: "access_list":[["all"]], not "access_list":["all"].
- Use model_id 3 for actual web search. Model 3 is the added gpt-5-mini browser/search tool worker.
- Dataset leakage is forbidden. Do not use benchmark mirrors, Hugging Face dataset viewers, Kaggle dataset pages, GitHub dataset dumps, or sources exposing answer/gold/reference_answer fields.
- access_list entries must be [] or ["all"].
- Final subtask must be: "Return only FINAL_ANSWER: <exact answer string>. Return the exact entity, title, name, number, code, token, identifier, or string requested. Do not describe the answer type."
- Output compact minified JSON only.
""".strip()


def build_recursive_conductor_input(question: str, previous_final_response: str) -> str:
    return f"""ORIGINAL QUESTION:
{question}

FINAL RESPONSE FROM PREVIOUS WORKFLOW:
{previous_final_response}

AVAILABLE WORKER MODELS:
Model 0: language model search planner.
Model 1: language model evidence verifier.
Model 2: final answer formatter.
Model 3: browser/search tool worker.

Pass through with empty lists or design a new web-search verification workflow."""


# ============================================================
# 5. Plan parsing
# ============================================================


def _json_object_from_text(raw: str) -> str:
    text = raw.strip()
    for pat in (r"^```json\s*", r"^```\s*", r"\s*```$"):
        text = re.sub(pat, "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


def parse_plan(raw: str) -> Dict[str, Any]:
    obj = json.loads(_json_object_from_text(raw))
    model_id = obj.get("model_id")
    subtasks = obj.get("subtasks")
    access_list = obj.get("access_list")

    if not (isinstance(model_id, list) and isinstance(subtasks, list) and isinstance(access_list, list)):
        raise ValueError("Plan must include list fields: model_id, subtasks, access_list")
    if len(model_id) == 0 or len(model_id) > MAX_WORKFLOW_STEPS:
        raise ValueError(f"Invalid workflow length: {len(model_id)}")
    if not (len(model_id) == len(subtasks) == len(access_list)):
        raise ValueError("model_id, subtasks, access_list lengths differ")

    subtasks = [str(s).strip() for s in subtasks]
    subtasks[-1] = (
        "Return only FINAL_ANSWER: <exact answer string>. "
        "Return the exact entity, title, name, number, code, token, identifier, or string requested. "
        "Do not describe the answer type."
    )

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
    obj = json.loads(_json_object_from_text(raw))
    mid = obj.get("model_id", [])
    sub = obj.get("subtasks", [])
    acc = obj.get("access_list", [])
    if isinstance(mid, list) and len(mid) == 0 and isinstance(sub, list) and len(sub) == 0 and isinstance(acc, list) and len(acc) == 0:
        return None
    return parse_plan(raw)


# ============================================================
# 6. Worker execution
# ============================================================


def api_model_for_worker(model_id: int) -> str:
    if model_id == BROWSER_WORKER_ID:
        return BROWSER_MODEL
    return WORKER_MODELS[model_id]


def worker_role_hint(model_id: int) -> str:
    return {
        0: "search planner / clue decomposer",
        1: "evidence verifier / candidate comparer",
        2: "final answer formatter",
        3: "browser/search tool worker",
    }.get(model_id, "worker")


def build_worker_instructions(model_id: int, is_final_step: bool) -> str:
    if model_id == BROWSER_WORKER_ID:
        return (
            "You are a browser/search tool worker for BrowseComp. Use web search when useful.\n"
            "Your job is to find exact candidate answers and evidence, not to guess.\n"
            "When the question asks for a code, token, identifier, filename, string, encoded value, title, name, date, or number, extract exact candidate strings verbatim.\n"
            "Preserve capitalization, punctuation, symbols, spacing, and special characters exactly.\n"
            "Do not summarize exact-string answers as descriptions like 'base64 data', 'unknown file', or 'likely ciphertext'.\n"
            "Return a concise evidence report. Prefer JSON with keys: queries_used, candidate_answers, evidence, uncertainty.\n"
            "For each candidate answer, put the exact answer string in an 'answer' field and explain which source supports it.\n"
            "Each evidence item should include source title or URL if available and a short summary of what it supports."
        )
    if is_final_step:
        return (
            "You are the final worker in a BrowseComp workflow.\n"
            "Use the visible evidence and candidate verification.\n"
            "Return exactly one line: FINAL_ANSWER: <exact answer string>\n"
            "Do not include explanation, citations, markdown, or extra text.\n"
            "Do not describe the answer type. Do not output phrases like 'base64 data', 'binary data', 'unknown', 'likely', or 'unresolved'.\n"
            "If previous evidence contains an exact candidate string, return that exact string only, preserving capitalization, punctuation, symbols, and spacing."
        )
    return (
        f"You are a {worker_role_hint(model_id)} in a BrowseComp workflow.\n"
        "Follow your assigned subtask. Be concise and explicit.\n"
        "If verifying, check each candidate against every clue and state unresolved uncertainty.\n"
        "Reject candidates that merely describe the answer type, such as 'base64-encoded data' or 'likely ciphertext'.\n"
        "Prefer exact strings, codes, names, titles, dates, or values that are directly supported by evidence."
    )


def build_worker_input(
    question: str,
    subtask: str,
    history: List[Dict[str, Any]],
    access: List[str],
    parent_response: Optional[str] = None,
) -> str:
    if access == ["all"]:
        parts: List[str] = []
        if parent_response and parent_response.strip():
            parts.append(f"[Previous round final response]\n{parent_response}")
        for h in history:
            parts.append(
                f"[Current round, step {h['step']}]\n"
                f"Worker: Model {h['model_id']} ({h.get('role_hint','')})\n"
                f"Subtask: {h['subtask']}\n"
                f"Response:\n{h['response']}"
            )
        history_text = "\n\n".join(parts) if parts else "No previous outputs visible."
    else:
        history_text = "No previous worker outputs are visible."

    return (
        f"ORIGINAL BROWSECOMP QUESTION:\n{question}\n\n"
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

    for step, (mid, subtask, access) in enumerate(zip(plan["model_id"], plan["subtasks"], plan["access_list"])):
        model = api_model_for_worker(mid)
        is_final = step == len(plan["model_id"]) - 1
        instr = build_worker_instructions(mid, is_final_step=is_final)
        usr_in = build_worker_input(question, subtask, history, access, parent_response=parent_response)
        use_web = mid == BROWSER_WORKER_ID
        temp = BROWSER_TEMPERATURE if use_web else WORKER_TEMPERATURE
        max_tok = MAX_BROWSER_OUTPUT_TOKENS if use_web else MAX_WORKER_OUTPUT_TOKENS

        response, usage = client.generate(
            model=model,
            instructions=instr,
            user_input=usr_in,
            temperature=temp,
            max_output_tokens=max_tok,
            use_web_search=use_web,
        )
        usage_by_model[model].add(usage)
        history.append({
            "step": step,
            "model_id": mid,
            "api_model": model,
            "role_hint": worker_role_hint(mid),
            "subtask": subtask,
            "access": access,
            "used_web_search": use_web,
            "response": response,
            "usage": usage.__dict__,
        })
        time.sleep(SLEEP_BETWEEN_CALLS)

    return (history[-1]["response"] if history else ""), history, usage_by_model


def execute_with_recursion(
    client: OpenAITextClient,
    question: str,
    initial_plan: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Usage], List[Dict[str, Any]]]:
    total_usage: Dict[str, Usage] = defaultdict(Usage)
    rounds: List[Dict[str, Any]] = []

    final_answer, trajectory, round_usage = execute_workflow(client, question, initial_plan)
    for model, u in round_usage.items():
        total_usage[model].add(u)
    rounds.append({"round": 0, "is_recursive": False, "recursion_triggered": False, "passed_through": False, "plan": initial_plan, "trajectory": trajectory, "raw_final_answer": final_answer, "n_steps": len(initial_plan["model_id"])})

    if not ENABLE_RECURSION:
        return final_answer, trajectory, total_usage, rounds

    current_answer = final_answer
    final_trajectory = trajectory
    for rec_round in range(1, MAX_RECURSIVE_CALLS + 1):
        raw_rec, rec_usage = client.generate(
            model=CONDUCTOR_MODEL,
            instructions=build_recursive_conductor_instructions(MAX_WORKFLOW_STEPS),
            user_input=build_recursive_conductor_input(question, current_answer),
            temperature=CONDUCTOR_TEMPERATURE,
            max_output_tokens=MAX_CONDUCTOR_OUTPUT_TOKENS,
        )
        total_usage[CONDUCTOR_MODEL].add(rec_usage)
        time.sleep(SLEEP_BETWEEN_CALLS)

        try:
            rec_plan = parse_recursive_plan(raw_rec)
        except Exception as e:
            rounds.append({"round": rec_round, "is_recursive": True, "parse_error": repr(e), "raw_plan": raw_rec, "recursion_triggered": False, "passed_through": False})
            break

        if rec_plan is None:
            rounds.append({"round": rec_round, "is_recursive": True, "recursion_triggered": False, "passed_through": True, "raw_plan": raw_rec})
            break

        new_answer, new_traj, new_usage = execute_workflow(client, question, rec_plan, parent_response=current_answer)
        for model, u in new_usage.items():
            total_usage[model].add(u)
        rounds.append({"round": rec_round, "is_recursive": True, "recursion_triggered": True, "passed_through": False, "plan": rec_plan, "trajectory": new_traj, "raw_final_answer": new_answer, "n_steps": len(rec_plan["model_id"])})
        current_answer = new_answer
        final_trajectory = new_traj

    return current_answer, final_trajectory, total_usage, rounds


# ============================================================
# 7. Answer extraction and judging
# ============================================================


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"FINAL_ANSWER\s*:\s*(.*)",
        r"Final answer\s*:\s*(.*)",
        r"Answer\s*:\s*(.*)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip().splitlines()[0].strip()
            return ans.strip(" .;:,\"'")
    # Fall back to last non-empty line.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return (lines[-1] if lines else "").strip(" .;:,\"'")


def normalize_short_answer(x: str) -> str:
    x = str(x or "").strip().lower()
    x = re.sub(r"^final_answer\s*:\s*", "", x, flags=re.I)
    x = x.strip().strip("`*_~ \t\n\r\"'")
    x = re.sub(r"\([^)]*\)$", "", x).strip()  # remove trailing parenthetical only
    x = x.replace("&", "and")
    x = re.sub(r"\b(the|a|an)\b", "", x)
    x = re.sub(r"[^a-z0-9가-힣一-龥ぁ-んァ-ン\s.-]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def exact_match(pred: str, gold: str) -> bool:
    return normalize_short_answer(pred) == normalize_short_answer(gold)


def judge_answer(client: OpenAITextClient, question: str, gold: str, model_answer: str) -> Tuple[Optional[bool], Dict[str, Any], Usage]:
    ext = extract_final_answer(model_answer)
    if exact_match(ext, gold):
        return True, {"method": "exact", "extracted_answer": ext, "reason": "normalized exact match"}, Usage()
    if not USE_LLM_JUDGE:
        return False, {"method": "exact", "extracted_answer": ext, "reason": "normalized exact mismatch"}, Usage()

    instr = (
        "You are a strict short-answer equivalence judge for BrowseComp.\n"
        "Return JSON only: {\"correct\":true/false,\"extracted_answer\":\"...\",\"reason\":\"short\"}.\n"
        "Mark correct only if the predicted answer refers to the same entity, title, date, number, or value as the gold answer.\n"
        "Do not give credit for partial answers or unsupported near misses."
    )
    usr = f"QUESTION:\n{question}\n\nGOLD ANSWER:\n{gold}\n\nMODEL RESPONSE:\n{model_answer}\n\nEXTRACTED GUESS:\n{ext}"
    raw, usage = client.generate(JUDGE_MODEL, instr, usr, JUDGE_TEMPERATURE, MAX_JUDGE_OUTPUT_TOKENS)
    try:
        obj = json.loads(_json_object_from_text(raw))
        return bool(obj.get("correct")), {"method": "llm_judge", **obj}, usage
    except Exception:
        return False, {"method": "llm_judge_parse_failed", "raw": raw, "extracted_answer": ext}, usage


# ============================================================
# 8. Metrics helpers
# ============================================================


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total + 1e-12, 2) for c in counts.values())


_ABDICATION_KEYWORDS = ["propose subtasks", "assign work", "direct the other", "orchestrate", "search strategy for later"]
_BROWSER_KW = ["search", "browse", "web", "url", "source", "evidence", "page"]
_VERIFIER_KW = ["verify", "check", "validate", "confirm", "evidence", "candidate"]
_REFINER_KW = ["refine", "revise", "missing", "conflicting", "resolve"]
_PLANNER_KW = ["plan", "strategy", "decompose", "clue", "query", "identify", "break down"]
_AGGREGATOR_KW = ["compare", "aggregate", "combine", "select", "choose"]
_FORMATTER_KW = ["return only final_answer", "final_answer", "format"]


def classify_role(subtask: str, model_id: Optional[int] = None) -> str:
    s = str(subtask or "").lower()
    if model_id == BROWSER_WORKER_ID:
        return "browser"
    if any(k in s for k in _ABDICATION_KEYWORDS):
        return "meta_orchestrator"
    if any(k in s for k in _FORMATTER_KW):
        return "formatter"
    if any(k in s for k in _AGGREGATOR_KW):
        return "aggregator"
    if any(k in s for k in _VERIFIER_KW):
        return "evidence_verifier"
    if any(k in s for k in _REFINER_KW):
        return "refiner"
    if any(k in s for k in _PLANNER_KW):
        return "search_planner"
    if any(k in s for k in _BROWSER_KW):
        return "browser"
    return "reasoner"


def classify_topology(access_list: List[List[str]], subtasks: Optional[List[str]] = None) -> str:
    if len(access_list) == 1:
        return "single"
    if all(a == [] for a in access_list):
        return "independent"
    if len(access_list) >= 3 and access_list[0] == [] and access_list[1] == [] and access_list[-1] == ["all"]:
        return "parallel_independent"
    if all(a == ["all"] for a in access_list[1:]):
        if subtasks and any(k in subtasks[0].lower() for k in _ABDICATION_KEYWORDS):
            return "abdication"
        return "chain"
    return "mixed"


def count_browser_calls(row: Dict[str, Any]) -> int:
    return sum(1 for mid in (row.get("plan") or {}).get("model_id", []) if mid == BROWSER_WORKER_ID)


def extract_browser_report_stats(trajectory: List[Dict[str, Any]]) -> Dict[str, int]:
    candidate_count = 0
    evidence_count = 0
    for step in trajectory or []:
        if step.get("model_id") != BROWSER_WORKER_ID:
            continue
        txt = step.get("response", "")
        # Best effort: parse JSON if the browser worker obeyed schema.
        try:
            obj = json.loads(_json_object_from_text(txt))
            cands = obj.get("candidate_answers") or obj.get("candidates") or []
            ev = obj.get("evidence") or []
            if isinstance(cands, list):
                candidate_count += len(cands)
            if isinstance(ev, list):
                evidence_count += len(ev)
        except Exception:
            # Heuristic fallback.
            candidate_count += len(re.findall(r"candidate", txt, flags=re.I))
            evidence_count += len(re.findall(r"https?://|source|evidence|url", txt, flags=re.I))
    return {"candidate_count": candidate_count, "evidence_count": evidence_count}




def detect_dataset_leakage_in_text(text: str) -> Dict[str, Any]:
    """Best-effort detector for benchmark-answer leakage in browser/verifier outputs."""
    if not LEAKAGE_GUARD_ENABLED:
        return {"dataset_leakage_detected": False, "leakage_hits": []}
    s = str(text or "").lower()
    hits: List[str] = []
    # Strong pattern: BrowseComp dataset/mirror + answer/gold-like fields.
    for domain in LEAKAGE_REJECT_DOMAINS:
        if domain.lower() in s:
            hits.append(f"domain:{domain}")
    for term in LEAKAGE_REJECT_TERMS:
        if term.lower() in s:
            hits.append(f"term:{term}")
    # More specific combined patterns.
    if "browsecomp" in s and any(k in s for k in ["answer", "gold", "reference", "target", "dataset viewer"]):
        hits.append("combined:browsecomp_answer_source")
    if "multiturnrl/browsecomp" in s:
        hits.append("dataset:MultiturnRL/BrowseComp")
    return {"dataset_leakage_detected": bool(hits), "leakage_hits": sorted(set(hits))}


def detect_dataset_leakage_in_trajectory(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    hits: List[str] = []
    leaking_steps: List[int] = []
    for step in trajectory or []:
        result = detect_dataset_leakage_in_text(step.get("response", ""))
        if result["dataset_leakage_detected"]:
            leaking_steps.append(step.get("step"))
            hits.extend(result.get("leakage_hits", []))
    return {
        "dataset_leakage_detected": bool(hits),
        "leakage_hits": sorted(set(hits)),
        "leaking_steps": leaking_steps,
    }

def compute_cost(usage_by_model: Dict[str, Usage]) -> Dict[str, Any]:
    total, details = 0.0, {}
    for model, u in usage_by_model.items():
        prices = PRICE_PER_1M.get(model)
        if not prices:
            details[model] = {**u.__dict__, "estimated_cost_usd": None}
            continue
        cost = (u.input_tokens / 1e6) * prices["input"] + (u.output_tokens / 1e6) * prices["output"]
        total += cost
        details[model] = {**u.__dict__, "estimated_cost_usd": cost}
    return {"estimated_total_cost_usd": total, "by_model": details}


def last_worker_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    traj = row.get("trajectory") or []
    if traj:
        last = traj[-1]
        return {"model_id": last.get("model_id"), "api_model": last.get("api_model"), "step": last.get("step"), "role": classify_role(last.get("subtask", ""), last.get("model_id")), "subtask": last.get("subtask", "")}
    return None


def rollout_answer_summary(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "question_id": r.get("question_id"),
        "rollout_id": r.get("rollout_id"),
        "parse_ok": r.get("parse_ok"),
        "executed_ok": r.get("executed_ok"),
        "is_correct": r.get("is_correct"),
        "gold_answer": r.get("gold_answer", r.get("gold")),
        "predicted_answer": r.get("predicted_answer", ""),
        "topology": r.get("topology"),
        "n_steps": r.get("n_steps"),
        "browser_call_count": r.get("browser_call_count", 0),
        "candidate_count": r.get("candidate_count", 0),
        "evidence_count": r.get("evidence_count", 0),
        "dataset_leakage_detected": r.get("dataset_leakage_detected", False),
        "leakage_hits": r.get("leakage_hits", []),
        "leaking_steps": r.get("leaking_steps", []),
        "is_recursive": r.get("is_recursive", False),
        "n_recursive_rounds_triggered": r.get("n_recursive_rounds_triggered", 0),
        "judge_method": (r.get("judge") or {}).get("method"),
        "judge_reason": (r.get("judge") or {}).get("reason"),
        "last_worker": last_worker_from_row(r),
        "error": r.get("error"),
    } for r in logs]


def analyze_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in logs if r.get("parse_ok") and r.get("executed_ok")]
    by_q = defaultdict(list)
    for r in valid:
        by_q[r["question_id"]].append(r)

    g_model, g_topo = Counter(), Counter()
    g_lens: List[int] = []
    corr_all: List[bool] = []
    q_summ = []

    for qid, rows in by_q.items():
        m_calls, m_seqs, topos, role_seqs, answers = Counter(), Counter(), Counter(), Counter(), Counter()
        lens, corr = [], []
        for r in rows:
            plan = r["plan"]
            mids = tuple(plan["model_id"])
            subs = plan["subtasks"]
            topo = classify_topology(plan["access_list"], subtasks=subs)
            pred = normalize_short_answer(r.get("predicted_answer", ""))
            for mid in mids:
                m_calls[mid] += 1
                g_model[mid] += 1
            m_seqs[mids] += 1
            topos[topo] += 1
            g_topo[topo] += 1
            role_seqs[tuple(classify_role(s, mid) for s, mid in zip(subs, mids))] += 1
            lens.append(len(mids))
            g_lens.append(len(mids))
            answers[pred] += 1
            if r.get("is_correct") is not None:
                corr.append(bool(r["is_correct"]))
                corr_all.append(bool(r["is_correct"]))
        q_summ.append({
            "question_id": qid,
            "n_rollouts": len(rows),
            "accuracy": statistics.mean(corr) if corr else None,
            "agent_selection_entropy": entropy(m_calls),
            "unique_model_sequences": len(m_seqs),
            "model_sequence_counts": {str(k): v for k, v in m_seqs.items()},
            "topology_entropy": entropy(topos),
            "topology_counts": dict(topos),
            "role_sequence_counts": {str(k): v for k, v in role_seqs.items()},
            "workflow_length_mean": statistics.mean(lens),
            "workflow_length_variance": statistics.pvariance(lens) if len(lens) > 1 else 0.0,
            "answer_entropy": entropy(answers),
            "answer_counts": dict(answers),
            "avg_browser_calls": statistics.mean([r.get("browser_call_count", 0) for r in rows]),
            "avg_evidence_count": statistics.mean([r.get("evidence_count", 0) for r in rows]),
            "avg_candidate_count": statistics.mean([r.get("candidate_count", 0) for r in rows]),
            "dataset_leakage_count": sum(1 for r in rows if r.get("dataset_leakage_detected")),
            "dataset_leakage_rate": sum(1 for r in rows if r.get("dataset_leakage_detected")) / len(rows),
        })

    parse_failed = [r for r in logs if r.get("parse_ok") is False]
    exec_failed = [r for r in logs if r.get("parse_ok") and r.get("executed_ok") is False]
    parse_ok_cnt = len([r for r in logs if r.get("parse_ok")])

    return {
        "n_total_logs": len(logs),
        "n_valid_logs": len(valid),
        "n_parse_failed": len(parse_failed),
        "parse_failure_rate": len(parse_failed) / len(logs) if logs else None,
        "n_execution_failed": len(exec_failed),
        "exec_failure_rate": len(exec_failed) / max(1, parse_ok_cnt) if logs else None,
        "n_questions": len(by_q),
        "global_accuracy": statistics.mean(corr_all) if corr_all else None,
        "global_agent_selection_entropy": entropy(g_model),
        "global_model_call_counts": dict(g_model),
        "global_topology_entropy": entropy(g_topo),
        "global_topology_counts": dict(g_topo),
        "global_workflow_length_mean": statistics.mean(g_lens) if g_lens else None,
        "global_workflow_length_variance": statistics.pvariance(g_lens) if len(g_lens) > 1 else None,
        "global_browser_call_count": sum(r.get("browser_call_count", 0) for r in valid),
        "global_avg_browser_calls_per_valid_rollout": statistics.mean([r.get("browser_call_count", 0) for r in valid]) if valid else None,
        "global_dataset_leakage_count": sum(1 for r in valid if r.get("dataset_leakage_detected")),
        "global_dataset_leakage_rate": (sum(1 for r in valid if r.get("dataset_leakage_detected")) / len(valid)) if valid else None,
        "global_accuracy_excluding_leakage": (statistics.mean([bool(r["is_correct"]) for r in valid if not r.get("dataset_leakage_detected") and r.get("is_correct") is not None]) if any((not r.get("dataset_leakage_detected") and r.get("is_correct") is not None) for r in valid) else None),
        "question_summaries": q_summ,
    }


def compute_exploration_report(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in logs if r.get("parse_ok") and r.get("executed_ok")]
    judged = [r for r in valid if r.get("is_correct") is not None]
    success = [r for r in judged if r["is_correct"]]
    failure = [r for r in judged if not r["is_correct"]]

    g_agents = Counter()
    g_topo = Counter()
    g_roles = Counter()
    lens = []
    for r in valid:
        plan = r["plan"]
        for mid, sub in zip(plan["model_id"], plan["subtasks"]):
            g_agents[mid] += 1
            g_roles[classify_role(sub, mid)] += 1
        g_topo[classify_topology(plan["access_list"], subtasks=plan["subtasks"])] += 1
        lens.append(len(plan["model_id"]))

    def _traj_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0}
        lens2 = [len(r["plan"]["model_id"]) for r in rows]
        topos = Counter(classify_topology(r["plan"]["access_list"], subtasks=r["plan"].get("subtasks")) for r in rows)
        roles = Counter()
        agents = Counter()
        for r in rows:
            for mid, s in zip(r["plan"]["model_id"], r["plan"]["subtasks"]):
                roles[classify_role(s, mid)] += 1
                agents[mid] += 1
        return {
            "n": len(rows),
            "workflow_length": {"mean": round(statistics.mean(lens2), 4), "variance": round(statistics.pvariance(lens2), 4) if len(lens2) > 1 else 0.0, "distribution": dict(Counter(lens2))},
            "topology_distribution": dict(topos),
            "topology_entropy": round(entropy(topos), 4),
            "role_distribution": dict(roles),
            "agent_selection": dict(agents),
            "agent_entropy": round(entropy(agents), 4),
            "avg_browser_calls": round(statistics.mean([r.get("browser_call_count", 0) for r in rows]), 4),
            "avg_evidence_count": round(statistics.mean([r.get("evidence_count", 0) for r in rows]), 4),
            "avg_candidate_count": round(statistics.mean([r.get("candidate_count", 0) for r in rows]), 4),
            "recursion_trigger_rate": sum(1 for r in rows if r.get("n_recursive_rounds_triggered", 0) > 0) / len(rows),
            "dataset_leakage_count": sum(1 for r in rows if r.get("dataset_leakage_detected")),
            "dataset_leakage_rate": round(sum(1 for r in rows if r.get("dataset_leakage_detected")) / len(rows), 4),
        }

    rec_triggered = [r for r in valid if r.get("n_recursive_rounds_triggered", 0) > 0]
    no_rec = [r for r in valid if r.get("n_recursive_rounds_triggered", 0) == 0]

    def _acc(rows: List[Dict[str, Any]]) -> Optional[float]:
        js = [r for r in rows if r.get("is_correct") is not None]
        return round(statistics.mean([bool(r["is_correct"]) for r in js]), 4) if js else None

    return {
        "agent_selection_entropy": {
            "global_entropy": round(entropy(g_agents), 4),
            "global_counts": dict(g_agents),
            "max_possible_bits": round(math.log2(BROWSER_WORKER_ID + 1), 4),
        },
        "topology_diversity": {
            "global_entropy": round(entropy(g_topo), 4),
            "global_counts": dict(g_topo),
        },
        "workflow_length_variance": {
            "global_mean": round(statistics.mean(lens), 4) if lens else None,
            "global_variance": round(statistics.pvariance(lens), 4) if len(lens) > 1 else 0.0,
            "global_min": min(lens) if lens else None,
            "global_max": max(lens) if lens else None,
            "length_distribution": dict(Counter(lens)),
        },
        "role_distribution": {"global_counts": dict(g_roles), "global_entropy": round(entropy(g_roles), 4)},
        "tool_usage": {
            "total_browser_calls": sum(r.get("browser_call_count", 0) for r in valid),
            "avg_browser_calls_per_valid_rollout": round(statistics.mean([r.get("browser_call_count", 0) for r in valid]), 4) if valid else None,
            "avg_evidence_count_per_valid_rollout": round(statistics.mean([r.get("evidence_count", 0) for r in valid]), 4) if valid else None,
            "avg_candidate_count_per_valid_rollout": round(statistics.mean([r.get("candidate_count", 0) for r in valid]), 4) if valid else None,
        },
        "dataset_leakage": {
            "enabled": LEAKAGE_GUARD_ENABLED,
            "n_leakage_detected": sum(1 for r in valid if r.get("dataset_leakage_detected")),
            "leakage_rate": round(sum(1 for r in valid if r.get("dataset_leakage_detected")) / len(valid), 4) if valid else 0,
            "accuracy_on_leakage_cases": _acc([r for r in valid if r.get("dataset_leakage_detected")]),
            "accuracy_excluding_leakage": _acc([r for r in valid if not r.get("dataset_leakage_detected")]),
            "leakage_hit_counts": dict(Counter(hit for r in valid for hit in r.get("leakage_hits", []))),
        },
        "success_failure_contrast": {"success": _traj_stats(success), "failure": _traj_stats(failure)},
        "recursion_stats": {
            "enabled": ENABLE_RECURSION,
            "max_recursive_calls": MAX_RECURSIVE_CALLS,
            "n_total_valid_rollouts": len(valid),
            "n_recursion_triggered": len(rec_triggered),
            "recursion_trigger_rate": round(len(rec_triggered) / len(valid), 4) if valid else 0,
            "accuracy_with_recursion": _acc(rec_triggered),
            "accuracy_without_recursion": _acc(no_rec),
            "round_distribution": dict(Counter(r.get("n_recursive_rounds_triggered", 0) for r in valid)),
        },
    }


# ============================================================
# 9. Main experiment
# ============================================================


def error_details(phase: str, error: Exception) -> Dict[str, str]:
    return {"phase": phase, "error_type": type(error).__name__, "message": str(error), "repr": repr(error)}


def main() -> None:
    api_key = OPENAI_API_KEY
    if api_key == "여기에_본인_API_KEY":
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY를 스크립트 또는 환경 변수로 설정하세요.")

    random.seed(RANDOM_SEED)
    client = OpenAITextClient(api_key=api_key)
    tasks = load_browsecomp_subset(N_QUESTIONS, RANDOM_SEED)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": "BrowseComp",
        "browsecomp_path": BROWSECOMP_PATH or os.getenv("BROWSECOMP_PATH", ""),
        "conductor_model": CONDUCTOR_MODEL,
        "worker_models": WORKER_MODELS,
        "browser_worker_id": BROWSER_WORKER_ID,
        "browser_model": BROWSER_MODEL,
        "web_search_tool_type": WEB_SEARCH_TOOL_TYPE,
        "leakage_guard_enabled": LEAKAGE_GUARD_ENABLED,
        "leakage_reject_domains": LEAKAGE_REJECT_DOMAINS,
        "leakage_reject_terms": LEAKAGE_REJECT_TERMS,
        "judge_model": JUDGE_MODEL,
        "n_questions": N_QUESTIONS,
        "n_rollouts": N_ROLLOUTS,
        "random_seed": RANDOM_SEED,
        "max_workflow_steps": MAX_WORKFLOW_STEPS,
        "enable_recursion": ENABLE_RECURSION,
        "max_recursive_calls": MAX_RECURSIVE_CALLS,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "tasks.json").write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    logs: List[Dict[str, Any]] = []
    total_usage: Dict[str, Usage] = defaultdict(Usage)

    print(f"Loaded {len(tasks)} BrowseComp tasks.")
    print(f"Rollouts: {N_ROLLOUTS} × {len(tasks)} = {N_ROLLOUTS * len(tasks)} trajectories.")
    print(f"Browser/search worker: Model {BROWSER_WORKER_ID} using Responses tool={WEB_SEARCH_TOOL_TYPE}")
    print(f"Recursion: {'ON' if ENABLE_RECURSION else 'OFF'}")
    print(f"Output: {run_dir}\n")

    def _save_logs() -> None:
        with (run_dir / "logs.jsonl").open("w", encoding="utf-8") as f:
            for item in logs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for task_i, task in enumerate(tasks, start=1):
        print(f"[{task_i}/{len(tasks)}] {task['id']}")
        for rollout_id in range(N_ROLLOUTS):
            row: Dict[str, Any] = {
                "question_id": task["id"],
                "dataset_index": task.get("dataset_index"),
                "question": task["question"],
                "gold": task["gold"],
                "gold_answer": task["gold"],
                "rollout_id": rollout_id,
            }

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
                row.update({"parse_ok": False, "executed_ok": False, "parse_error": det, "error": f"{det['phase']}: {det['repr']}"})
                logs.append(row); _save_logs(); continue

            try:
                plan = parse_plan(raw_plan)
                row.update({"raw_plan": raw_plan, "plan": plan, "parse_ok": True, "fallback_used": False})
            except Exception as e:
                det = error_details("plan_parse", e)
                row.update({"raw_plan": raw_plan, "parse_ok": False, "executed_ok": False, "parse_error": det, "error": f"{det['phase']}: {det['repr']}"})
                logs.append(row); _save_logs()
                print(f"  [{rollout_id}] parse FAILED: {det['message']}")
                continue

            try:
                final_answer, final_traj, worker_usage, rounds = execute_with_recursion(client, task["question"], plan)
                for model, u in worker_usage.items():
                    total_usage[model].add(u)

                n_rec_triggered = sum(1 for rnd in rounds if rnd.get("is_recursive") and rnd.get("recursion_triggered"))
                predicted = extract_final_answer(final_answer)
                browser_stats = extract_browser_report_stats(final_traj)
                leakage_stats = detect_dataset_leakage_in_trajectory(final_traj)

                row.update({
                    "executed_ok": True,
                    "trajectory": final_traj,
                    "rounds": rounds,
                    "final_answer": final_answer,
                    "predicted_answer": predicted,
                    "topology": classify_topology(plan["access_list"], subtasks=plan["subtasks"]),
                    "n_steps": len(plan["model_id"]),
                    "browser_call_count": count_browser_calls(row),
                    "candidate_count": browser_stats["candidate_count"],
                    "evidence_count": browser_stats["evidence_count"],
                    "dataset_leakage_detected": leakage_stats["dataset_leakage_detected"],
                    "leakage_hits": leakage_stats["leakage_hits"],
                    "leaking_steps": leakage_stats["leaking_steps"],
                    "is_recursive": n_rec_triggered > 0,
                    "n_recursive_rounds_triggered": n_rec_triggered,
                    "last_worker": last_worker_from_row({"trajectory": final_traj}),
                })
            except Exception as e:
                det = error_details("worker_execution", e)
                row.update({"executed_ok": False, "execution_error": det, "error": f"{det['phase']}: {det['repr']}"})
                logs.append(row); _save_logs()
                print(f"  [{rollout_id}] exec FAILED: {det['message']}")
                continue

            try:
                correct, judge_info, ju = judge_answer(client, task["question"], task["gold"], final_answer)
                total_usage[JUDGE_MODEL].add(ju)
                row.update({"is_correct": correct, "judge": judge_info})
                print(
                    f"  [{rollout_id}] steps={row['n_steps']} topo={row['topology']} "
                    f"browser={row['browser_call_count']} rec={'Y('+str(n_rec_triggered)+')' if n_rec_triggered else 'N'} "
                    f"correct={correct} gold={task['gold']!r} pred={row.get('predicted_answer','')!r}"
                )
            except Exception as e:
                det = error_details("judge", e)
                row.update({"is_correct": None, "judge_error": det})

            logs.append(row)
            _save_logs()
        print()

    metrics = analyze_logs(logs)
    exp_report = compute_exploration_report(logs)
    cost = compute_cost(total_usage)
    judged = [r for r in logs if r.get("parse_ok") and r.get("executed_ok") and r.get("is_correct") is not None]
    success = [r for r in judged if r["is_correct"]]
    failure = [r for r in judged if not r["is_correct"]]
    contrast = exp_report["success_failure_contrast"]

    _save_logs()
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "exploration_report.json").write_text(json.dumps(exp_report, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "success_failure_contrast.json").write_text(json.dumps(contrast, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "rollout_answer_summary.json").write_text(json.dumps(rollout_answer_summary(logs), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "usage_cost_estimate.json").write_text(json.dumps(cost, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("BROWSECOMP EXPERIMENT DONE")
    print("=" * 60)
    print(f"  logs.jsonl              → {run_dir/'logs.jsonl'}")
    print(f"  exploration_report.json → {run_dir/'exploration_report.json'}")
    print(f"  metrics.json            → {run_dir/'metrics.json'}")
    print(f"  success_failure_contrast.json")
    print(f"  rollout_answer_summary.json")
    print(f"  usage_cost_estimate.json")

    print("\n── Key Exploration Metrics ──────────────────────────────")
    print(json.dumps({
        "global_accuracy_valid": metrics.get("global_accuracy"),
        "end_to_end_success_rate": len(success) / len(logs) if logs else None,
        "parse_failure_rate": metrics.get("parse_failure_rate"),
        "agent_selection_entropy": exp_report["agent_selection_entropy"]["global_entropy"],
        "agent_max_possible_bits": exp_report["agent_selection_entropy"]["max_possible_bits"],
        "topology_entropy": exp_report["topology_diversity"]["global_entropy"],
        "topology_counts": exp_report["topology_diversity"]["global_counts"],
        "workflow_length_mean": exp_report["workflow_length_variance"]["global_mean"],
        "workflow_length_variance": exp_report["workflow_length_variance"]["global_variance"],
        "total_browser_calls": exp_report["tool_usage"]["total_browser_calls"],
        "avg_browser_calls_per_valid_rollout": exp_report["tool_usage"]["avg_browser_calls_per_valid_rollout"],
        "dataset_leakage_rate": exp_report["dataset_leakage"]["leakage_rate"],
        "accuracy_excluding_leakage": exp_report["dataset_leakage"]["accuracy_excluding_leakage"],
        "recursion_trigger_rate": exp_report["recursion_stats"]["recursion_trigger_rate"],
        "estimated_total_cost_usd": cost["estimated_total_cost_usd"],
    }, indent=2, ensure_ascii=False))

    print("\n── Success / Failure Contrast ───────────────────────────")
    print(json.dumps(contrast, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
