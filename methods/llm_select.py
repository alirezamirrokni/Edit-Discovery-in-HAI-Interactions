from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from models import build_main_llm


MATH500_SELECTOR_PROMPT = """You are selecting examples for human review.

You will receive a batch of model-generated math answers. Select the items most likely to contain an error and therefore most worth sending to a human corrector.

Rules:
- Respect the total budget {budget}. Each item has a listed cost.
- Select any subset of items whose total cost is at most the budget.
- Use only the problem and the model answer.
- Do not use or assume the gold answer.
- Prefer examples with suspicious reasoning, arithmetic mistakes, missing final answers, format violations, or unsupported conclusions.
- You may select zero items if none appear worth human correction.
- Return only valid JSON in exactly this format:
  {{"selected_indices": [0, 3, 4]}}
- Do not include explanations.
- Do not include markdown.
- Do not include any text outside the JSON object.

Batch:
{items}
"""


POPQA_SELECTOR_PROMPT = """You are selecting examples for human review.

You will receive a batch of model-generated PopQA answers. Select the items most likely to be factually wrong, missing, malformed, or otherwise worth sending to a human corrector.

Rules:
- Respect the total budget {budget}. Each item has a listed cost.
- Select any subset of items whose total cost is at most the budget.
- Use only the question and the model answer.
- Do not use or assume the gold answer.
- The desired answer is a short factual answer: an entity, title, place, date, number, or short phrase.
- Prefer examples with likely factual errors, missing answers, multiple conflicting answers, unsupported hedging, non-short-answer formatting, or responses that look too vague to exactly match a valid PopQA answer alias.
- You may select zero items if none appear worth human correction.
- Return only valid JSON in exactly this format:
  {{"selected_indices": [0, 3, 4]}}
- Do not include explanations.
- Do not include markdown.
- Do not include any text outside the JSON object.

Batch:
{items}
"""


MMLUPRO_SELECTOR_PROMPT = """You are selecting examples for human review.

You will receive a batch of model-generated MMLU-Pro answers. Each item is a multiple-choice academic or professional reasoning question with several options. Select the items whose model answers are most likely to be wrong, invalid, ambiguous, or otherwise worth sending to a human corrector.

Rules:
- Respect the total budget {budget}. Each item has a listed cost.
- Select any subset of items whose total cost is at most the budget.
- Use only the question, answer options, and model answer.
- Do not use or assume the gold answer.
- The desired answer is a single option letter such as A, B, C, ..., or J.
- Prefer examples with questionable reasoning, answers that name no valid option, multiple conflicting options, unsupported certainty, or a final answer that appears inconsistent with the question and options.
- You may select zero items if none appear worth human correction.
- Return only valid JSON in exactly this format:
  {{"selected_indices": [0, 3, 4]}}
- Do not include explanations.
- Do not include markdown.
- Do not include any text outside the JSON object.

Batch:
{items}
"""

GPQA_SELECTOR_PROMPT = """You are selecting examples for human review.

You will receive a batch of model-generated GPQA answers. Each item is a graduate-level science multiple-choice question with four options. Select the items whose model answers are most likely to be wrong, invalid, ambiguous, or otherwise worth sending to a human corrector.

Rules:
- Respect the total budget {budget}. Each item has a listed cost.
- Select any subset of items whose total cost is at most the budget.
- Use only the question, answer options, and model answer.
- Do not use or assume the gold answer.
- The desired answer is a single option letter: A, B, C, or D.
- Prefer examples with questionable scientific reasoning, answers that name no valid option, multiple conflicting options, unsupported certainty, or a final answer that appears inconsistent with the question and options.
- You may select zero items if none appear worth human correction.
- Return only valid JSON in exactly this format:
  {{"selected_indices": [0, 3, 4]}}
- Do not include explanations.
- Do not include markdown.
- Do not include any text outside the JSON object.

Batch:
{items}
"""


MATH500_SELECTOR_SYSTEM_PROMPT = """You are an expert review-selection system.
Your only task is to choose which model-generated answers should be sent for human correction.
Return only valid JSON. Do not solve the problems. Do not include explanations."""


POPQA_SELECTOR_SYSTEM_PROMPT = """You are an expert review-selection system for PopQA-style factual question answering.
Your only task is to choose which model-generated short answers should be sent for human correction.
Return only valid JSON. Do not answer the questions. Do not include explanations."""


MMLUPRO_SELECTOR_SYSTEM_PROMPT = """You are an expert review-selection system for MMLU-Pro multiple-choice reasoning.
Your only task is to choose which model-generated option-letter answers should be sent for human correction.
Return only valid JSON. Do not solve the questions. Do not include explanations."""

GPQA_SELECTOR_SYSTEM_PROMPT = """You are an expert review-selection system for GPQA graduate-level science multiple-choice reasoning.
Your only task is to choose which model-generated option-letter answers should be sent for human correction.
Return only valid JSON. Do not solve the questions. Do not include explanations."""


def _canonical_dataset_name(name: Any) -> str:
    text = str(name or "math500").lower().replace("-", "_")
    aliases = {
        "popqa500": "popqa",
        "mmlu_pro": "mmlupro",
        "mmlu_pro500": "mmlupro",
        "mmlupro500": "mmlupro",
        "gpqa_diamond": "gpqa",
        "gpqa_main": "gpqa",
        "gpqa_extended": "gpqa",
    }
    return aliases.get(text, text)


DEFAULT_SELECTOR_SYSTEM_PROMPTS = {
    "math500": MATH500_SELECTOR_SYSTEM_PROMPT,
    "popqa": POPQA_SELECTOR_SYSTEM_PROMPT,
    "mmlupro": MMLUPRO_SELECTOR_SYSTEM_PROMPT,
    "gpqa": GPQA_SELECTOR_SYSTEM_PROMPT,
}

DEFAULT_SELECTOR_PROMPTS = {
    "math500": MATH500_SELECTOR_PROMPT,
    "popqa": POPQA_SELECTOR_PROMPT,
    "mmlupro": MMLUPRO_SELECTOR_PROMPT,
    "gpqa": GPQA_SELECTOR_PROMPT,
}


class LLMSelect:


    needs_score_model = False

    def __init__(self, cfg: Dict[str, Any], score_model=None, state: Dict[str, Any] | None = None):
        self.cfg = cfg
        self.policy = cfg["policy"]
        self.seed = int(cfg.get("seed", 0))
        self.dataset_name = _canonical_dataset_name(cfg.get("data", {}).get("name", "math500"))
        selector_cfg = dict(cfg.get("selector_llm") or cfg.get("main_llm"))
        if not selector_cfg.get("system_prompt"):
            selector_cfg["system_prompt"] = DEFAULT_SELECTOR_SYSTEM_PROMPTS.get(
                self.dataset_name,
                MATH500_SELECTOR_SYSTEM_PROMPT,
            )
        self.selector_llm = build_main_llm(selector_cfg)
        self.prompt_template = self.policy.get("prompt") or DEFAULT_SELECTOR_PROMPTS.get(
            self.dataset_name,
            MATH500_SELECTOR_PROMPT,
        )

    @staticmethod
    def _truncate(text: str, max_chars: int = 1200) -> str:
        text = str(text)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20] + " ... [truncated]"

    def _format_items(self, batch_df: pd.DataFrame) -> str:
        chunks = []
        if self.dataset_name == "math500":
            question_label = "Problem"
        elif self.dataset_name in {"mmlupro", "gpqa"}:
            question_label = "Question and options"
        else:
            question_label = "Question"
        for i, row in batch_df.reset_index(drop=True).iterrows():
            chunks.append(
                f"Index: {i}\n"
                f"Cost: {float(row['cost']):g}\n"
                f"{question_label}:\n{self._truncate(row['question'], 900)}\n"
                f"Model answer:\n{self._truncate(row['model_answer'], 900)}\n"
            )
        return "\n---\n".join(chunks)

    def _build_prompt(self, batch_df: pd.DataFrame, budget: float) -> str:
        prompt = self.prompt_template.format(
            budget=f"{budget:g}",
            items=self._format_items(batch_df),
        )


        max_prompt_chars = int(self.policy.get("max_prompt_chars", 8000))
        if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars].rstrip() + "\n\n[Prompt truncated due to input-size limit.]"
        return prompt

    @staticmethod
    def _parse_indices(text: str) -> List[int]:
        text = str(text).strip()


        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("selected_indices"), list):
                return [int(x) for x in obj["selected_indices"]]
        except Exception:
            pass


        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and isinstance(obj.get("selected_indices"), list):
                    return [int(x) for x in obj["selected_indices"]]
            except Exception:
                pass


        nums = re.findall(r"-?\d+", text)
        return [int(x) for x in nums]

    @staticmethod
    def _validate_indices(indices: List[int], costs: np.ndarray, budget: float) -> np.ndarray:
        n = len(costs)
        selected = np.zeros(n, dtype=int)
        spent = 0.0
        seen = set()

        for idx in indices:
            if idx in seen or idx < 0 or idx >= n:
                continue
            c = float(costs[idx])
            if c <= 0:
                continue
            if spent + c <= budget + 1e-12:
                selected[idx] = 1
                spent += c
                seen.add(idx)

        return selected

    def process_batch(self, batch_df: pd.DataFrame, t: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        budget = float(self.policy["budget_per_batch"])
        costs = batch_df["cost"].to_numpy(dtype=float)
        prompt = self._build_prompt(batch_df, budget)

        try:
            response = self.selector_llm.generate(prompt)
            indices = self._parse_indices(response)
            selected = self._validate_indices(indices, costs, budget)
        except Exception as exc:
            print(f"[llm_select] selector call/parse failed at batch {t}: {exc}")
            selected = np.zeros(len(batch_df), dtype=int)

        out = batch_df.copy()
        out["eta"] = np.nan
        out["alpha"] = np.nan
        out["beta"] = np.nan
        out["selected"] = selected

        return out, {}
