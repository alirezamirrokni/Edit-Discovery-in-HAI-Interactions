from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from models import build_main_llm
from .ours import OursSelection


MATH500_SCORE_SYSTEM_PROMPT = """You are an expert evaluator of model-generated math answers.
Your task is to estimate whether a human evaluator would need to modify the model answer.
Return only valid JSON. Do not include explanations, markdown, or extra text."""


MATH500_SCORE_PROMPT = """Estimate the posterior edit probability for this prompt-response pair.

Definition:
- score = probability in [0, 1] that a human evaluator would modify the model answer.
- High score means the answer likely contains a mathematical error, unsupported conclusion, missing final answer, ambiguity, or format problem.
- Low score means the answer is likely correct and would be confirmed without modification.

Rules:
- Use only the problem and the model answer below.
- Do not assume access to the gold answer.
- Return only valid JSON in exactly this format:
  {{"score": 0.73}}

Problem:
{question}

Model answer:
{model_answer}
"""


POPQA_SCORE_SYSTEM_PROMPT = """You are an expert evaluator of model-generated PopQA answers.
Your task is to estimate whether a human evaluator would need to correct the model's short factual answer.
Return only valid JSON. Do not include explanations, markdown, or extra text."""


POPQA_SCORE_PROMPT = """Estimate the posterior edit probability for this prompt-response pair.

Definition:
- score = probability in [0, 1] that a human evaluator would modify the model answer.
- High score means the answer is likely factually wrong, missing, too vague, over-specific, ambiguous, malformed, or unlikely to exactly match a valid PopQA answer alias.
- Low score means the answer is likely a correct short factual answer and would be confirmed without modification.

Rules:
- Use only the PopQA question and the model answer below.
- Do not assume access to the gold answer.
- The expected response is a concise answer such as an entity, title, place, date, number, or short phrase.
- Treat explanations, multiple conflicting answers, unsupported hedging, and non-answer text as evidence that human correction may be needed.
- Return only valid JSON in exactly this format:
  {{"score": 0.73}}

Question:
{question}

Model answer:
{model_answer}
"""


MMLUPRO_SCORE_SYSTEM_PROMPT = """You are an expert evaluator of model-generated MMLU-Pro answers.
Your task is to estimate whether a human evaluator would need to correct the model's multiple-choice answer.
Return only valid JSON. Do not include explanations, markdown, or extra text."""


MMLUPRO_SCORE_PROMPT = """Estimate the posterior edit probability for this prompt-response pair.

Definition:
- score = probability in [0, 1] that a human evaluator would modify the model answer.
- High score means the answer is likely the wrong option, not a valid option letter, ambiguous, unsupported, or inconsistent with the question and choices.
- Low score means the answer is likely the correct single option and would be confirmed without modification.

Rules:
- Use only the MMLU-Pro question, answer options, and model answer below.
- Do not assume access to the gold answer.
- The expected response is one option letter such as A, B, C, ..., or J.
- Treat missing final answers, multiple conflicting choices, answer text without a clear option letter, and non-answer text as evidence that human correction may be needed.
- Return only valid JSON in exactly this format:
  {{"score": 0.73}}

Question and options:
{question}

Model answer:
{model_answer}
"""

GPQA_SCORE_SYSTEM_PROMPT = """You are an expert evaluator of model-generated GPQA answers.
Your task is to estimate whether a human evaluator would need to correct the model's graduate-level science multiple-choice answer.
Return only valid JSON. Do not include explanations, markdown, or extra text."""


GPQA_SCORE_PROMPT = """Estimate the posterior edit probability for this prompt-response pair.

Definition:
- score = probability in [0, 1] that a human evaluator would modify the model answer.
- High score means the answer is likely the wrong option, not a valid option letter, ambiguous, unsupported, or inconsistent with the GPQA question and choices.
- Low score means the answer is likely the correct single option and would be confirmed without modification.

Rules:
- Use only the GPQA question, answer options, and model answer below.
- Do not assume access to the gold answer.
- The expected response is one option letter: A, B, C, or D.
- Treat missing final answers, multiple conflicting choices, answer text without a clear option letter, and non-answer text as evidence that human correction may be needed.
- Return only valid JSON in exactly this format:
  {{"score": 0.73}}

Question and options:
{question}

Model answer:
{model_answer}
"""


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


DEFAULT_SCORE_SYSTEM_PROMPTS = {
    "math500": MATH500_SCORE_SYSTEM_PROMPT,
    "popqa": POPQA_SCORE_SYSTEM_PROMPT,
    "mmlupro": MMLUPRO_SCORE_SYSTEM_PROMPT,
    "gpqa": GPQA_SCORE_SYSTEM_PROMPT,
}

DEFAULT_SCORE_PROMPTS = {
    "math500": MATH500_SCORE_PROMPT,
    "popqa": POPQA_SCORE_PROMPT,
    "mmlupro": MMLUPRO_SCORE_PROMPT,
    "gpqa": GPQA_SCORE_PROMPT,
}


class OursLLMSelection:


    needs_score_model = False

    def __init__(self, cfg: Dict[str, Any], score_model=None, state: Dict[str, Any] | None = None):
        self.cfg = cfg
        self.policy = cfg["policy"]
        self.seed = int(cfg.get("seed", 0))
        self.dataset_name = _canonical_dataset_name(cfg.get("data", {}).get("name", "math500"))

        state = state or {}
        self.alpha = float(state.get("alpha", self.policy.get("initial_alpha", 0.0)))
        self.score_cache: Dict[str, float] = {
            str(k): float(v) for k, v in (state.get("score_cache", {}) or {}).items()
        }

        score_llm_cfg = dict(
            cfg.get("score_llm")
            or cfg.get("selector_llm")
            or {
                "provider": "groq",
                "model_name": "llama-3.3-70b-versatile",
                "temperature": 0.0,
                "max_output_tokens": 64,
                "request_timeout": 120,
                "retry_attempts": 8,
                "retry_sleep": 2.0,
                "min_seconds_between_calls": 2.5,
            }
        )
        score_llm_cfg.setdefault("provider", "groq")
        score_llm_cfg.setdefault("model_name", "llama-3.3-70b-versatile")
        score_llm_cfg.setdefault("temperature", 0.0)
        score_llm_cfg.setdefault("max_output_tokens", 64)
        score_llm_cfg.setdefault("request_timeout", 120)
        score_llm_cfg.setdefault("retry_attempts", 8)
        score_llm_cfg.setdefault("retry_sleep", 2.0)
        score_llm_cfg.setdefault("min_seconds_between_calls", 2.5)
        if not score_llm_cfg.get("system_prompt"):
            score_llm_cfg["system_prompt"] = DEFAULT_SCORE_SYSTEM_PROMPTS.get(
                self.dataset_name,
                MATH500_SCORE_SYSTEM_PROMPT,
            )

        self.score_llm_cfg = score_llm_cfg
        self.score_llm = build_main_llm(score_llm_cfg)
        self.prompt_template = self.policy.get("score_prompt") or DEFAULT_SCORE_PROMPTS.get(
            self.dataset_name,
            MATH500_SCORE_PROMPT,
        )
        self.fallback_score = float(self.policy.get("fallback_score", 0.5))

    @staticmethod
    def _truncate(text: str, max_chars: int = 2500) -> str:
        text = str(text)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20] + " ... [truncated]"

    def _build_prompt(self, row: Dict[str, Any]) -> str:
        return self.prompt_template.format(
            question=self._truncate(row.get("question", ""), int(self.policy.get("max_question_chars", 2500))),
            model_answer=self._truncate(row.get("model_answer", ""), int(self.policy.get("max_answer_chars", 2500))),
        )

    def _cache_key(self, row: Dict[str, Any]) -> str:
        payload = {
            "model_name": self.score_llm_cfg.get("model_name"),
            "system_prompt": self.score_llm_cfg.get("system_prompt"),
            "prompt_template": self.prompt_template,
            "question": str(row.get("question", "")),
            "model_answer": str(row.get("model_answer", "")),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _coerce_score(value: Any) -> float:
        score = float(value)
        if score > 1.0 and score <= 100.0:
            score = score / 100.0
        return float(np.clip(score, 0.0, 1.0))

    @classmethod
    def _parse_score(cls, text: str) -> float:
        text = str(text).strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for key in ("score", "eta", "edit_probability", "probability"):
                    if key in obj:
                        return cls._coerce_score(obj[key])
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    for key in ("score", "eta", "edit_probability", "probability"):
                        if key in obj:
                            return cls._coerce_score(obj[key])
            except Exception:
                pass


        for number in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
            try:
                return cls._coerce_score(number)
            except Exception:
                continue

        raise ValueError(f"Could not parse an edit-probability score from response: {text[:200]!r}")

    def _score_row(self, row: Dict[str, Any], t: int, idx: int) -> float:
        key = self._cache_key(row)
        cached = self.score_cache.get(key)
        if cached is not None:
            return float(cached)

        prompt = self._build_prompt(row)
        try:
            response = self.score_llm.generate(prompt)
            score = self._parse_score(response)
        except Exception as exc:
            print(f"[ours_llm] scoring failed at batch {t}, row {idx}: {exc}")
            score = float(np.clip(self.fallback_score, 0.0, 1.0))

        self.score_cache[key] = float(score)
        return float(score)

    def _score_batch(self, batch_df: pd.DataFrame, t: int) -> np.ndarray:
        rows = batch_df.to_dict("records")
        scores = [self._score_row(row, t=t, idx=i) for i, row in enumerate(rows)]
        return np.asarray(scores, dtype=np.float32)

    def process_batch(self, batch_df: pd.DataFrame, t: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        costs = batch_df["cost"].to_numpy(dtype=float)
        A = batch_df["A"].to_numpy(dtype=float)

        budget = float(self.policy["budget_per_batch"])
        epsilon = float(self.policy["epsilon"])
        gamma = float(self.policy["alpha_step_size"])
        warm_start_batches = int(self.policy.get("warm_start_batches", 0))

        rng = np.random.default_rng(self.seed + 1000003 * int(t))

        eta = self._score_batch(batch_df, t=t)

        if t < warm_start_batches:
            selected = OursSelection._budgeted_random_selection(costs=costs, budget=budget, rng=rng)
            beta = 0.0
            selection_mode = "warm_start"
        else:
            selected, beta = OursSelection._budgeted_threshold_selection(
                eta=eta,
                costs=costs,
                alpha=self.alpha,
                budget=budget,
                rng=rng,
            )
            selection_mode = "threshold"

        n_sel = float(selected.sum())
        confirmation_rate = OursSelection._safe_ratio(
            float((selected * (1.0 - A)).sum()),
            n_sel,
        )

        old_alpha = self.alpha
        self.alpha = max(0.0, self.alpha + gamma * (confirmation_rate - epsilon))

        out = batch_df.copy()
        out["eta"] = eta
        out["alpha"] = old_alpha
        out["beta"] = beta
        out["selected"] = selected
        out["selection_mode"] = selection_mode

        return out, self.state_dict()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "score_cache": {str(k): float(v) for k, v in self.score_cache.items()},
        }
