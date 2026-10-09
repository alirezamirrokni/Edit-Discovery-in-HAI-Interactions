from __future__ import annotations

import json
import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from datasets import load_dataset


MMLUPRO_MAIN_SYSTEM_PROMPT = """You are a careful multiple-choice reasoning system.

Hard requirements:
- Solve the question and choose exactly one option.
- The final answer must be only the option letter.
- End with exactly one final line in this format:
#### <letter>

The final line must contain only the marker #### followed by a single letter such as A, B, C, D, E, F, G, H, I, or J."""


_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _load_hf_dataset(hf_name: str, hf_config: Any, split: str):
    if hf_config in {None, "", "null", "none", "default"}:
        return load_dataset(hf_name, split=split)
    return load_dataset(hf_name, hf_config, split=split)


def _as_options(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
        try:
            import ast

            obj = ast.literal_eval(text)
            if isinstance(obj, (list, tuple)):
                return [str(x) for x in obj]
        except Exception:
            pass
        return [part.strip() for part in re.split(r"\s*\|\|\|\s*|\n+", text) if part.strip()]
    return [str(value)]


def _choice_letter_from_index(index: Any) -> Optional[str]:
    try:
        idx = int(index)
    except Exception:
        return None
    if 0 <= idx < len(_CHOICE_LETTERS):
        return _CHOICE_LETTERS[idx]
    return None


def _choice_letter_from_answer(answer: Any, options: Sequence[str]) -> Optional[str]:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None


    m = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", text, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        if letter in _CHOICE_LETTERS[: max(1, len(options))]:
            return letter


    letter = _choice_letter_from_index(text)
    if letter is not None and _CHOICE_LETTERS.index(letter) < len(options):
        return letter


    norm_answer = _normalize_text(text)
    for idx, opt in enumerate(options):
        if _normalize_text(opt) == norm_answer:
            return _CHOICE_LETTERS[idx]
    return None


def _normalize_text(text: Any) -> str:
    s = str(text).lower().translate(_PUNCT_TABLE)
    return " ".join(s.split())


def format_mmlupro_question(question: str, options: Sequence[str], category: str | None = None) -> str:
    option_lines = []
    for idx, option in enumerate(options):
        option_lines.append(f"{_CHOICE_LETTERS[idx]}. {option}")
    category_line = f"Category: {category}\n\n" if category else ""
    return f"{category_line}Question:\n{question}\n\nOptions:\n" + "\n".join(option_lines)


def extract_mmlupro_choice(text: str | None, num_options: int = 10) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    valid = _CHOICE_LETTERS[: max(1, min(int(num_options), len(_CHOICE_LETTERS)))]

    marker_matches = re.findall(r"####\s*\(?\s*([A-Z])\s*\)?", s, flags=re.IGNORECASE)
    if marker_matches:
        cand = marker_matches[-1].upper()
        return cand if cand in valid else None

    patterns = [
        r"final\s+answer\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?",
        r"answer\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?",
        r"option\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?",
        r"^\s*\(?\s*([A-Z])\s*\)?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, s, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            cand = matches[-1].upper()
            if cand in valid:
                return cand


    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        m = re.search(r"\b([A-Z])\b", lines[-1], flags=re.IGNORECASE)
        if m:
            cand = m.group(1).upper()
            if cand in valid:
                return cand

    return None


def mmlupro_exact_match(pred_answer: str | None, gold_final: str | None) -> bool:
    if pred_answer is None or gold_final is None:
        return False
    return str(pred_answer).strip().upper() == str(gold_final).strip().upper()


class MMLUProWrapper:


    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.subset_size = int(cfg.get("subset_size", cfg.get("max_samples", 500)))
        self.subset_seed = int(cfg.get("subset_seed", 42))
        self.subset_strategy = str(cfg.get("subset_strategy", "stratified")).lower()

    def _select_source_indices(self, ds) -> List[int]:
        n_total = len(ds)
        if self.subset_size <= 0:
            raise ValueError("data.subset_size must be positive for mmlupro.")
        if self.subset_size > n_total:
            raise ValueError(
                f"Requested mmlupro subset_size={self.subset_size}, but split has only {n_total} rows."
            )

        if self.subset_strategy in {"random", "shuffle", "sample"}:
            rng = np.random.default_rng(self.subset_seed)
            return rng.permutation(n_total)[: self.subset_size].tolist()

        if self.subset_strategy in {"stratified", "category", "subject"}:
            groups: Dict[str, List[int]] = defaultdict(list)
            for idx in range(n_total):
                row = ds[int(idx)]
                category = str(row.get("category") or row.get("subject") or row.get("domain") or "unknown")
                groups[category].append(idx)

            raw_alloc = {
                cat: self.subset_size * len(indices) / float(n_total)
                for cat, indices in groups.items()
            }
            alloc = {cat: int(math.floor(v)) for cat, v in raw_alloc.items()}
            remaining = self.subset_size - sum(alloc.values())
            remainders = sorted(
                raw_alloc,
                key=lambda c: (raw_alloc[c] - alloc[c], len(groups[c]), c),
                reverse=True,
            )
            for cat in remainders[:remaining]:
                alloc[cat] += 1

            rng = np.random.default_rng(self.subset_seed)
            selected: List[int] = []
            for cat in sorted(groups):
                k = alloc.get(cat, 0)
                if k <= 0:
                    continue
                chosen = rng.choice(np.asarray(groups[cat], dtype=int), size=k, replace=False)
                selected.extend(int(x) for x in chosen.tolist())
            return sorted(selected)

        raise ValueError(
            f"Unknown mmlupro subset_strategy={self.subset_strategy!r}. "
            "Use 'stratified' or 'random'."
        )

    def load_records(self) -> List[Dict[str, Any]]:
        hf_name = self.cfg.get("hf_name", "TIGER-Lab/MMLU-Pro")
        hf_config = self.cfg.get("hf_config", None)
        split = self.cfg.get("split", "test")

        ds = _load_hf_dataset(hf_name, hf_config, split)
        selected_source_indices = self._select_source_indices(ds)

        records: List[Dict[str, Any]] = []
        for source_idx in selected_source_indices:
            row = dict(ds[int(source_idx)])
            raw_question = row.get("question") or row.get("Question") or row.get("input")
            options = _as_options(row.get("options") or row.get("choices"))
            category = row.get("category") or row.get("subject") or row.get("domain")
            if not raw_question:
                raise ValueError(f"MMLU-Pro example at source index {source_idx} has no question.")
            if len(options) < 2:
                raise ValueError(f"MMLU-Pro example at source index {source_idx} has fewer than two options.")

            answer = row.get("answer")
            if answer is None:
                answer = row.get("gold") or row.get("label") or row.get("target")
            gold_letter = _choice_letter_from_answer(answer, options)
            if gold_letter is None:
                gold_letter = _choice_letter_from_index(row.get("answer_index"))
            if gold_letter is None:
                raise ValueError(f"Could not infer gold option for MMLU-Pro source index {source_idx}.")

            gold_idx = _CHOICE_LETTERS.index(gold_letter)
            gold_answer = options[gold_idx] if gold_idx < len(options) else gold_letter
            formatted_question = format_mmlupro_question(str(raw_question), options, str(category) if category else None)

            records.append(
                {
                    "example_id": int(source_idx),
                    "question": formatted_question,
                    "gold_answer": f"{gold_letter}. {gold_answer}",
                    "gold_final": gold_letter,


                    "category": str(category) if category else "unknown",
                    "source_index": int(source_idx),
                }
            )

        return records

    @staticmethod
    def parse_prediction(model_answer: str) -> str | None:


        return extract_mmlupro_choice(model_answer, num_options=10)

    @staticmethod
    def failure_label(pred_answer: str | None, gold_final: str | None) -> int:
        return int(not mmlupro_exact_match(pred_answer, gold_final))

    @staticmethod
    def build_prompt(question: str) -> str:
        return (
            "Solve the following MMLU-Pro multiple-choice question.\n"
            "Choose the single best option. Think carefully if needed, but do not include reasoning in the output.\n"
            "End with exactly one final line in this format: #### <letter>\n\n"
            f"{question}"
        )

    @staticmethod
    def main_system_prompt() -> str:
        return MMLUPRO_MAIN_SYSTEM_PROMPT
