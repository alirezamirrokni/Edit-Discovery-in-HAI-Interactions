from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from datasets import load_dataset


GPQA_MAIN_SYSTEM_PROMPT = """You are a careful graduate-level science reasoning system.

Hard requirements:
- Solve the multiple-choice question and choose exactly one option.
- The final answer must be only the option letter.
- End with exactly one final line in this format:
#### <letter>

The final line must contain only the marker #### followed by a single letter: A, B, C, or D."""


_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_hf_dataset(hf_name: str, hf_config: Any, split: str):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    kwargs: Dict[str, Any] = {"split": split}
    if token:
        kwargs["token"] = token
    if hf_config in {None, "", "null", "none", "default"}:
        return load_dataset(hf_name, **kwargs)
    return load_dataset(hf_name, hf_config, **kwargs)


def _clean_text(value: Any) -> str:
    return str(value).strip()


def _first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in row and row.get(key) is not None:
            text = _clean_text(row.get(key))
            if text:
                return text
    return None


def _choice_letter_from_index(index: Any) -> Optional[str]:
    try:
        idx = int(index)
    except Exception:
        return None
    if 0 <= idx < len(_CHOICE_LETTERS):
        return _CHOICE_LETTERS[idx]
    return None


def format_gpqa_question(question: str, options: Sequence[str], subject: str | None = None) -> str:
    option_lines = [f"{_CHOICE_LETTERS[idx]}. {str(option).strip()}" for idx, option in enumerate(options)]
    subject_line = f"Subject: {subject}\n\n" if subject else ""
    return f"{subject_line}Question:\n{question.strip()}\n\nOptions:\n" + "\n".join(option_lines)


def extract_gpqa_choice(text: str | None, num_options: int = 4) -> Optional[str]:
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


def gpqa_exact_match(pred_answer: str | None, gold_final: str | None) -> bool:
    if pred_answer is None or gold_final is None:
        return False
    return str(pred_answer).strip().upper() == str(gold_final).strip().upper()


class GPQAWrapper:


    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.subset_size = cfg.get("subset_size", cfg.get("max_samples", None))
        self.subset_size = None if self.subset_size in {None, "", "null", "none"} else int(self.subset_size)
        self.subset_seed = int(cfg.get("subset_seed", 42))
        self.subset_strategy = str(cfg.get("subset_strategy", "random")).lower()
        self.shuffle_choices = bool(cfg.get("shuffle_choices", True))

    def _select_source_indices(self, n_total: int) -> List[int]:
        if self.subset_size is None:
            return list(range(n_total))
        if self.subset_size <= 0:
            raise ValueError("data.subset_size/max_samples must be positive for gpqa.")
        if self.subset_size > n_total:
            raise ValueError(
                f"Requested gpqa subset_size={self.subset_size}, but split has only {n_total} rows."
            )
        if self.subset_size == n_total:
            return list(range(n_total))

        if self.subset_strategy in {"random", "shuffle", "sample"}:
            rng = np.random.default_rng(self.subset_seed)
            return rng.permutation(n_total)[: self.subset_size].astype(int).tolist()

        if self.subset_strategy in {"first", "sequential", "head"}:
            return list(range(self.subset_size))

        raise ValueError(
            f"Unknown gpqa subset_strategy={self.subset_strategy!r}. "
            "Use 'random' or 'first'."
        )

    def _choices_for_row(self, row: Dict[str, Any], source_idx: int) -> tuple[List[str], str]:
        question = _first_nonempty(row, ("Question", "question", "input", "prompt"))
        if not question:
            raise ValueError(f"GPQA example at source index {source_idx} has no question.")

        correct = _first_nonempty(row, ("Correct Answer", "correct_answer", "answer", "target"))
        if not correct:
            raise ValueError(f"GPQA example at source index {source_idx} has no correct answer.")

        incorrects = []
        for key in (
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
            "incorrect_answer_1",
            "incorrect_answer_2",
            "incorrect_answer_3",
        ):
            if key in row and row.get(key) is not None:
                text = _clean_text(row.get(key))
                if text:
                    incorrects.append(text)
        if len(incorrects) < 3:

            options = row.get("options") or row.get("choices")
            if isinstance(options, (list, tuple)) and len(options) >= 4:
                opts = [_clean_text(x) for x in options]
                gold_letter = None
                answer = row.get("answer") or row.get("gold") or row.get("label")
                if answer is not None:
                    m = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", str(answer).strip(), flags=re.IGNORECASE)
                    if m:
                        gold_letter = m.group(1).upper()
                if gold_letter and gold_letter in _CHOICE_LETTERS[: len(opts)]:
                    return opts[:4], gold_letter

            raise ValueError(f"GPQA example at source index {source_idx} has fewer than three incorrect answers.")

        labeled_choices = [(correct, True)] + [(x, False) for x in incorrects[:3]]
        if self.shuffle_choices:
            rng = np.random.default_rng(self.subset_seed + 1000003 * int(source_idx))
            order = rng.permutation(len(labeled_choices)).tolist()
            labeled_choices = [labeled_choices[i] for i in order]

        options = [text for text, _ in labeled_choices]
        gold_idx = next(idx for idx, (_, is_correct) in enumerate(labeled_choices) if is_correct)
        return options, _CHOICE_LETTERS[gold_idx]

    def load_records(self) -> List[Dict[str, Any]]:
        hf_name = self.cfg.get("hf_name", "Idavidrein/gpqa")
        hf_config = self.cfg.get("hf_config", "gpqa_diamond")
        split = self.cfg.get("split", "train")

        ds = _load_hf_dataset(hf_name, hf_config, split)
        selected_source_indices = self._select_source_indices(len(ds))

        records: List[Dict[str, Any]] = []
        for source_idx in selected_source_indices:
            row = dict(ds[int(source_idx)])
            question = _first_nonempty(row, ("Question", "question", "input", "prompt"))
            options, gold_letter = self._choices_for_row(row, int(source_idx))
            gold_idx = _CHOICE_LETTERS.index(gold_letter)
            gold_answer = options[gold_idx]
            subject = _first_nonempty(row, ("Subdomain", "High-level domain", "domain", "subject", "category"))
            formatted_question = format_gpqa_question(str(question), options, subject)

            records.append(
                {
                    "example_id": int(source_idx),
                    "question": formatted_question,
                    "gold_answer": f"{gold_letter}. {gold_answer}",
                    "gold_final": gold_letter,
                }
            )

        return records

    @staticmethod
    def parse_prediction(model_answer: str) -> str | None:
        return extract_gpqa_choice(model_answer, num_options=4)

    @staticmethod
    def failure_label(pred_answer: str | None, gold_final: str | None) -> int:
        return int(not gpqa_exact_match(pred_answer, gold_final))

    @staticmethod
    def build_prompt(question: str) -> str:
        return (
            "Solve the following GPQA graduate-level science multiple-choice question.\n"
            "Choose the single best option. Think carefully if needed.\n"
            "End with exactly one final line in this format: #### <letter>\n\n"
            f"{question}"
        )

    @staticmethod
    def main_system_prompt() -> str:
        return GPQA_MAIN_SYSTEM_PROMPT
