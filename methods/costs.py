from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _response_word_count(text: Any) -> int:
    return max(1, len(str(text).split()))


def prepare_cost_context(gen_df: pd.DataFrame | None, variant: str) -> dict[str, float]:

    if variant != "review_length":
        return {}

    if gen_df is None or "model_answer" not in gen_df.columns or len(gen_df) == 0:
        return {"review_length_median": 1.0}

    lengths = gen_df["model_answer"].map(_response_word_count).to_numpy(dtype=float)
    finite = lengths[np.isfinite(lengths) & (lengths > 0)]
    median = float(np.median(finite)) if finite.size else 1.0
    if not np.isfinite(median) or median <= 0:
        median = 1.0
    return {"review_length_median": median}


def compute_cost(row: dict, variant: str, context: Mapping[str, float] | None = None) -> float:
    if variant == "constant":
        return 1.0
    if variant == "answer_length":
        return float(max(1, len(str(row.get("model_answer", "")).split())))
    if variant == "question_answer_length":
        q = len(str(row.get("question", "")).split())
        a = len(str(row.get("model_answer", "")).split())
        return float(max(1, q + a))
    if variant == "review_length":
        context = context or {}
        median = float(context.get("review_length_median", 1.0))
        if not np.isfinite(median) or median <= 0:
            median = 1.0
        length = float(_response_word_count(row.get("model_answer", "")))
        return float(np.clip(length / median, 0.25, 4.0))
    raise ValueError(f"Unknown cost variant: {variant}")
