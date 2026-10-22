import json
import re

import numpy as np

from src.llm import LLM
from src.methods.prompts import SELECT, SELECT_SYSTEM

QUESTION_LABELS = {"math500": "Problem", "mmlupro": "Question and options", "gpqa": "Question and options", "popqa": "Question"}


def _truncate(text, max_chars=900):
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 20] + " ... [truncated]"


def parse_indices(text):
    text = str(text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    for candidate in (text, match.group(0) if match else None):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and isinstance(obj.get("selected_indices"), list):
                return [int(x) for x in obj["selected_indices"]]
        except Exception:
            pass
    return [int(x) for x in re.findall(r"-?\d+", text)]


class LLMSelect:
    def __init__(self, cfg):
        self.dataset = cfg["data"]["name"]
        self.budget = float(cfg["policy"]["budget_per_batch"])
        self.llm = LLM(cfg["selector_llm"], SELECT_SYSTEM[self.dataset])

    def build_prompt(self, batch):
        label = QUESTION_LABELS[self.dataset]
        items = "\n---\n".join(
            f"Index: {i}\nCost: {float(row['cost']):g}\n{label}:\n{_truncate(row['question'])}\nModel answer:\n{_truncate(row['model_answer'])}\n"
            for i, row in batch.reset_index(drop=True).iterrows()
        )
        prompt = SELECT[self.dataset].format(budget=f"{self.budget:g}", items=items)
        if len(prompt) > 8000:
            prompt = prompt[:8000].rstrip() + "\n\n[Prompt truncated due to input-size limit.]"
        return prompt

    def process_batch(self, batch, t):
        costs = batch["cost"].to_numpy(dtype=float)
        selected = np.zeros(len(batch), dtype=int)
        try:
            indices = parse_indices(self.llm.generate(self.build_prompt(batch)))
        except Exception as exc:
            print(f"[llm_select] selector call/parse failed at batch {t}: {exc}")
            indices = []
        spent = 0.0
        for idx in indices:
            if 0 <= idx < len(costs) and not selected[idx] and costs[idx] > 0 and spent + costs[idx] <= self.budget + 1e-12:
                selected[idx] = 1
                spent += costs[idx]
        out = batch.copy()
        out["selected"] = selected
        return out, {}
