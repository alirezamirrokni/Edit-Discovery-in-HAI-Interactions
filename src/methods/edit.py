import hashlib
import json
import re

import numpy as np

from src.llm import LLM
from src.methods.prompts import SCORE, SCORE_SYSTEM
from src.utils import safe_div, sigmoid


def threshold_selection(eta, costs, alpha, budget, rng):
    eta = np.asarray(eta, dtype=float)
    selected = np.zeros(len(eta), dtype=int)
    if len(eta) == 0 or budget <= 0:
        return selected, 0.0
    eligible = np.where((eta >= alpha) & (costs > 0))[0]
    if len(eligible) == 0:
        return selected, 0.0
    rng.shuffle(eligible)
    ratios = (eta[eligible] - alpha) / np.maximum(costs[eligible], 1e-12)
    spent = 0.0
    for i in eligible[np.argsort(-ratios, kind="mergesort")]:
        if spent + costs[i] <= budget + 1e-12:
            selected[i] = 1
            spent += costs[i]
    if selected.sum() == 0:
        return selected, 0.0
    idx = np.where(selected == 1)[0]
    return selected, max(0.0, float(np.min((eta[idx] - alpha) / np.maximum(costs[idx], 1e-12))))


class Edit:
    def __init__(self, cfg, scorer, state):
        self.policy = cfg["policy"]
        self.seed = int(cfg["seed"])
        self.scorer = scorer
        self.alpha = float(state.get("alpha", 0.0))

    def process_batch(self, batch, t):
        costs = batch["cost"].to_numpy(dtype=float)
        A = batch["A"].to_numpy(dtype=float)
        rng = np.random.default_rng(self.seed + 1000003 * int(t))
        eta = self.scorer.score(batch, t)
        selected, beta = threshold_selection(eta, costs, self.alpha, float(self.policy["budget_per_batch"]), rng)

        out = batch.copy()
        out["eta"] = eta
        out["alpha"] = self.alpha
        out["beta"] = beta
        out["selected"] = selected

        confirmation_rate = safe_div(float((selected * (1.0 - A)).sum()), float(selected.sum()))
        step = float(self.policy["alpha_step_size"]) * (confirmation_rate - float(self.policy["epsilon"]))
        self.alpha = max(0.0, self.alpha + step)
        self.scorer.update(selected, eta, A)
        return out, {"alpha": float(self.alpha), **self.scorer.state()}


class RepScorer:
    def __init__(self, cfg, state, encoder=None):
        if encoder is None:
            from src.encoders import Encoder

            encoder = Encoder(cfg["encoder"], cfg["data"]["name"])
        self.encoder = encoder
        self.seed = int(cfg["seed"])
        self.lr = float(cfg["policy"]["theta_step_size"])
        theta = state.get("theta")
        self.theta = None if theta is None else np.array(theta, dtype=np.float32)

    def score(self, batch, t):
        self.features = self.encoder.encode_rows(batch.to_dict("records"))
        if self.theta is None:
            rng = np.random.default_rng(self.seed + 1729)
            self.theta = rng.normal(0.0, 0.1, size=self.features.shape[1]).astype(np.float32)
        return sigmoid(self.features @ self.theta).astype(np.float32)

    def update(self, selected, eta, A):
        n_selected = float(selected.sum())
        if n_selected > 0:
            grad = ((selected * (eta - A))[:, None] * self.features).sum(axis=0) / n_selected
            self.theta = (self.theta - self.lr * grad).astype(np.float32)

    def state(self):
        return {"theta": self.theta.astype(float).tolist()}


def _truncate(text, max_chars):
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 20] + " ... [truncated]"


def _coerce_score(value):
    score = float(value)
    if 1.0 < score <= 100.0:
        score /= 100.0
    return float(np.clip(score, 0.0, 1.0))


def parse_score(text):
    text = str(text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    for candidate in (text, match.group(0) if match else None):
        try:
            obj = json.loads(candidate)
            for key in ("score", "eta", "edit_probability", "probability"):
                if key in obj:
                    return _coerce_score(obj[key])
        except Exception:
            pass
    for number in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return _coerce_score(number)
        except Exception:
            continue
    raise ValueError(f"Could not parse an edit-probability score from response: {text[:200]!r}")


class LLMScorer:
    def __init__(self, cfg, state):
        dataset = cfg["data"]["name"]
        self.llm_cfg = cfg["selector_llm"]
        self.system_prompt = SCORE_SYSTEM[dataset]
        self.template = SCORE[dataset]
        self.llm = LLM(self.llm_cfg, self.system_prompt)
        self.cache = {str(k): float(v) for k, v in state.get("score_cache", {}).items()}

    def _key(self, row):
        payload = {
            "model_name": self.llm_cfg["model"],
            "system_prompt": self.system_prompt,
            "prompt_template": self.template,
            "question": str(row.get("question", "")),
            "model_answer": str(row.get("model_answer", "")),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    def _score_row(self, row, t, idx):
        key = self._key(row)
        if key not in self.cache:
            prompt = self.template.format(
                question=_truncate(row.get("question", ""), 2500),
                model_answer=_truncate(row.get("model_answer", ""), 2500),
            )
            try:
                self.cache[key] = parse_score(self.llm.generate(prompt))
            except Exception as exc:
                print(f"[edit_llm] scoring failed at batch {t}, row {idx}: {exc}")
                self.cache[key] = 0.5
        return self.cache[key]

    def score(self, batch, t):
        rows = batch.to_dict("records")
        return np.asarray([self._score_row(row, t, i) for i, row in enumerate(rows)], dtype=np.float32)

    def update(self, selected, eta, A):
        pass

    def state(self):
        return {"score_cache": dict(self.cache)}


class RandomScorer:
    def __init__(self, cfg):
        self.seed = int(cfg["seed"])

    def score(self, batch, t):
        rng = np.random.default_rng(self.seed + 271828 + 2000003 * int(t))
        return rng.uniform(0.0, 1.0, size=len(batch)).astype(np.float32)

    def update(self, selected, eta, A):
        pass

    def state(self):
        return {}
