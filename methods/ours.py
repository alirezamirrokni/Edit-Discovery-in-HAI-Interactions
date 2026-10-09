from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from utils import sigmoid


class OursSelection:


    needs_score_model = True

    def __init__(self, cfg: Dict[str, Any], score_model, state: Dict[str, Any] | None = None):
        if score_model is None:
            raise ValueError("OursSelection requires a score_model.")

        self.cfg = cfg
        self.score_model = score_model
        self.policy = cfg["policy"]

        state = state or {}

        self.seed = int(cfg.get("seed", 0))
        self.alpha = float(state.get("alpha", self.policy.get("initial_alpha", 0.0)))

        theta = state.get("theta")
        self.theta = None if theta is None else np.array(theta, dtype=np.float32)

    def _ensure_theta(self, dim: int) -> None:

        if self.theta is not None:
            return

        init_scale = float(self.policy.get("theta_init_scale", 0.0))

        if init_scale > 0:
            rng = np.random.default_rng(self.seed + 1729)
            self.theta = rng.normal(loc=0.0, scale=init_scale, size=dim).astype(np.float32)
        else:
            self.theta = np.zeros(dim, dtype=np.float32)

    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        return 0.0 if den <= 0 else float(num / den)

    @staticmethod
    def _budgeted_random_selection(costs: np.ndarray, budget: float, rng: np.random.Generator) -> np.ndarray:

        n = len(costs)
        selected = np.zeros(n, dtype=int)
        spent = 0.0

        order = np.arange(n)
        rng.shuffle(order)

        for i in order:
            c = float(costs[i])
            if c <= 0:
                continue
            if spent + c <= budget + 1e-12:
                selected[i] = 1
                spent += c

        return selected

    @staticmethod
    def _budgeted_threshold_selection(
        eta: np.ndarray,
        costs: np.ndarray,
        alpha: float,
        budget: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:

        eta = np.asarray(eta, dtype=float)
        costs = np.asarray(costs, dtype=float)

        n = len(eta)
        selected = np.zeros(n, dtype=int)
        spent = 0.0

        if n == 0 or budget <= 0:
            return selected, 0.0


        eligible = np.where((eta >= alpha) & (costs > 0))[0]

        if len(eligible) == 0:
            return selected, 0.0


        eligible = np.array(eligible, dtype=int)
        rng.shuffle(eligible)

        ratios = (eta[eligible] - alpha) / np.maximum(costs[eligible], 1e-12)


        order = eligible[np.argsort(-ratios, kind="mergesort")]

        for i in order:
            c = float(costs[i])
            if spent + c <= budget + 1e-12:
                selected[i] = 1
                spent += c

        if selected.sum() == 0:
            return selected, 0.0


        selected_idx = np.where(selected == 1)[0]
        beta = float(np.min((eta[selected_idx] - alpha) / np.maximum(costs[selected_idx], 1e-12)))
        beta = max(0.0, beta)

        return selected, beta

    def process_batch(self, batch_df: pd.DataFrame, t: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        features = self.score_model.encode_rows(batch_df.to_dict("records"))
        self._ensure_theta(features.shape[1])

        costs = batch_df["cost"].to_numpy(dtype=float)
        A = batch_df["A"].to_numpy(dtype=float)

        budget = float(self.policy["budget_per_batch"])
        epsilon = float(self.policy["epsilon"])
        gamma = float(self.policy["alpha_step_size"])
        theta_lr = float(self.policy["theta_step_size"])
        l2_reg = float(self.policy.get("l2_reg", 0.0))
        warm_start_batches = int(self.policy.get("warm_start_batches", 0))

        rng = np.random.default_rng(self.seed + 1000003 * int(t))

        eta = sigmoid(features @ self.theta).astype(np.float32)

        if t < warm_start_batches:
            selected = self._budgeted_random_selection(costs=costs, budget=budget, rng=rng)
            beta = 0.0
            selection_mode = "warm_start"
        else:
            selected, beta = self._budgeted_threshold_selection(
                eta=eta,
                costs=costs,
                alpha=self.alpha,
                budget=budget,
                rng=rng,
            )
            selection_mode = "threshold"

        n_sel = float(selected.sum())

        confirmation_rate = self._safe_ratio(
            float((selected * (1.0 - A)).sum()),
            n_sel,
        )

        old_alpha = self.alpha

        self.alpha = max(
            0.0,
            self.alpha + gamma * (confirmation_rate - epsilon),
        )

        if n_sel > 0:
            grad = ((selected * (eta - A))[:, None] * features).sum(axis=0) / n_sel

            if l2_reg > 0:
                grad = grad + l2_reg * self.theta

            self.theta = (self.theta - theta_lr * grad).astype(np.float32)

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
            "theta": None if self.theta is None else self.theta.astype(float).tolist(),
        }