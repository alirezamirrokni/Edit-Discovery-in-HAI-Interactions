from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .ours import OursSelection


class OursRandomSelection:


    needs_score_model = False

    def __init__(self, cfg: Dict[str, Any], score_model=None, state: Dict[str, Any] | None = None):
        self.cfg = cfg
        self.policy = cfg["policy"]
        self.seed = int(cfg.get("seed", 0))

        state = state or {}
        self.alpha = float(state.get("alpha", self.policy.get("initial_alpha", 0.0)))
        self.random_score_seed_offset = int(self.policy.get("random_score_seed_offset", 271828))

    def _score_batch(self, batch_df: pd.DataFrame, t: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + self.random_score_seed_offset + 2000003 * int(t))
        return rng.uniform(0.0, 1.0, size=len(batch_df)).astype(np.float32)

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
        return {"alpha": float(self.alpha)}
