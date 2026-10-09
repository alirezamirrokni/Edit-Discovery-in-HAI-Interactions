from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


class RandomSelection:


    needs_score_model = False

    def __init__(self, cfg: Dict[str, Any], score_model=None, state: Dict[str, Any] | None = None):
        self.cfg = cfg
        self.policy = cfg["policy"]
        self.seed = int(cfg.get("seed", 0))

    def process_batch(self, batch_df: pd.DataFrame, t: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        budget = float(self.policy["budget_per_batch"])
        rng = np.random.default_rng(self.seed + 10_000 * t)
        order = rng.permutation(len(batch_df))
        selected = np.zeros(len(batch_df), dtype=int)
        spent = 0.0
        costs = batch_df["cost"].to_numpy(dtype=float)
        for j in order:
            if spent + costs[j] <= budget + 1e-12:
                selected[j] = 1
                spent += costs[j]

        out = batch_df.copy()
        out["eta"] = np.nan
        out["alpha"] = np.nan
        out["beta"] = np.nan
        out["selected"] = selected
        return out, {}
