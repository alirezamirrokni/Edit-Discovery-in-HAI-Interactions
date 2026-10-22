import numpy as np


class RandomSelection:
    def __init__(self, cfg):
        self.seed = int(cfg["seed"])
        self.budget = float(cfg["policy"]["budget_per_batch"])

    def process_batch(self, batch, t):
        costs = batch["cost"].to_numpy(dtype=float)
        selected = np.zeros(len(batch), dtype=int)
        spent = 0.0
        for j in np.random.default_rng(self.seed + 10_000 * t).permutation(len(batch)):
            if spent + costs[j] <= self.budget + 1e-12:
                selected[j] = 1
                spent += costs[j]
        out = batch.copy()
        out["selected"] = selected
        return out, {}
