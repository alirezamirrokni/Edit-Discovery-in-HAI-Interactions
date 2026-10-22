from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_yaml, output_paths, safe_div

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
MODELS = {"llama": "Llama-3.3-70B-Instruct", "qwen": "Qwen3-32B"}
DATASETS = {"math500": "MATH-500", "mmlupro": "MMLU-Pro", "popqa": "PopQA", "gpqa": "GPQA"}
METHODS = {"edit_rep": "EDIT-REP", "edit_llm": "EDIT-LLM", "edit_rand": "EDIT-RAND", "llm_select": "LLM-Select"}


def config(model, dataset, name):
    return load_yaml(CONFIG_DIR / model / dataset / f"{name}.yaml")


def load_run(model, dataset, name):
    path = output_paths(config(model, dataset, name))["run_csv"]
    return pd.read_csv(path) if path.exists() else None


def load_generations(model, dataset):
    path = output_paths(config(model, dataset, "random"))["generations"]
    return pd.read_csv(path).drop_duplicates("example_id", keep="last").reset_index(drop=True)


def running_metrics(run):
    rows = []
    for t, g in run.groupby("t", sort=True):
        s, a, c = (g[k].to_numpy(dtype=float) for k in ("selected", "A", "cost"))
        limit = float(g["budget"].iloc[0])
        cost = float((s * c).sum())
        rows.append(
            {
                "Round": int(t) + 1,
                "Type-I": safe_div(float((s * (1 - a)).sum()), float(s.sum())),
                "Type-II": safe_div(float(((1 - s) * a).sum()), float((1 - s).sum())),
                "Cost": min(cost, limit) if cost <= limit + 1e-9 else cost,
            }
        )
    df = pd.DataFrame(rows)
    n = np.arange(1, len(df) + 1)
    for col in ("Type-I", "Type-II", "Cost"):
        df[col] = df[col].cumsum() / n
    df["epsilon"] = run["epsilon"].iloc[0]
    df["budget"] = float(run["budget"].iloc[0])
    return df


def discovery_metrics(run):
    per_round = [
        (
            safe_div(float((g.selected * (1 - g.A)).sum()), float(g.selected.sum())),
            safe_div(float(((1 - g.selected) * g.A).sum()), float((1 - g.selected).sum())),
            float((g.selected * g.cost).sum()),
        )
        for _, g in run.astype({"selected": float, "A": float, "cost": float}).groupby("t", sort=True)
    ]
    s, a = run["selected"].to_numpy(dtype=float), run["A"].to_numpy(dtype=float)
    type_i, type_ii, cost = (float(np.mean(x)) for x in zip(*per_round))
    return {
        "type_i": type_i,
        "type_ii": type_ii,
        "mean_review_cost": cost,
        "edit_recall": safe_div(float((s * a).sum()), float(a.sum())),
        "missed_edits": float(((1 - s) * a).sum()),
        "edit_precision": safe_div(float((s * a).sum()), float(s.sum())),
    }
