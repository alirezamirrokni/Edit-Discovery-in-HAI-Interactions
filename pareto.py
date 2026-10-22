import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plots import TEXT, TICK, set_style
from src.analysis import DATASETS, MODELS, config, discovery_metrics, load_generations, load_run
from src.methods import build_method

BUDGET_COLORS = {2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c", 5: "#d62728", 7: "#9467bd"}
EPS_MARKERS = {0.1: "o", 0.2: "s", 0.3: "D", 0.4: "^", 0.5: "v", 0.6: "P"}


class Precomputed:
    def __init__(self, features):
        self.features = features

    def encode_rows(self, rows):
        return self.features[[int(r["pool_pos"]) for r in rows]]


def load_features(model, key, cfg, pool, cache_dir):
    path = cache_dir / f"{model}_{key}_minilm.npz"
    ids = pool["example_id"].astype(int).to_numpy()
    if path.exists():
        obj = np.load(path)
        if np.array_equal(obj["example_id"], ids):
            return obj["features"].astype(np.float32)
    from src.encoders import Encoder

    features = Encoder(cfg["encoder"], cfg["data"]["name"]).encode_rows(pool[["question", "model_answer"]].to_dict("records"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, example_id=ids, features=features)
    return features


def sweep_run(cfg, pool, features, seed, epsilon, budget, num_batches, batch_size):
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    cfg["policy"].update(epsilon=epsilon, budget_per_batch=budget)
    method = build_method(cfg, {}, encoder=Precomputed(features))
    stream = np.random.default_rng(seed).choice(len(pool), size=(num_batches, batch_size), replace=True)
    out = []
    for t, idx in enumerate(stream):
        batch = pd.DataFrame(
            {
                "t": t,
                "batch_pos": np.arange(batch_size),
                "pool_pos": idx,
                "example_id": pool["example_id"].to_numpy()[idx],
                "A": pool["A"].astype(int).to_numpy()[idx],
                "cost": np.ones(batch_size),
            }
        )
        out.append(method.process_batch(batch, t)[0])
    return pd.concat(out, ignore_index=True)


def plot(summary, baselines, budgets, eps_grid, x, y, xlabel, ylabel, path):
    set_style("talk")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for i, (ax, label) in enumerate(zip(axes.flatten(), DATASETS.values())):
        ds = summary[summary["Dataset"] == label]
        for b in budgets:
            sub = ds[ds["budget"] == b].sort_values("epsilon")
            ax.plot(sub[f"{x}_mean"], sub[f"{y}_mean"], color=BUDGET_COLORS[b], linewidth=2.6, alpha=0.95)
            for _, row in sub.iterrows():
                ax.errorbar(
                    row[f"{x}_mean"], row[f"{y}_mean"], xerr=row[f"{x}_ci95"], yerr=row[f"{y}_ci95"],
                    fmt=EPS_MARKERS[round(row["epsilon"], 1)], color=BUDGET_COLORS[b], markersize=8,
                    markeredgewidth=1.0, capsize=2.5, elinewidth=1.2, capthick=1.2, zorder=3,
                )
        base = baselines[baselines["Dataset"] == label]
        if len(base):
            ax.scatter(base[x], base[y], marker="x", s=150, linewidths=2.7, color="black", zorder=5)
        ax.set_title(label, fontsize=TEXT, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel if i >= 2 else "")
        ax.set_ylabel(ylabel if i % 2 == 0 else "")
        if y == "edit_recall":
            ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="both", labelsize=TICK)
        ax.grid(True, alpha=0.35, linewidth=0.8)
    method_handles = [Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=12, markeredgewidth=2.5, label="LLM-Select, B=5")]
    method_handles += [Line2D([0], [0], color=BUDGET_COLORS[b], linewidth=2.6, marker="o", markersize=7, label=f"EDIT-REP, B={b}") for b in budgets]
    eps_handles = [Line2D([0], [0], color="black", marker=EPS_MARKERS[round(e, 1)], linestyle="None", markersize=8, label=rf"$\epsilon={e:.1f}$") for e in eps_grid]
    style = {"loc": "lower center", "frameon": False, "prop": {"size": 13, "weight": "bold"}, "handletextpad": 0.5}
    fig.add_artist(fig.legend(handles=method_handles, bbox_to_anchor=(0.5, 0.055), ncol=len(method_handles), handlelength=1.9, columnspacing=1.0, **style))
    fig.legend(handles=eps_handles, bbox_to_anchor=(0.5, 0.005), ncol=len(eps_handles), handlelength=1.2, columnspacing=1.2, **style)
    fig.tight_layout(rect=(0.02, 0.14, 0.995, 0.99), w_pad=2.0, h_pad=2.1)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="llama")
    parser.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 3, 4, 5, 7])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--num_batches", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--out_dir", type=Path, default=Path("figures/pareto"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs, baselines = [], []
    for key, label in DATASETS.items():
        cfg = config(args.model, key, "edit_rep")
        pool = load_generations(args.model, key)
        features = load_features(args.model, key, cfg, pool, args.out_dir / "cache")
        for seed in args.seeds:
            for eps in args.eps:
                for budget in args.budgets:
                    print(f"[{label}] seed={seed} epsilon={eps} B={budget}")
                    run = sweep_run(cfg, pool, features, seed, eps, budget, args.num_batches, args.batch_size)
                    runs.append({"Dataset": label, "seed": seed, "epsilon": eps, "budget": budget, **discovery_metrics(run)})
        llm_select = load_run(args.model, key, "llm_select")
        if llm_select is not None:
            baselines.append({"Dataset": label, **discovery_metrics(llm_select)})

    runs = pd.DataFrame(runs)
    metrics = ["type_i", "type_ii", "edit_recall", "mean_review_cost"]
    grouped = runs.groupby(["Dataset", "epsilon", "budget"], sort=False)[metrics]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).fillna(0.0).mul(1.96).div(np.sqrt(grouped.size()), axis=0).add_suffix("_ci95")).reset_index()
    baselines = pd.DataFrame(baselines, columns=["Dataset", *metrics])
    runs.to_csv(args.out_dir / "pareto_runs.csv", index=False)
    summary.to_csv(args.out_dir / "pareto_summary.csv", index=False)
    plot(summary, baselines, args.budgets, args.eps, "type_i", "type_ii", "Type-I error", "Type-II error", args.out_dir / "pareto_type_i_type_ii")
    plot(summary, baselines, args.budgets, args.eps, "mean_review_cost", "edit_recall", "Cost", "Edit recall", args.out_dir / "pareto_cost_recall")


if __name__ == "__main__":
    main()
