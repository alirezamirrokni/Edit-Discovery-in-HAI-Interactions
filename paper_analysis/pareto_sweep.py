from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from paper_analysis.common import (
    DATASET_SPECS,
    MODEL_LABELS,
    compute_metrics,
    load_generation_pool,
    load_main_config,
    load_or_build_minilm_features,
    locate_main_run_files,
    run_edit_rep_offline,
    sample_stream,
    stream_from_existing_run,
    summarize_mean_ci,
)


PLOT_TEXT_SIZE = 20
PLOT_TICK_SIZE = 18
BUDGET_COLORS = {
    2: "#1f77b4",
    3: "#ff7f0e",
    4: "#2ca02c",
    5: "#d62728",
    7: "#9467bd",
}
EPS_MARKERS = {
    0.1: "o",
    0.2: "s",
    0.3: "D",
    0.4: "^",
    0.5: "v",
    0.6: "P",
}


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the paper's offline EDIT-REP operating-point sweep from "
            "cached interactions. No response-model or selector-model API is called."
        )
    )
    parser.add_argument(
        "--model",
        choices=("llama", "qwen"),
        default="llama",
        help="Response model whose cached interactions are swept (paper default: llama).",
    )
    parser.add_argument(
        "--eps",
        default="0.1,0.2,0.3,0.4,0.5,0.6",
        help="Comma-separated epsilon grid.",
    )
    parser.add_argument(
        "--budgets",
        default="2,3,4,5,7",
        help="Comma-separated per-round review-budget grid.",
    )
    parser.add_argument(
        "--seeds",
        default="7,17,27,37,47",
        help="Comma-separated stream seeds.",
    )
    parser.add_argument("--num-batches", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("paper_artifacts/pareto_sweep"),
        help="Output directory for summaries/figures (never outputs/).",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("paper_artifacts/cache"),
        help="Local cache for frozen MiniLM features.",
    )
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(context="talk", style="darkgrid", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.size": PLOT_TEXT_SIZE,
            "axes.titlesize": PLOT_TEXT_SIZE,
            "axes.labelsize": PLOT_TEXT_SIZE,
            "xtick.labelsize": PLOT_TICK_SIZE,
            "ytick.labelsize": PLOT_TICK_SIZE,
            "legend.fontsize": PLOT_TEXT_SIZE,
            "legend.title_fontsize": PLOT_TEXT_SIZE,
            "figure.titlesize": PLOT_TEXT_SIZE,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )


def _method_and_epsilon_legends(
    fig: plt.Figure, budgets: list[int], eps_grid: list[float]
) -> None:
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            linestyle="None",
            markersize=12,
            markeredgewidth=2.5,
            label="LLM-Select, B=5",
        )
    ]
    for budget in budgets:
        method_handles.append(
            Line2D(
                [0],
                [0],
                color=BUDGET_COLORS[budget],
                linewidth=2.6,
                marker="o",
                markersize=7,
                label=f"EDIT-REP, B={budget}",
            )
        )
    eps_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=EPS_MARKERS[round(float(eps), 1)],
            linestyle="None",
            markersize=8,
            label=fr"$\epsilon={eps:.1f}$",
        )
        for eps in eps_grid
    ]


    first = fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=len(method_handles),
        frameon=False,
        prop={"size": 13, "weight": "bold"},
        handlelength=1.9,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.add_artist(first)
    fig.legend(
        handles=eps_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(eps_handles),
        frameon=False,
        prop={"size": 13, "weight": "bold"},
        handlelength=1.2,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def plot_type_i_type_ii(
    summary: pd.DataFrame,
    baselines: pd.DataFrame,
    budgets: list[int],
    eps_grid: list[float],
    artifact_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=False)
    axes = axes.flatten()
    for panel_idx, (ax, spec) in enumerate(zip(axes, DATASET_SPECS)):
        ds = summary[summary["Dataset"] == spec.label].copy()
        ax.set_title(spec.label, fontsize=PLOT_TEXT_SIZE, fontweight="bold", pad=12)
        for budget in budgets:
            sub = ds[ds["budget"] == float(budget)].sort_values("epsilon")
            if sub.empty:
                continue
            color = BUDGET_COLORS[budget]
            ax.plot(sub["type_i_mean"], sub["type_ii_mean"], color=color, linewidth=2.6, alpha=0.95)
            for _, row in sub.iterrows():
                ax.errorbar(
                    row["type_i_mean"],
                    row["type_ii_mean"],
                    xerr=row["type_i_ci95"],
                    yerr=row["type_ii_ci95"],
                    fmt=EPS_MARKERS[round(float(row["epsilon"]), 1)],
                    color=color,
                    markersize=8,
                    markeredgewidth=1.0,
                    capsize=2.5,
                    elinewidth=1.2,
                    capthick=1.2,
                    zorder=3,
                )
        baseline = baselines[baselines["Dataset"] == spec.label]
        if not baseline.empty:
            row = baseline.iloc[0]
            ax.scatter(
                [row["type_i"]],
                [row["type_ii"]],
                marker="x",
                s=150,
                linewidths=2.7,
                color="black",
                zorder=5,
            )
        ax.set_xlabel("Type-I error" if panel_idx >= 2 else "")
        ax.set_ylabel("Type-II error" if panel_idx % 2 == 0 else "")
        ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)
        ax.grid(True, alpha=0.35, linewidth=0.8)

    _method_and_epsilon_legends(fig, budgets, eps_grid)
    fig.tight_layout(rect=(0.02, 0.14, 0.995, 0.99), w_pad=2.0, h_pad=2.1)
    fig.savefig(artifact_dir / "fig_pareto_typeI_typeII_all_datasets.pdf", bbox_inches="tight")
    fig.savefig(
        artifact_dir / "fig_pareto_typeI_typeII_all_datasets.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_cost_recall(
    summary: pd.DataFrame,
    baselines: pd.DataFrame,
    budgets: list[int],
    eps_grid: list[float],
    artifact_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=False)
    axes = axes.flatten()
    for panel_idx, (ax, spec) in enumerate(zip(axes, DATASET_SPECS)):
        ds = summary[summary["Dataset"] == spec.label].copy()
        ax.set_title(spec.label, fontsize=PLOT_TEXT_SIZE, fontweight="bold", pad=12)
        for budget in budgets:
            sub = ds[ds["budget"] == float(budget)].sort_values("epsilon")
            if sub.empty:
                continue
            color = BUDGET_COLORS[budget]
            ax.plot(
                sub["mean_review_cost_mean"],
                sub["edit_recall_mean"],
                color=color,
                linewidth=2.6,
                alpha=0.95,
            )
            for _, row in sub.iterrows():
                ax.errorbar(
                    row["mean_review_cost_mean"],
                    row["edit_recall_mean"],
                    xerr=row["mean_review_cost_ci95"],
                    yerr=row["edit_recall_ci95"],
                    fmt=EPS_MARKERS[round(float(row["epsilon"]), 1)],
                    color=color,
                    markersize=8,
                    markeredgewidth=1.0,
                    capsize=2.5,
                    elinewidth=1.2,
                    capthick=1.2,
                    zorder=3,
                )
        baseline = baselines[baselines["Dataset"] == spec.label]
        if not baseline.empty:
            row = baseline.iloc[0]
            ax.scatter(
                [row["mean_review_cost"]],
                [row["edit_recall"]],
                marker="x",
                s=150,
                linewidths=2.7,
                color="black",
                zorder=5,
            )
        ax.set_xlabel("Cost" if panel_idx >= 2 else "")
        ax.set_ylabel("Edit recall" if panel_idx % 2 == 0 else "")
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)
        ax.grid(True, alpha=0.35, linewidth=0.8)

    _method_and_epsilon_legends(fig, budgets, eps_grid)
    fig.tight_layout(rect=(0.02, 0.14, 0.995, 0.99), w_pad=2.0, h_pad=2.1)
    fig.savefig(artifact_dir / "fig_pareto_cost_recall_all_datasets.pdf", bbox_inches="tight")
    fig.savefig(
        artifact_dir / "fig_pareto_cost_recall_all_datasets.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    eps_grid = parse_float_list(args.eps)
    budgets = parse_int_list(args.budgets)
    seeds = parse_int_list(args.seeds)

    unsupported_budgets = sorted(set(budgets).difference(BUDGET_COLORS))
    unsupported_eps = sorted(round(x, 1) for x in set(eps_grid) if round(x, 1) not in EPS_MARKERS)
    if unsupported_budgets:
        raise ValueError(f"Add plotting colors for budgets: {unsupported_budgets}")
    if unsupported_eps:
        raise ValueError(f"Add plotting markers for eps values: {unsupported_eps}")

    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    run_rows: list[dict] = []
    baseline_rows: list[dict] = []

    for spec in DATASET_SPECS:
        cfg = load_main_config(args.model, spec)
        pool = load_generation_pool(cfg)
        main_files = locate_main_run_files(cfg)
        edit_rep_main = main_files.get("EDIT-REP")
        llm_select_main = main_files.get("LLM-Select")
        if edit_rep_main is None or llm_select_main is None:
            raise FileNotFoundError(
                f"Need existing EDIT-REP and LLM-Select main runs for {spec.label}."
            )

        features = load_or_build_minilm_features(
            cfg, spec, pool, args.feature_cache_dir, args.model
        )

        streams: dict[int, np.ndarray] = {}
        for seed in seeds:
            if seed == 7:

                streams[seed] = stream_from_existing_run(edit_rep_main, pool)
            else:
                streams[seed] = sample_stream(
                    len(pool), seed, args.num_batches, args.batch_size
                )

        gamma = float(cfg["policy"]["alpha_step_size"])


        total = len(seeds) * len(eps_grid) * len(budgets)
        step = 0
        for seed in seeds:
            for epsilon in eps_grid:
                for budget in budgets:
                    step += 1
                    print(
                        f"[{spec.label}] {step}/{total} seed={seed} "
                        f"epsilon={epsilon} B={budget} gamma={gamma}"
                    )
                    run = run_edit_rep_offline(
                        cfg,
                        pool,
                        features,
                        streams[seed],
                        seed=seed,
                        epsilon=epsilon,
                        budget=budget,
                    )
                    metrics = compute_metrics(run)
                    metrics.update(
                        {
                            "Model": MODEL_LABELS[args.model],
                            "Dataset": spec.label,
                            "seed": seed,
                            "epsilon": epsilon,
                            "budget": budget,
                        }
                    )
                    run_rows.append(metrics)

        baseline_metrics = compute_metrics(pd.read_csv(llm_select_main))
        baseline_metrics.update(
            {
                "Model": MODEL_LABELS[args.model],
                "Dataset": spec.label,
                "Method": "LLM-Select",
                "budget": 5,
                "source_csv": str(llm_select_main),
            }
        )
        baseline_rows.append(baseline_metrics)

    runs = pd.DataFrame(run_rows)
    runs.to_csv(artifact_dir / "pareto_runs_all_seeds.csv", index=False)

    summary = summarize_mean_ci(
        runs,
        ["Model", "Dataset", "epsilon", "budget"],
        ["type_i", "type_ii", "edit_recall", "mean_review_cost"],
    )
    summary.to_csv(artifact_dir / "pareto_grid_summary.csv", index=False)

    baselines = pd.DataFrame(baseline_rows)
    baselines.to_csv(artifact_dir / "llm_select_reference_points.csv", index=False)

    plot_type_i_type_ii(summary, baselines, budgets, eps_grid, artifact_dir)
    plot_cost_recall(summary, baselines, budgets, eps_grid, artifact_dir)
    print(f"\nWrote sweep artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
