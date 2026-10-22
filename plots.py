import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from src.analysis import DATASETS, METHODS, MODELS, config, load_run, running_metrics
from src.experiment import cost_fn
from src.utils import output_paths

TEXT, TICK = 20, 18
METRICS = [("Type-II", "Type-II error"), ("Type-I", "Type-I error"), ("Cost", "Cost")]
METHOD_COLORS = {"EDIT-REP": "#ff7f0e", "EDIT-LLM": "#1f77b4", "EDIT-RAND": "#9467bd", "LLM-Select": "#d62728"}
ENCODER_COLORS = {"Qwen3-0.6B": "#6BAED6", "Qwen3-1.7B": "#2171B5", "Qwen3-4B": "#08306B"}
SHIFT_PHASES = {
    "mmlupro_shift": ("MMLU-Pro", [(0, 49, "STEM"), (50, 99, "Life / Health"), (100, 149, "Humanities / Social Sci."), (150, 199, "Law / Others")]),
    "popqa_shift": ("PopQA", [(0, 59, "Head"), (60, 119, "Middle"), (120, 179, "Long-tail")]),
}
PHASE_COLORS = ["#F6E8C3", "#D8E8F1", "#E6D8F2", "#DDEED6", "#F3D6D8"]


def set_style(context="paper"):
    sns.set_theme(context=context, style="darkgrid", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.size": TEXT, "axes.titlesize": TEXT, "axes.labelsize": TEXT, "xtick.labelsize": TICK,
            "ytick.labelsize": TICK, "legend.fontsize": TEXT, "legend.title_fontsize": TEXT, "figure.titlesize": TEXT,
            "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold",
        }
    )
    if context == "paper":
        plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def set_limits(ax, curves, col, refs):
    df = pd.concat(curves, ignore_index=True)
    ax.set_xlim(max(1.0, float(df["Round"].min())), float(df["Round"].max()))
    y = df[col].dropna()
    late = y[df.loc[y.index, "Round"] > 6] if len(y) else y
    values = (late if len(late) >= 8 else y).astype(float).tolist()
    if values:
        lo, hi = min(values), max(values)
        pad = 0.35 * max(hi - lo, abs(lo), abs(hi), 1e-12)
        values += [r for r in refs if lo - pad <= r <= hi + pad]
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else max(abs(lo), abs(hi), 1.0)
    ax.set_ylim(lo - 0.08 * span, hi + 0.08 * span)


def draw_panel(ax, curves, col, colors, linewidth, ref_style):
    df = pd.concat(curves.values())
    for label, curve in curves.items():
        ax.plot(curve["Round"], curve[col], color=colors[label], linewidth=linewidth, label=label, zorder=5)
    refs = []
    if col == "Type-I" and df["epsilon"].notna().any():
        refs.append(float(df["epsilon"].dropna().min()))
    if col == "Cost":
        refs.append(float(df["budget"].min()))
    for r in refs:
        ax.axhline(r, **ref_style)
    set_limits(ax, list(curves.values()), col, refs)


def add_legend(fig, colors, labels, y, ncol, **kwargs):
    handles = [Line2D([0], [0], color=colors[l], linewidth=2.6) for l in labels]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, prop={"size": TEXT, "weight": "bold"}, **kwargs)


def save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_dir / name}.png")


def grid(panels, colors, out_dir, name, linewidth, spines=True, ncol=6, columnspacing=1.6):
    fig, axes = plt.subplots(3, 4, figsize=(20.0, 11.0))
    labels = []
    for c, (title, curves) in enumerate(panels):
        labels += [l for l in curves if l not in labels]
        for r, (col, ylabel) in enumerate(METRICS):
            ax = axes[r][c]
            draw_panel(ax, curves, col, colors, linewidth, {"linestyle": "--", "linewidth": 1.4, "color": "black"})
            ax.set_title(title if r == 0 else "", fontsize=TEXT, fontweight="bold", pad=12)
            ax.set_ylabel(ylabel if c == 0 else "", fontsize=TEXT, fontweight="bold")
            ax.set_xlabel("Round" if r == 2 else "", fontsize=TEXT, fontweight="bold")
            ax.tick_params(axis="both", labelsize=TICK, labelbottom=r == 2)
            ax.grid(True, alpha=0.35, linewidth=0.8)
            if not spines:
                ax.spines[["top", "right"]].set_visible(False)
    labels = [l for l in colors if l in labels]
    add_legend(fig, colors, labels, 0.005, min(len(labels), ncol), handlelength=2.8, columnspacing=columnspacing)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.98), w_pad=1.4, h_pad=1.2)
    save(fig, out_dir, name)


def method_curves(model, dataset, methods=METHODS):
    curves = {}
    for name, label in methods.items():
        run = load_run(model, dataset, name)
        if run is not None:
            curves[label] = running_metrics(run)
    return curves


def model_grid(model, out_dir):
    set_style("talk")
    panels = [(label, method_curves(model, key)) for key, label in DATASETS.items()]
    grid(panels, METHOD_COLORS, out_dir, f"model_grid_{model}", linewidth=2.6)


def encoder_grid(out_dir):
    set_style()
    encoders = {"edit_rep_qwen3-0.6b": "Qwen3-0.6B", "edit_rep_qwen3-1.7b": "Qwen3-1.7B", "edit_rep_qwen3-4b": "Qwen3-4B"}
    panels = [(label, method_curves("llama", key, encoders)) for key, label in DATASETS.items()]
    grid(panels, ENCODER_COLORS, out_dir, "encoder_size_grid", linewidth=2.8, spines=False, ncol=3, columnspacing=1.8)


def review_length_grid(out_dir):
    set_style()
    datasets = {"math500_review_length": "MATH-500", "gpqa_review_length": "GPQA"}
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 7.8))
    for r, (key, label) in enumerate(datasets.items()):
        curves = method_curves("llama", key)
        for c, (col, title) in enumerate(METRICS):
            ax = axes[r][c]
            draw_panel(ax, curves, col, METHOD_COLORS, 2.8, {"linestyle": "--", "linewidth": 1.4, "color": "black"})
            ax.set_title(title if r == 0 else "", fontsize=TEXT, fontweight="bold", pad=12)
            ax.set_ylabel(label if c == 0 else "", fontsize=TEXT, fontweight="bold", labelpad=16)
            ax.set_xlabel("Round" if r == 1 else "", fontsize=TEXT, fontweight="bold")
            ax.tick_params(axis="both", labelsize=TICK, labelbottom=r == 1)
            ax.grid(True, alpha=0.35, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    labels = [l for l in METHOD_COLORS if l in curves]
    add_legend(fig, METHOD_COLORS, labels, 0.005, min(len(labels), 6), handlelength=2.8, columnspacing=1.6)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.96), w_pad=1.6, h_pad=1.3)
    save(fig, out_dir, "review_length_grid")


def shift_figure(out_dir):
    set_style()
    fig, axes = plt.subplots(3, 2, figsize=(14.8, 9.0))
    for c, (key, (title, phases)) in enumerate(SHIFT_PHASES.items()):
        curves = method_curves("llama", key)
        for r, (col, ylabel) in enumerate(METRICS):
            ax = axes[r][c]
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            for i, (start, end, label) in enumerate(phases):
                ax.axvspan(start + 1, end + 1, color=PHASE_COLORS[i % len(PHASE_COLORS)], alpha=0.26, lw=0, zorder=0)
                if r == 0:
                    ax.text(
                        (start + end + 2) / 2.0, 1.055, label, ha="center", va="bottom", transform=trans, fontsize=11,
                        fontweight="bold", clip_on=False, zorder=7,
                        bbox={"boxstyle": "round,pad=0.22,rounding_size=0.12", "fc": "white", "ec": "0.35", "lw": 0.65, "alpha": 0.96},
                    )
            for start, _, _ in phases[1:]:
                ax.axvspan(start - 0.2, start + 1.2, color="black", alpha=0.045, lw=0, zorder=1)
                ax.axvline(start + 0.5, color="0.12", lw=0.9, linestyle=(0, (1.2, 2.2)), zorder=4)
            ax.grid(True, alpha=0.25, linewidth=0.75)
            draw_panel(ax, curves, col, METHOD_COLORS, 2.8, {"color": "0.10", "lw": 1.15, "linestyle": (0, (4, 2)), "zorder": 4})
            ax.set_xlim(phases[0][0] + 1, phases[-1][1] + 1)
            ax.set_title(title if r == 0 else "", fontsize=TEXT, fontweight="bold", pad=44)
            ax.set_ylabel(ylabel if c == 0 else "", fontsize=TEXT, fontweight="bold")
            ax.set_xlabel("Round" if r == 2 else "", fontsize=TEXT, fontweight="bold")
            ax.tick_params(axis="both", labelsize=TICK, labelbottom=r == 2)
            ax.spines[["top", "right"]].set_visible(False)
    labels = [l for l in METHOD_COLORS if l in curves]
    add_legend(fig, METHOD_COLORS, labels, -0.005, min(len(labels), 5), handlelength=2.6, columnspacing=1.4)
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.94), w_pad=1.6, h_pad=1.25)
    fig.canvas.draw()
    for c, (_, phases) in enumerate(SHIFT_PHASES.values()):
        top, bottom = axes[0][c], axes[-1][c]
        for start, _, _ in phases[1:]:
            x = fig.transFigure.inverted().transform(top.transData.transform((start + 0.5, 0)))[0]
            fig.add_artist(
                Line2D([x, x], [bottom.get_position().y0, top.get_position().y1], transform=fig.transFigure, color="0.10",
                       linewidth=1.2, linestyle=(0, (1.2, 2.2)), alpha=0.78, zorder=20, clip_on=False)
            )
    save(fig, out_dir, "distribution_shift")


def cost_distribution(out_dir):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for ax, (label, keys) in zip(axes, {"MATH-500": ["math500"], "PopQA": ["popqa", "popqa_shift"]}.items()):
        frames = []
        for key in keys:
            path = output_paths(config("llama", key, "random"))["generations"]
            if path.exists():
                gen = pd.read_csv(path)
                gen["cost"] = gen["model_answer"].map(cost_fn(gen, "review_length"))
                frames.append(gen)
        d = pd.concat(frames).drop_duplicates(["example_id", "question", "model_answer"])
        d["Model"] = MODELS["llama"]
        sns.violinplot(data=d, x="cost", y="Model", inner=None, cut=0, linewidth=1.5, saturation=0.85, color=sns.color_palette("deep")[0], ax=ax)
        sns.boxplot(
            data=d, x="cost", y="Model", width=0.22, showfliers=False, showcaps=True, ax=ax,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.25, "alpha": 0.88},
            medianprops={"color": "black", "linewidth": 2.0}, whiskerprops={"color": "black", "linewidth": 1.1}, capprops={"color": "black", "linewidth": 1.1},
        )
        sns.stripplot(data=d, x="cost", y="Model", color="black", alpha=0.16, size=2.0, jitter=0.20, ax=ax)
        ax.axvline(1.0, linestyle="--", linewidth=1.4, color="black", alpha=0.85)
        lo, hi = np.percentile(d["cost"], [1.0, 99.0])
        lo, hi = min(lo, 1.0), max(hi, 1.0)
        margin = 0.12 * (hi - lo if hi > lo else max(abs(lo), abs(hi), 1.0))
        ax.set_xlim(max(0.0, lo - margin), min(4.05, hi + margin))
        ax.set_title(label, fontsize=TEXT, fontweight="bold", pad=13)
        ax.set_xlabel("Review-length cost", fontsize=TEXT, fontweight="bold")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=TICK)
        ax.text(0.98, 0.96, f"n={len(d):,}", transform=ax.transAxes, ha="right", va="top", fontsize=16, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.2})
        ax.grid(True, axis="x", alpha=0.32, linewidth=0.8)
        ax.grid(False, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    save(fig, out_dir, "review_length_cost_distribution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    for model in MODELS:
        model_grid(model, args.out_dir)
    encoder_grid(args.out_dir)
    review_length_grid(args.out_dir)
    shift_figure(args.out_dir)
    cost_distribution(args.out_dir)
