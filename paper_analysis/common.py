from __future__ import annotations

import copy
import gc
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from methods.ours import OursSelection
from models.minilm import MiniLMScoreModel
from utils import generation_cache_name, load_yaml, model_data_name_from_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    suffix: str


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("math500", "MATH-500", "math500-test"),
    DatasetSpec("mmlupro", "MMLU-Pro", "mmlupro-test_n500_seed42"),
    DatasetSpec("popqa", "PopQA", "popqa-test_n500_seed42"),
    DatasetSpec("gpqa", "GPQA", "gpqa-train_main_n448"),
)

DATASET_ORDER = tuple(spec.label for spec in DATASET_SPECS)

MODEL_CONFIG_PREFIXES: Mapping[str, str] = {
    "llama": "llama-3.3-70b-versatile",
    "qwen": "qwen-qwen3-32b",
}

MODEL_LABELS: Mapping[str, str] = {
    "llama": "Llama-3.3-70B",
    "qwen": "Qwen3-32B",
}

METHOD_ORDER = ("EDIT-REP", "EDIT-LLM", "EDIT-RAND", "LLM-Select")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_path(model_key: str, spec: DatasetSpec) -> Path:
    prefix = MODEL_CONFIG_PREFIXES[model_key]
    return PROJECT_ROOT / "configs" / f"{prefix}_{spec.suffix}" / "ours.yaml"


def load_main_config(model_key: str, spec: DatasetSpec) -> dict:
    path = config_path(model_key, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing main config: {path}")
    return load_yaml(path)


def output_dir_from_config(cfg: dict) -> Path:
    output_root = repo_path(cfg.get("output_dir", "outputs"))
    return output_root / model_data_name_from_config(cfg)


def generation_cache_from_config(cfg: dict) -> Path:
    return output_dir_from_config(cfg) / generation_cache_name(cfg)


def _best_file(paths: Iterable[Path]) -> Path | None:
    paths = list(paths)
    if not paths:
        return None
    return max(paths, key=lambda p: (p.stat().st_size, p.stat().st_mtime_ns))


def locate_main_run_files(cfg: dict) -> Dict[str, Path]:

    out_dir = output_dir_from_config(cfg)
    if not out_dir.exists():
        raise FileNotFoundError(f"Missing output directory: {out_dir}")
    files = list(out_dir.glob("*.csv"))

    edit_rep = _best_file(
        p
        for p in files
        if p.name.startswith("ours_")
        and "sentence-transformers-all-MiniLM-L6-v2" in p.name
        and not p.name.startswith("ours_llm_")
        and not p.name.startswith("ours_random_")
    )
    edit_llm = _best_file(p for p in files if p.name.startswith("ours_llm_"))
    edit_rand = _best_file(p for p in files if p.name.startswith("ours_random_"))
    llm_select = _best_file(p for p in files if p.name.startswith("llm_select_"))

    found = {
        "EDIT-REP": edit_rep,
        "EDIT-LLM": edit_llm,
        "EDIT-RAND": edit_rand,
        "LLM-Select": llm_select,
    }
    return {k: v for k, v in found.items() if v is not None}


def safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num / den)


def compute_metrics(run_df: pd.DataFrame) -> dict[str, float]:

    required = {"t", "selected", "A", "cost"}
    missing = required.difference(run_df.columns)
    if missing:
        raise ValueError(f"Run dataframe is missing columns: {sorted(missing)}")

    d = run_df.copy()
    d["A"] = pd.to_numeric(d["A"], errors="raise").astype(float)
    d["selected"] = pd.to_numeric(d["selected"], errors="raise").astype(float)
    d["cost"] = pd.to_numeric(d["cost"], errors="raise").astype(float)

    round_type_i: list[float] = []
    round_type_ii: list[float] = []
    round_cost: list[float] = []

    for _, g in d.groupby("t", sort=True):
        s = g["selected"].to_numpy(dtype=float)
        a = g["A"].to_numpy(dtype=float)
        c = g["cost"].to_numpy(dtype=float)
        round_type_i.append(safe_div(float((s * (1.0 - a)).sum()), float(s.sum())))
        round_type_ii.append(
            safe_div(float(((1.0 - s) * a).sum()), float((1.0 - s).sum()))
        )
        round_cost.append(float((s * c).sum()))

    s = d["selected"].to_numpy(dtype=float)
    a = d["A"].to_numpy(dtype=float)
    c = d["cost"].to_numpy(dtype=float)

    all_edits = float(a.sum())
    selected_edits = float((s * a).sum())
    missed_edits = float(((1.0 - s) * a).sum())
    selected_count = float(s.sum())
    total_review_cost = float((s * c).sum())

    return {
        "type_i": float(np.mean(round_type_i)),
        "type_ii": float(np.mean(round_type_ii)),
        "mean_review_cost": float(np.mean(round_cost)),
        "edit_recall": safe_div(selected_edits, all_edits),
        "absolute_missed_edits": missed_edits,
        "edit_precision": safe_div(selected_edits, selected_count),
        "selected_count": selected_count,
        "selected_edits": selected_edits,
        "all_edits": all_edits,
        "total_review_cost": total_review_cost,
        "review_rate": safe_div(selected_count, float(len(d))),
    }


def mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    mean = float(arr.mean())
    if len(arr) == 1:
        return (mean, 0.0)
    ci = 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return (mean, ci)


def summarize_mean_ci(
    df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in df.groupby(list(group_cols), sort=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(len(group))
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            mean, ci = mean_ci95(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def load_generation_pool(cfg: dict) -> pd.DataFrame:
    path = generation_cache_from_config(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing generation cache: {path}. The paper analysis never calls the main-model API."
        )
    df = pd.read_csv(path).drop_duplicates("example_id", keep="last").reset_index(drop=True)
    required = {"example_id", "question", "model_answer", "A"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Generation cache {path} is missing: {sorted(missing)}")
    return df


def stream_from_existing_run(run_csv: Path, generation_pool: pd.DataFrame) -> np.ndarray:

    run = pd.read_csv(run_csv).sort_values(["t", "batch_pos"]).reset_index(drop=True)
    lookup = {
        int(example_id): i
        for i, example_id in enumerate(generation_pool["example_id"].astype(int).tolist())
    }
    counts = run.groupby("t").size().to_numpy()
    if len(counts) == 0 or len(set(counts.tolist())) != 1:
        raise ValueError(f"Expected a constant batch size in {run_csv}")
    n_rounds = int(run["t"].max()) + 1
    batch_size = int(counts[0])
    stream = np.empty((n_rounds, batch_size), dtype=int)
    for t, group in run.groupby("t", sort=True):
        group = group.sort_values("batch_pos")
        try:
            stream[int(t)] = np.asarray(
                [lookup[int(x)] for x in group["example_id"].tolist()], dtype=int
            )
        except KeyError as exc:
            raise KeyError(
                f"example_id {exc.args[0]} from {run_csv} is absent from generation cache"
            ) from exc
    return stream


def sample_stream(pool_size: int, seed: int, num_batches: int, batch_size: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.choice(
        int(pool_size), size=(int(num_batches), int(batch_size)), replace=True
    )


class PrecomputedScoreModel:


    def __init__(self, features: np.ndarray):
        self.features = np.asarray(features, dtype=np.float32)

    def encode_rows(self, rows: list[dict]) -> np.ndarray:
        idx = np.fromiter((int(row["pool_pos"]) for row in rows), dtype=int)
        return self.features[idx]


def feature_cache_path(cache_dir: Path, model_key: str, spec: DatasetSpec) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{model_key}_{spec.suffix}")
    return cache_dir / f"{safe}_minilm_features.npz"


def load_or_build_minilm_features(
    cfg: dict,
    spec: DatasetSpec,
    generation_pool: pd.DataFrame,
    cache_dir: Path,
    model_key: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = feature_cache_path(cache_dir, model_key, spec)
    example_ids = generation_pool["example_id"].astype(int).to_numpy()

    if cache_path.exists():
        obj = np.load(cache_path)
        cached_ids = obj["example_id"].astype(int)
        features = obj["features"].astype(np.float32)
        if np.array_equal(cached_ids, example_ids) and len(features) == len(generation_pool):
            print(f"[feature-cache] loaded {spec.label}: {features.shape}")
            return features

    score_cfg = copy.deepcopy(cfg["score_model"])
    score_cfg["dataset_name"] = cfg["data"]["name"]
    score_cfg["encode_batch_size"] = max(64, int(score_cfg.get("encode_batch_size", 64)))

    encoder = MiniLMScoreModel(score_cfg)
    rows = generation_pool[["question", "model_answer"]].to_dict("records")
    features = encoder.encode_rows(rows).astype(np.float32)
    np.savez_compressed(cache_path, example_id=example_ids, features=features)
    print(f"[feature-cache] wrote {cache_path}: {features.shape}")

    del encoder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return features


def run_edit_rep_offline(
    cfg: dict,
    generation_pool: pd.DataFrame,
    features: np.ndarray,
    stream_indices: np.ndarray,
    *,
    seed: int,
    epsilon: float,
    budget: float,
) -> pd.DataFrame:

    local_cfg = copy.deepcopy(cfg)
    local_cfg["seed"] = int(seed)
    local_cfg["policy"]["epsilon"] = float(epsilon)
    local_cfg["policy"]["budget_per_batch"] = float(budget)
    local_cfg["data"]["num_batches"] = int(stream_indices.shape[0])
    local_cfg["data"]["batch_size"] = int(stream_indices.shape[1])

    score_model = PrecomputedScoreModel(features)
    method = OursSelection(local_cfg, score_model=score_model, state=None)

    chunks: list[pd.DataFrame] = []
    for t in range(stream_indices.shape[0]):
        pool_idx = stream_indices[t].astype(int)
        group = generation_pool.iloc[pool_idx]
        batch = pd.DataFrame(
            {
                "t": int(t),
                "batch_pos": np.arange(len(pool_idx), dtype=int),
                "pool_pos": pool_idx,
                "example_id": group["example_id"].astype(int).to_numpy(),
                "A": group["A"].astype(int).to_numpy(),

                "cost": np.ones(len(pool_idx), dtype=float),
            }
        )
        out, _ = method.process_batch(batch, t=t)
        chunks.append(out)
    return pd.concat(chunks, ignore_index=True)
