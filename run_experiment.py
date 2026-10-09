import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from data_wrappers import build_data_wrapper
from models import build_main_llm, build_score_model
from methods import build_method
from methods.costs import compute_cost, prepare_cost_context
from utils import load_json, load_yaml, project_paths, read_csv_or_empty, save_json_atomic, write_csv_atomic

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*_check_is_size will be removed in a future PyTorch release.*",
    category=FutureWarning,
    module=r"bitsandbytes.*",
)


GEN_COLUMNS = [
    "example_id",
    "question",
    "gold_answer",
    "gold_final",
    "model_answer",
    "pred_answer",
    "A",
]

RUN_COLUMNS = [
    "config",
    "method",
    "t",
    "batch_pos",
    "example_id",
    "question",
    "gold_answer",
    "gold_final",
    "model_answer",
    "pred_answer",
    "A",
    "cost",
    "eta",
    "alpha",
    "beta",
    "selected",
    "budget",
    "epsilon",
    "main_llm_provider",
    "main_llm",
    "score_model_provider",
    "score_model",
    "selector_llm_provider",
    "selector_llm",
]


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num / den)


def _normalize_filter_value(value: Any) -> str:

    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _record_matches_phase_filter(
    rec: Dict[str, Any],
    phase_filter: Dict[str, Any],
    popularity_bounds: tuple[float, float] | None = None,
) -> bool:

    if not phase_filter:
        return True

    if "category_in" in phase_filter:
        allowed = {_normalize_filter_value(v) for v in phase_filter.get("category_in", [])}
        category = _normalize_filter_value(rec.get("category") or rec.get("subject") or rec.get("domain"))
        if category not in allowed:
            return False

    if "popularity_quantile" in phase_filter:
        if popularity_bounds is None:
            raise ValueError("Internal error: popularity bounds were not computed for popularity_quantile filter.")
        lo, hi = popularity_bounds
        try:
            popularity = float(rec.get("popularity"))
        except Exception:
            return False

        if not (lo <= popularity <= hi):
            return False

    return True


def _popularity_bounds(records: List[Dict[str, Any]], qlo: float, qhi: float) -> tuple[float, float]:
    values = []
    for rec in records:
        try:
            val = float(rec.get("popularity"))
        except Exception:
            continue
        if np.isfinite(val):
            values.append(val)
    if not values:
        raise ValueError("data.shift_stream uses popularity_quantile, but records have no finite popularity metadata.")
    values_arr = np.asarray(values, dtype=float)
    qlo = float(qlo)
    qhi = float(qhi)
    if not (0.0 <= qlo < qhi <= 1.0):
        raise ValueError("popularity_quantile must be [lo, hi] with 0 <= lo < hi <= 1.")
    return (float(np.quantile(values_arr, qlo)), float(np.quantile(values_arr, qhi)))


def sample_shift_batches(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[List[Dict[str, Any]]]:

    data_cfg = cfg["data"]
    stream_cfg = data_cfg.get("shift_stream") or {}
    phases = stream_cfg.get("phases") or []
    if not phases:
        raise ValueError("data.shift_stream.phases must contain at least one phase.")

    default_batch_size = int(data_cfg.get("batch_size", 10))
    if default_batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")

    expected_num_batches = data_cfg.get("num_batches")
    phase_total = sum(int(phase.get("num_batches", 0)) for phase in phases)
    if phase_total <= 0:
        raise ValueError("Each shift_stream phase must set a positive num_batches.")
    if expected_num_batches is not None and int(expected_num_batches) != int(phase_total):
        raise ValueError(
            f"data.num_batches={expected_num_batches} does not match the shift-stream "
            f"phase total {phase_total}. Keep them equal to avoid ambiguous output lengths."
        )

    seed = int(data_cfg.get("sample_seed", cfg.get("seed", 0)))
    batches: List[List[Dict[str, Any]]] = []

    for phase_idx, phase in enumerate(phases):
        phase_name = str(phase.get("name", f"phase_{phase_idx + 1}"))
        phase_filter = dict(phase.get("filter") or {})
        popularity_bounds = None
        if "popularity_quantile" in phase_filter:
            qlo, qhi = phase_filter["popularity_quantile"]
            popularity_bounds = _popularity_bounds(records, qlo, qhi)

        phase_records = [
            rec
            for rec in records
            if _record_matches_phase_filter(rec, phase_filter, popularity_bounds=popularity_bounds)
        ]

        if not phase_records:
            categories = sorted(
                {
                    _normalize_filter_value(r.get("category") or r.get("subject") or r.get("domain"))
                    for r in records
                    if _normalize_filter_value(r.get("category") or r.get("subject") or r.get("domain"))
                }
            )
            raise ValueError(
                f"Shift-stream phase {phase_name!r} matched zero records. "
                f"Available normalized categories: {categories[:50]}"
            )

        num_batches = int(phase.get("num_batches", 0))
        batch_size = int(phase.get("batch_size", default_batch_size))
        if num_batches <= 0:
            raise ValueError(f"Shift-stream phase {phase_name!r} has non-positive num_batches.")
        if batch_size <= 0:
            raise ValueError(f"Shift-stream phase {phase_name!r} has non-positive batch_size.")

        rng = np.random.default_rng(seed + 104729 * phase_idx)
        indices = rng.choice(len(phase_records), size=(num_batches, batch_size), replace=True)
        batches.extend([[phase_records[int(i)] for i in row] for row in indices])

        print(
            f"[shift] phase={phase_idx + 1}/{len(phases)} name={phase_name!r} "
            f"pool={len(phase_records)} batches={num_batches} batch_size={batch_size}"
        )

    return batches


def sample_batches(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[List[Dict[str, Any]]]:

    if not records:
        raise RuntimeError("Dataset wrapper returned no records.")

    data_cfg = cfg["data"]
    stream_cfg = data_cfg.get("shift_stream") or {}
    if stream_cfg.get("enabled", False) or stream_cfg.get("phases"):
        return sample_shift_batches(records, cfg)

    batch_size = int(data_cfg.get("batch_size", 10))
    num_batches = int(data_cfg.get("num_batches", 200))
    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")
    if num_batches <= 0:
        raise ValueError("data.num_batches must be positive.")

    seed = int(data_cfg.get("sample_seed", cfg.get("seed", 0)))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=(num_batches, batch_size), replace=True)
    return [[records[int(i)] for i in row] for row in indices]


def flatten_batches(batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [rec for batch in batches for rec in batch]


def print_run_summary(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        print("[summary] no rows available")
        return

    A = df["A"].astype(float)
    selected = df["selected"].astype(float)
    cost = df["cost"].astype(float)
    budget = df["budget"].astype(float)

    n = len(df)
    n_selected = int(selected.sum())
    n_unselected = int(n - n_selected)

    model_correct = int((1.0 - A).sum())
    model_accuracy = _safe_div(model_correct, n)

    type_i = _safe_div(float((selected * (1.0 - A)).sum()), float(selected.sum()))
    type_ii = _safe_div(float(((1.0 - selected) * A).sum()), float((1.0 - selected).sum()))

    spent_total = float((selected * cost).sum())
    budget_total = float(budget.groupby(df["t"]).first().sum()) if "t" in df else float(budget.sum())
    budget_used = _safe_div(spent_total, budget_total)

    print("\n[summary]")
    print(f"  rows              : {n}")
    print(f"  unique examples   : {df['example_id'].nunique()}")
    print(f"  model accuracy    : {model_accuracy:.4f} ({model_correct}/{n})")
    print(f"  selected          : {n_selected}")
    print(f"  unselected        : {n_unselected}")
    print(f"  type-I            : {type_i:.4f}")
    print(f"  type-II           : {type_ii:.4f}")
    print(f"  budget used       : {spent_total:.2f}/{budget_total:.2f} ({100.0 * budget_used:.1f}%)")


def apply_dataset_llm_defaults(cfg: Dict[str, Any], data_wrapper) -> None:

    main_cfg = cfg.get("main_llm")
    if isinstance(main_cfg, dict) and not main_cfg.get("system_prompt"):
        prompt_getter = getattr(data_wrapper, "main_system_prompt", None)
        if callable(prompt_getter):
            system_prompt = prompt_getter()
            if system_prompt:
                main_cfg["system_prompt"] = system_prompt


def ensure_generations(
    records: List[Dict[str, Any]],
    data_wrapper,
    main_llm,
    cache_path: Path,
    allow_generate: bool = True,
) -> pd.DataFrame:

    cache = read_csv_or_empty(cache_path, GEN_COLUMNS)
    if len(cache):
        cache = cache.drop_duplicates("example_id", keep="last")

    target_ids = {int(r["example_id"]) for r in records}
    done_ids = set(cache["example_id"].astype(int).tolist()) if len(cache) else set()
    done_target_ids = done_ids.intersection(target_ids)

    missing_by_id: Dict[int, Dict[str, Any]] = {}
    for r in records:
        ex_id = int(r["example_id"])
        if ex_id not in done_ids and ex_id not in missing_by_id:
            missing_by_id[ex_id] = r
    missing = list(missing_by_id.values())

    cache_relevant = cache[cache["example_id"].astype(int).isin(target_ids)] if len(cache) else cache
    cached_seen = len(cache_relevant)
    cached_correct = int((1 - cache_relevant["A"].astype(int)).sum()) if len(cache_relevant) else 0
    cached_acc = _safe_div(cached_correct, cached_seen)

    print(f"[cache] generation cache: {cache_path}")
    print(
        f"[cache] cached_total={len(cache)} loaded_unique_for_this_run={len(done_target_ids)} "
        f"missing_unique={len(missing)} requested_rows={len(records)} requested_unique={len(target_ids)}"
    )
    if cached_seen:
        print(f"[cache] cached model accuracy on requested unique rows={cached_acc:.4f} ({cached_correct}/{cached_seen})")

    if missing and not allow_generate:
        raise RuntimeError(
            f"Generation cache is incomplete: {len(missing)} missing unique examples. "
            f"Run `python generate_cache.py --config <config>` first, or rerun without `--no_generate`."
        )

    rows = cache.to_dict("records") if len(cache) else []
    if not missing:
        return cache

    if main_llm is None:
        raise RuntimeError("main_llm is required because generation cache has missing examples.")

    seen = cached_seen
    correct = cached_correct
    pbar = tqdm(missing, desc="main LLM generations", dynamic_ncols=True)
    for rec in pbar:
        prompt = data_wrapper.build_prompt(rec["question"])
        model_answer = main_llm.generate(prompt)
        pred_answer = data_wrapper.parse_prediction(model_answer)
        A = data_wrapper.failure_label(pred_answer, rec["gold_final"])
        seen += 1
        correct += int(A == 0)
        running_acc = _safe_div(correct, seen)
        pbar.set_postfix(acc=f"{running_acc:.3f}", correct=f"{correct}/{seen}")
        rows.append(
            {
                "example_id": rec["example_id"],
                "question": rec["question"],
                "gold_answer": rec["gold_answer"],
                "gold_final": rec["gold_final"],
                "model_answer": model_answer,
                "pred_answer": pred_answer,
                "A": int(A),
            }
        )
        write_csv_atomic(pd.DataFrame(rows, columns=GEN_COLUMNS).drop_duplicates("example_id", keep="last"), cache_path)

    return pd.DataFrame(rows, columns=GEN_COLUMNS).drop_duplicates("example_id", keep="last")


def make_batch_rows(
    cfg: Dict[str, Any],
    gen_by_id: pd.DataFrame,
    records: List[Dict[str, Any]],
    t: int,
) -> pd.DataFrame:
    rows = []
    policy = cfg["policy"]
    cost_variant = policy.get("cost_variant", "constant")
    cost_context = prepare_cost_context(gen_by_id, cost_variant)
    score_cfg = cfg.get("score_model", {}) or {}
    selector_cfg = cfg.get("selector_llm", {}) or cfg.get("score_llm", {}) or {}

    for pos, rec in enumerate(records):
        cached = gen_by_id.loc[int(rec["example_id"])].to_dict()
        row = {
            "config": cfg.get("run_name", cfg["method"]),
            "method": cfg["method"],
            "t": int(t),
            "batch_pos": int(pos),
            "example_id": int(rec["example_id"]),
            "question": cached["question"],
            "gold_answer": cached["gold_answer"],
            "gold_final": cached["gold_final"],
            "model_answer": cached["model_answer"],
            "pred_answer": cached["pred_answer"],
            "A": int(cached["A"]),
            "cost": 1.0,
            "eta": float("nan"),
            "alpha": float("nan"),
            "beta": float("nan"),
            "selected": 0,
            "budget": float(policy["budget_per_batch"]),
            "epsilon": float(policy.get("epsilon", "nan")),
            "main_llm_provider": cfg["main_llm"].get("provider"),
            "main_llm": cfg["main_llm"].get("model_name"),
            "score_model_provider": score_cfg.get("provider", "none"),
            "score_model": score_cfg.get("model_name", "none"),
            "selector_llm_provider": selector_cfg.get("provider", "none"),
            "selector_llm": selector_cfg.get("model_name", "none"),
        }
        row["cost"] = compute_cost(row, cost_variant, cost_context)
        rows.append(row)
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def run(cfg_path: str, reset: bool = False, reset_generations: bool = False, no_generate: bool = False) -> None:
    load_dotenv()
    cfg = load_yaml(cfg_path)
    paths = project_paths(cfg)
    cfg["run_name"] = paths["run_name"]

    if reset:
        for p in [paths["run_csv"], paths["state_json"]]:
            if p.exists():
                p.unlink()
    if reset_generations and paths["generation_cache"].exists():
        paths["generation_cache"].unlink()

    print(f"[run] name={cfg['run_name']} method={cfg['method']}")
    print(f"[run] run csv={paths['run_csv']}")

    data_wrapper = build_data_wrapper(cfg["data"])
    apply_dataset_llm_defaults(cfg, data_wrapper)
    records_pool = data_wrapper.load_records()
    batches = sample_batches(records_pool, cfg)
    sampled_records = flatten_batches(batches)

    main_llm = None
    if not no_generate:
        main_llm = build_main_llm(cfg["main_llm"])
    gen_cache = ensure_generations(
        sampled_records,
        data_wrapper,
        main_llm,
        paths["generation_cache"],
        allow_generate=not no_generate,
    )
    gen_by_id = gen_cache.drop_duplicates("example_id", keep="last").set_index("example_id")

    run_df = read_csv_or_empty(paths["run_csv"], RUN_COLUMNS)
    done_batches = set(run_df["t"].astype(int).tolist()) if len(run_df) else set()
    if done_batches:
        print(f"[resume] loaded {len(done_batches)} completed batches / {len(run_df)} rows")

    state = load_json(paths["state_json"], default={})
    method_state = state.get("method_state", {})

    score_model = None
    method_name = str(cfg.get("method", "")).lower().replace("-", "_")
    if method_name == "ours":
        score_cfg = dict(cfg.get("score_model", {"provider": "none"}) or {})
        score_cfg.setdefault("dataset_name", cfg.get("data", {}).get("name"))
        score_model = build_score_model(score_cfg)
    method = build_method(cfg, score_model=score_model, state=method_state)

    all_rows = run_df.to_dict("records") if len(run_df) else []
    method_pbar = tqdm(range(len(batches)), desc=cfg["run_name"], dynamic_ncols=True)

    for t in method_pbar:
        if t in done_batches:
            partial_df = pd.DataFrame(all_rows, columns=RUN_COLUMNS) if all_rows else run_df
            if len(partial_df):
                running_acc = _safe_div(int((1 - partial_df["A"].astype(int)).sum()), len(partial_df))
                method_pbar.set_postfix(acc=f"{running_acc:.3f}")
            continue

        batch_records = batches[t]
        batch_df = make_batch_rows(cfg, gen_by_id, batch_records, t)
        out_batch, new_method_state = method.process_batch(batch_df, t=t)
        out_batch = out_batch[RUN_COLUMNS]

        existing = pd.DataFrame(all_rows, columns=RUN_COLUMNS) if all_rows else pd.DataFrame(columns=RUN_COLUMNS)
        if len(existing):
            existing = existing[existing["t"].astype(int) != int(t)]
        all_rows = existing.to_dict("records") + out_batch.to_dict("records")
        current_df = pd.DataFrame(all_rows, columns=RUN_COLUMNS)
        write_csv_atomic(current_df, paths["run_csv"])

        done_batches.add(t)
        save_json_atomic(
            {
                "config": cfg.get("run_name", cfg["method"]),
                "next_batch": int(t) + 1,
                "method_state": new_method_state,
            },
            paths["state_json"],
        )

        n_sel = int(out_batch["selected"].sum())
        type_i = 0.0 if n_sel == 0 else float(((out_batch["selected"] * (1 - out_batch["A"])).sum()) / n_sel)
        spent = float((out_batch["selected"] * out_batch["cost"]).sum())
        running_correct = int((1 - current_df["A"].astype(int)).sum())
        running_acc = _safe_div(running_correct, len(current_df))
        method_pbar.set_postfix(acc=f"{running_acc:.3f}", rows=len(current_df))
        print(
            f"[batch {t:03d}] selected={n_sel:3d} spent={spent:.1f} "
            f"type-I={type_i:.3f} model-acc={running_acc:.3f} ({running_correct}/{len(current_df)})"
        )

    final_df = pd.DataFrame(all_rows, columns=RUN_COLUMNS)
    print_run_summary(final_df)
    print(f"[done] wrote {paths['run_csv']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--reset", action="store_true", help="Delete this config's run CSV/state before running.")
    parser.add_argument("--reset_generations", action="store_true", help="Delete shared main-LLM generation cache too.")
    parser.add_argument(
        "--no_generate",
        action="store_true",
        help="Never call the main LLM. Require all requested examples to already exist in the generation cache.",
    )
    args = parser.parse_args()
    run(args.config, reset=args.reset, reset_generations=args.reset_generations, no_generate=args.no_generate)


if __name__ == "__main__":
    main()
