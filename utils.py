import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_name(name: Any) -> str:
    text = str(name)
    text = text.replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "none"


def fmt_float(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return safe_name(x)
    return f"{val:g}".replace("-", "m").replace(".", "p")


def _dataset_name(cfg: Dict[str, Any]) -> str:
    data = cfg["data"]
    raw_name = str(data.get("name", "data")).lower().replace("-", "_")
    name_aliases = {
        "popqa500": "popqa",
        "mmlu_pro": "mmlupro",
        "mmlu_pro500": "mmlupro",
        "mmlupro500": "mmlupro",
        "gpqa_diamond": "gpqa",
        "gpqa_main": "gpqa",
        "gpqa_extended": "gpqa",
    }
    name = safe_name(name_aliases.get(raw_name, raw_name))
    split = data.get("split")
    base = f"{name}-{safe_name(split)}" if split is not None else name

    if name in {"popqa", "mmlupro"}:
        subset_size = data.get("subset_size", data.get("max_samples", 500))
        subset_seed = data.get("subset_seed", 42)
        base = f"{base}_n{safe_name(subset_size)}_seed{safe_name(subset_seed)}"
        dataset_tag = data.get("dataset_tag", data.get("experiment_tag", None))
        if dataset_tag not in {None, "", "null", "none"}:
            base = f"{base}_{safe_name(dataset_tag)}"

    if name == "gpqa":
        hf_config = data.get("hf_config", "gpqa_diamond")
        subset_aliases = {
            "gpqa_diamond": "diamond",
            "gpqa_main": "main",
            "gpqa_extended": "extended",
        }
        subset = safe_name(subset_aliases.get(str(hf_config), str(hf_config)))
        subset_size = data.get("subset_size", data.get("max_samples", None))
        if subset_size in {None, "", "null", "none"}:
            base = f"{base}_{subset}"
        else:
            base = f"{base}_{subset}_n{safe_name(subset_size)}"

    return base


def _main_llm_name(cfg: Dict[str, Any]) -> str:
    main = cfg["main_llm"]


    return safe_name(main.get("display_name", main.get("model_name", main.get("provider", "main"))))


def model_data_name_from_config(cfg: Dict[str, Any]) -> str:
    return f"{_main_llm_name(cfg)}_{_dataset_name(cfg)}"


def _score_model_name(cfg: Dict[str, Any]) -> str:
    score = cfg.get("score_model", {}) or {}
    provider = score.get("provider", "none")
    if provider in {None, "none"}:
        return "none"
    model_name = safe_name(score.get("model_name", provider))
    return model_name


def _selector_llm_name(cfg: Dict[str, Any]) -> str:
    selector = cfg.get("selector_llm", {}) or {}
    provider = selector.get("provider", "none")
    if provider in {None, "none"}:
        return "none"
    return safe_name(selector.get("display_name", selector.get("model_name", provider)))


def _budget_variant(cfg: Dict[str, Any]) -> str:
    policy = cfg["policy"]
    return f"budget{fmt_float(policy.get('budget_per_batch', 0))}"


def _method_params(cfg: Dict[str, Any]) -> list[str]:
    method = str(cfg.get("method", "method")).lower().replace("-", "_")
    policy = cfg["policy"]
    seed = safe_name(cfg.get("seed", 0))

    if method == "random":
        return [f"seed{seed}"]

    if method == "ours":
        return [
            _score_model_name(cfg),
            f"eps{fmt_float(policy.get('epsilon', 0))}",
            f"alpha{fmt_float(policy.get('alpha_step_size', 0))}",
            f"theta{fmt_float(policy.get('theta_step_size', 0))}",
        ]

    if method == "ours_random":
        return [
            f"eps{fmt_float(policy.get('epsilon', 0))}",
            f"alpha{fmt_float(policy.get('alpha_step_size', 0))}",
            f"seed{seed}",
        ]

    if method == "ours_llm":
        selector_name = _selector_llm_name(cfg)
        if selector_name == "none":
            selector_name = safe_name((cfg.get("score_llm", {}) or {}).get("model_name", "llama-3.3-70b-versatile"))
        return [
            selector_name,
            f"eps{fmt_float(policy.get('epsilon', 0))}",
            f"alpha{fmt_float(policy.get('alpha_step_size', 0))}",
        ]

    if method == "llm_select":
        return [
            _selector_llm_name(cfg),
            f"seed{seed}",
        ]

    params = [f"seed{seed}"]
    for key in sorted(policy):
        if key in {"budget_per_batch", "cost_variant"}:
            continue
        params.append(f"{safe_name(key)}{fmt_float(policy[key])}")
    return params


def run_name_from_config(cfg: Dict[str, Any]) -> str:

    parts = [
        safe_name(str(cfg.get("method", "method")).lower().replace("-", "_")),
        _main_llm_name(cfg),
        _dataset_name(cfg),
        _budget_variant(cfg),
    ]
    parts.extend(_method_params(cfg))
    return "_".join(parts)


def generation_cache_name(cfg: Dict[str, Any]) -> str:

    return f"gen_{_main_llm_name(cfg)}_{_dataset_name(cfg)}.csv"


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def write_csv_atomic(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def read_csv_or_empty(path: str | Path, columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def load_json(path: str | Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def project_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:

    out_root = ensure_dir(cfg.get("output_dir", "outputs"))
    model_data = model_data_name_from_config(cfg)
    output_subdir = cfg.get("output_subdir", None)
    out_name = safe_name(output_subdir) if output_subdir not in {None, "", "null", "none"} else model_data
    out = ensure_dir(out_root / out_name)
    paths = {
        "output_root": out_root,
        "model_data_name": model_data,
        "output_dir": out,
        "generation_cache": out / generation_cache_name(cfg),
    }

    if "method" in cfg and "policy" in cfg:
        run_stem = run_name_from_config(cfg)
        paths.update(
            {
                "run_name": run_stem,
                "run_csv": out / f"{run_stem}.csv",
                "state_json": out / f"{run_stem}_state.json",
            }
        )
    else:
        paths["run_name"] = None

    return paths
