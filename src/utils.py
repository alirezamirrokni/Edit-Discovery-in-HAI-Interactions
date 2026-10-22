import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).replace("/", "-"))
    return text.strip("_") or "none"


def fmt_float(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def safe_div(num, den):
    return 0.0 if den <= 0 else float(num / den)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def read_csv(path, columns):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def dataset_id(data):
    name = data["name"]
    base = f"{name}-{safe_name(data['split'])}"
    if name in {"popqa", "mmlupro"}:
        base += f"_n{data['subset_size']}_seed{data['subset_seed']}"
        if data.get("tag"):
            base += f"_{safe_name(data['tag'])}"
    if name == "gpqa":
        base += f"_{data['hf_config'].removeprefix('gpqa_')}_n{data['subset_size']}"
    return base


def run_name(cfg, model_data):
    method = cfg["method"]
    policy = cfg["policy"]
    seed = f"seed{cfg['seed']}"
    eps = f"eps{fmt_float(policy.get('epsilon', 0))}"
    alpha = f"alpha{fmt_float(policy.get('alpha_step_size', 0))}"
    params = {
        "random": [seed],
        "edit_rep": [safe_name(cfg.get("encoder")), eps, alpha, f"theta{fmt_float(policy.get('theta_step_size', 0))}"],
        "edit_rand": [eps, alpha, seed],
        "edit_llm": [safe_name(cfg.get("selector_llm", {}).get("name")), eps, alpha],
        "llm_select": [safe_name(cfg.get("selector_llm", {}).get("name")), seed],
    }[method]
    return "_".join([method, model_data, f"budget{fmt_float(policy['budget_per_batch'])}", *params])


def output_paths(cfg):
    root = Path(cfg.get("output_dir", "outputs"))
    model_data = f"{safe_name(cfg['main_llm']['name'])}_{dataset_id(cfg['data'])}"
    paths = {"generations": root / model_data / f"gen_{model_data}.csv"}
    if "method" in cfg:
        cost_variant = cfg["policy"].get("cost_variant", "constant")
        out = root / (model_data if cost_variant == "constant" else f"{model_data}_{cost_variant}")
        name = run_name(cfg, model_data)
        paths.update(run_name=name, run_csv=out / f"{name}.csv", state=out / f"{name}_state.json")
    return paths
