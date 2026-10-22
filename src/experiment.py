import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import build_dataset
from src.llm import LLM
from src.methods import build_method
from src.utils import load_json, output_paths, read_csv, safe_div, save_json, write_csv

GEN_COLUMNS = ["example_id", "question", "gold_answer", "gold_final", "model_answer", "pred_answer", "A"]
RUN_COLUMNS = [
    "config", "method", "t", "batch_pos", *GEN_COLUMNS, "cost", "eta", "alpha", "beta", "selected",
    "budget", "epsilon", "main_llm", "encoder", "selector_llm",
]


def _norm(value):
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _phase_records(records, phase_filter):
    if "category_in" in phase_filter:
        allowed = {_norm(c) for c in phase_filter["category_in"]}
        return [r for r in records if _norm(r["category"]) in allowed]
    pops = np.asarray([r["popularity"] for r in records], dtype=float)
    lo, hi = (float(np.quantile(pops[np.isfinite(pops)], q)) for q in phase_filter["popularity_quantile"])
    return [r for r in records if lo <= float(r["popularity"]) <= hi]


def sample_batches(records, cfg):
    data = cfg["data"]
    seed = int(data.get("sample_seed", cfg["seed"]))
    batch_size = int(data["batch_size"])
    if "shift_phases" not in data:
        rng = np.random.default_rng(seed)
        return [[records[int(i)] for i in row] for row in rng.choice(len(records), size=(int(data["num_batches"]), batch_size), replace=True)]
    batches = []
    for k, phase in enumerate(data["shift_phases"]):
        pool = _phase_records(records, phase["filter"])
        rng = np.random.default_rng(seed + 104729 * k)
        indices = rng.choice(len(pool), size=(int(phase["num_batches"]), batch_size), replace=True)
        batches.extend([[pool[int(i)] for i in row] for row in indices])
    return batches


def ensure_generations(records, dataset, llm_cfg, path):
    cache = read_csv(path, GEN_COLUMNS).drop_duplicates("example_id", keep="last")
    done = set(cache["example_id"].astype(int))
    missing = [r for r in records if int(r["example_id"]) not in done]
    if not missing:
        return cache
    llm = LLM(llm_cfg, dataset.system_prompt)
    rows = cache.to_dict("records")
    correct = int((1 - cache["A"].astype(int)).sum())
    pbar = tqdm(missing, desc="generations", dynamic_ncols=True)
    for rec in pbar:
        answer = llm.generate(dataset.build_prompt(rec["question"]))
        pred = dataset.parse_prediction(answer)
        A = int(dataset.failure_label(pred, rec["gold_final"]))
        correct += 1 - A
        rows.append({**{k: rec[k] for k in GEN_COLUMNS[:4]}, "model_answer": answer, "pred_answer": pred, "A": A})
        pbar.set_postfix(acc=f"{correct / len(rows):.3f}")
        write_csv(pd.DataFrame(rows, columns=GEN_COLUMNS), path)
    return pd.DataFrame(rows, columns=GEN_COLUMNS)


def _words(text):
    return max(1, len(str(text).split()))


def cost_fn(generations, variant):
    if variant == "constant":
        return lambda answer: 1.0
    median = float(np.median(generations["model_answer"].map(_words).to_numpy(dtype=float)))
    return lambda answer: float(np.clip(_words(answer) / median, 0.25, 4.0))


def make_batch(cfg, gen, records, t, cost):
    policy = cfg["policy"]
    rows = []
    for pos, rec in enumerate(records):
        cached = gen.loc[int(rec["example_id"])].to_dict()
        rows.append(
            {
                "config": cfg["run_name"],
                "method": cfg["method"],
                "t": int(t),
                "batch_pos": pos,
                "example_id": int(rec["example_id"]),
                **{k: cached[k] for k in GEN_COLUMNS[1:]},
                "A": int(cached["A"]),
                "cost": cost(cached["model_answer"]),
                "eta": np.nan,
                "alpha": np.nan,
                "beta": np.nan,
                "selected": 0,
                "budget": float(policy["budget_per_batch"]),
                "epsilon": float(policy.get("epsilon", "nan")),
                "main_llm": cfg["main_llm"]["name"],
                "encoder": cfg.get("encoder", "none"),
                "selector_llm": cfg.get("selector_llm", {}).get("name", "none"),
            }
        )
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def summarize(df):
    A, s, c = (df[k].astype(float) for k in ("A", "selected", "cost"))
    print(
        f"[summary] rows={len(df)} accuracy={1 - A.mean():.4f} selected={int(s.sum())} "
        f"type-I={safe_div((s * (1 - A)).sum(), s.sum()):.4f} "
        f"type-II={safe_div(((1 - s) * A).sum(), (1 - s).sum()):.4f} "
        f"avg-cost={(s * c).sum() / df['t'].nunique():.3f}"
    )


def run(cfg, reset=False):
    paths = output_paths(cfg)
    cfg["run_name"] = paths["run_name"]
    if reset:
        for p in (paths["run_csv"], paths["state"]):
            p.unlink(missing_ok=True)
    print(f"[run] {paths['run_csv']}")

    dataset = build_dataset(cfg["data"])
    records = dataset.load_records()
    generations = ensure_generations(records, dataset, cfg["main_llm"], paths["generations"])
    gen = generations.drop_duplicates("example_id", keep="last").set_index("example_id")
    cost = cost_fn(gen, cfg["policy"].get("cost_variant", "constant"))
    batches = sample_batches(records, cfg)

    run_df = read_csv(paths["run_csv"], RUN_COLUMNS)
    done = set(run_df["t"].astype(int))
    rows = run_df.to_dict("records")
    method = build_method(cfg, load_json(paths["state"]).get("method_state", {}))

    for t in tqdm(range(len(batches)), desc=cfg["run_name"], dynamic_ncols=True):
        if t in done:
            continue
        out, state = method.process_batch(make_batch(cfg, gen, batches[t], t, cost), t)
        rows.extend(out[RUN_COLUMNS].to_dict("records"))
        write_csv(pd.DataFrame(rows, columns=RUN_COLUMNS), paths["run_csv"])
        save_json({"config": cfg["run_name"], "next_batch": t + 1, "method_state": state}, paths["state"])

    summarize(pd.DataFrame(rows, columns=RUN_COLUMNS))
