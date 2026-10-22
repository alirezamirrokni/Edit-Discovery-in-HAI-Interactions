import pandas as pd

from src.analysis import DATASETS, METHODS, MODELS, config, discovery_metrics, load_run
from src.utils import output_paths

SHIFT = {"mmlupro_shift": "MMLU-Pro", "popqa_shift": "PopQA"}


def accuracy_table():
    rows = []
    for model, model_label in MODELS.items():
        for key, label in DATASETS.items():
            path = output_paths(config(model, key, "random"))["generations"]
            if path.exists():
                gen = pd.read_csv(path).drop_duplicates("example_id", keep="last")
                rows.append({"Model": model_label, "Dataset": label, "Accuracy": 1 - gen["A"].mean(), "n": len(gen)})
    for key, label in SHIFT.items():
        run = next((r for r in (load_run("llama", key, m) for m in ("random", *METHODS)) if r is not None), None)
        if run is not None:
            stream = run.drop_duplicates(["t", "batch_pos"], keep="last")
            rows.append({"Model": MODELS["llama"], "Dataset": f"{label} (shift stream)", "Accuracy": 1 - stream["A"].mean(), "n": len(stream)})
    return pd.DataFrame(rows)


def discovery_table():
    rows = []
    for model, model_label in MODELS.items():
        for key, label in DATASETS.items():
            for method, method_label in METHODS.items():
                run = load_run(model, key, method)
                if run is not None:
                    rows.append({"Model": model_label, "Dataset": label, "Method": method_label, **discovery_metrics(run)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("Response-model accuracy\n")
    print(accuracy_table().to_string(index=False, float_format="%.3f"))
    print("\nDiscovery metrics (Type-I, Type-II and cost are running averages over rounds)\n")
    print(discovery_table().to_string(index=False, float_format="%.3f"))
