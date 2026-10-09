from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DATASET_ORDER = ("MATH-500", "MMLU-Pro", "PopQA", "GPQA")
MODEL_ORDER = ("Llama-3.3-70B", "Qwen3-32B")
SHIFT_DATASETS = ("MMLU-Pro", "PopQA")

ROW_COLORS = {
    "MATH-500": "mathsoft",
    "MMLU-Pro": "mmlusoft",
    "PopQA": "popsoft",
    "GPQA": "gpqasoft",
}

EXPECTED_MAIN_COUNTS = {
    "MATH-500": 500,
    "MMLU-Pro": 500,
    "PopQA": 500,
    "GPQA": 448,
}


@dataclass(frozen=True)
class AccuracyResult:
    experiment: str
    dataset: str
    model: str
    accuracy: float
    correct: int
    total: int
    basis: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute all model-accuracy values needed by the paper tables."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Output directory to scan (default: ./outputs).",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of decimal places used in printed/LaTeX values (default: 3).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print results only; do not write stats_results.json and stats_latex_rows.txt.",
    )
    return parser.parse_args()


def normalized_path(path: Path) -> str:
    return path.as_posix().lower().replace("_", "-")


def detect_model(path: Path) -> str | None:
    text = normalized_path(path)
    if "llama-3.3-70b" in text:
        return "Llama-3.3-70B"
    if "qwen3-32b" in text or "qwen-qwen3-32b" in text:
        return "Qwen3-32B"
    return None


def detect_dataset(path: Path) -> str | None:
    text = normalized_path(path)
    if "math500" in text or "math-500" in text:
        return "MATH-500"
    if "mmlupro" in text or "mmlu-pro" in text:
        return "MMLU-Pro"
    if "popqa" in text:
        return "PopQA"
    if "gpqa" in text:
        return "GPQA"
    return None


def is_review_length(path: Path) -> bool:
    text = normalized_path(path)
    return "review-length" in text or "reviewlength" in text


def is_shift(path: Path) -> bool:
    text = normalized_path(path)
    return any(token in text for token in ("multishift", "popshift", "distribution-shift"))


def parse_binary(value: object, *, path: Path, row_number: int) -> int:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{path}: row {row_number}: invalid A value {value!r}") from exc
    if not math.isfinite(number) or number not in (0.0, 1.0):
        raise ValueError(f"{path}: row {row_number}: A must be 0 or 1, got {value!r}")
    return int(number)


def read_accuracy(
    path: Path,
    *,
    unique_examples: bool,
    stream_mode: bool = False,
) -> tuple[float, int, int, tuple[tuple[str, str, str, int], ...]]:

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "A" not in reader.fieldnames:
            raise ValueError(f"{path}: CSV does not contain an 'A' column")

        by_example: dict[str, int] = {}
        stream_rows: dict[tuple[int, int], tuple[str, int]] = {}
        all_labels: list[int] = []

        for row_number, row in enumerate(reader, start=2):
            a = parse_binary(row.get("A"), path=path, row_number=row_number)

            if unique_examples:
                example_id = str(row.get("example_id", "")).strip()
                if not example_id:
                    raise ValueError(
                        f"{path}: row {row_number}: unique-example accuracy requires example_id"
                    )
                by_example[example_id] = a
            elif stream_mode:
                try:
                    t = int(float(str(row.get("t", "")).strip()))
                    pos = int(float(str(row.get("batch_pos", "")).strip()))
                except ValueError as exc:
                    raise ValueError(
                        f"{path}: row {row_number}: stream accuracy requires integer t and batch_pos"
                    ) from exc
                example_id = str(row.get("example_id", "")).strip()
                stream_rows[(t, pos)] = (example_id, a)
            else:
                all_labels.append(a)

    if unique_examples:
        labels = list(by_example.values())
        signature: tuple[tuple[str, str, str, int], ...] = ()
    elif stream_mode:
        ordered = sorted(stream_rows.items())
        labels = [item[1][1] for item in ordered]
        signature = tuple(
            (str(t), str(pos), example_id, a)
            for (t, pos), (example_id, a) in ordered
        )
    else:
        labels = all_labels
        signature = ()

    if not labels:
        raise ValueError(f"{path}: no valid rows found")

    correct = sum(1 - a for a in labels)
    total = len(labels)
    return correct / total, correct, total, signature


def choose_largest(results: Sequence[tuple[Path, float, int, int]]) -> tuple[Path, float, int, int]:
    return max(results, key=lambda item: (item[3], item[0].stat().st_mtime_ns))


def collect_main_accuracy(outputs_dir: Path) -> list[AccuracyResult]:
    grouped: dict[tuple[str, str], list[tuple[Path, float, int, int]]] = {}

    for path in outputs_dir.rglob("gen_*.csv"):
        if is_review_length(path) or is_shift(path):
            continue
        model = detect_model(path)
        dataset = detect_dataset(path)
        if model not in MODEL_ORDER or dataset not in DATASET_ORDER:
            continue
        accuracy, correct, total, _ = read_accuracy(path, unique_examples=True)
        grouped.setdefault((dataset, model), []).append((path, accuracy, correct, total))

    results: list[AccuracyResult] = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            candidates = grouped.get((dataset, model), [])
            if not candidates:
                continue
            path, accuracy, correct, total = choose_largest(candidates)
            expected = EXPECTED_MAIN_COUNTS[dataset]
            if total < expected:
                print(
                    f"[warning] {model} / {dataset}: generation cache has {total} unique rows; "
                    f"the expected main-pool size is {expected}.",
                    file=sys.stderr,
                )
            if len(candidates) > 1:
                print(
                    f"[note] {model} / {dataset}: found {len(candidates)} main caches; "
                    f"using the largest one: {path}",
                    file=sys.stderr,
                )
            results.append(
                AccuracyResult(
                    experiment="main",
                    dataset=dataset,
                    model=model,
                    accuracy=accuracy,
                    correct=correct,
                    total=total,
                    basis="unique generation-cache examples",
                    source=str(path),
                )
            )
    return results


def non_generation_csvs(directory: Path) -> Iterable[Path]:
    for path in directory.glob("*.csv"):
        if not path.name.startswith("gen_"):
            yield path


def collect_shift_accuracy(outputs_dir: Path) -> list[AccuracyResult]:
    shift_dirs = sorted({p.parent for p in outputs_dir.rglob("*.csv") if is_shift(p)})
    results: list[AccuracyResult] = []

    for dataset in SHIFT_DATASETS:
        matching_dirs = [d for d in shift_dirs if detect_dataset(d) == dataset and detect_model(d) == "Llama-3.3-70B"]
        if not matching_dirs:
            continue


        best: tuple[Path, float, int, int, tuple[tuple[str, str, str, int], ...]] | None = None
        complete_runs: list[tuple[Path, float, int, int, tuple[tuple[str, str, str, int], ...]]] = []

        for directory in matching_dirs:
            for path in non_generation_csvs(directory):
                try:
                    accuracy, correct, total, signature = read_accuracy(
                        path, unique_examples=False, stream_mode=True
                    )
                except (ValueError, OSError) as exc:
                    print(f"[warning] skipping {path}: {exc}", file=sys.stderr)
                    continue
                item = (path, accuracy, correct, total, signature)
                complete_runs.append(item)
                if best is None or (total, path.stat().st_mtime_ns) > (
                    best[3], best[0].stat().st_mtime_ns
                ):
                    best = item

        if best is None:

            cache_candidates = [
                p
                for directory in matching_dirs
                for p in directory.glob("gen_*.csv")
            ]
            if not cache_candidates:
                continue
            cache_results = []
            for path in cache_candidates:
                accuracy, correct, total, _ = read_accuracy(path, unique_examples=True)
                cache_results.append((path, accuracy, correct, total))
            path, accuracy, correct, total = choose_largest(cache_results)
            print(
                f"[warning] {dataset}: no shift run CSV found; using unique generation-cache "
                "accuracy instead of exact stream accuracy.",
                file=sys.stderr,
            )
            results.append(
                AccuracyResult(
                    experiment="distribution_shift",
                    dataset=dataset,
                    model="Llama-3.3-70B",
                    accuracy=accuracy,
                    correct=correct,
                    total=total,
                    basis="unique shift generation-cache examples (fallback)",
                    source=str(path),
                )
            )
            continue

        path, accuracy, correct, total, signature = best


        for other_path, _, _, other_total, other_signature in complete_runs:
            if other_path == path or other_total != total:
                continue
            if other_signature != signature:
                print(
                    f"[warning] shift streams differ between {path.name} and {other_path.name}. "
                    f"Using {path.name} because it was selected as the reference run.",
                    file=sys.stderr,
                )

        results.append(
            AccuracyResult(
                experiment="distribution_shift",
                dataset=dataset,
                model="Llama-3.3-70B",
                accuracy=accuracy,
                correct=correct,
                total=total,
                basis="exact sampled shift-stream rows",
                source=str(path),
            )
        )

    return results


def fmt(value: float | None, digits: int) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def result_lookup(results: Sequence[AccuracyResult]) -> dict[tuple[str, str, str], AccuracyResult]:
    return {(r.experiment, r.dataset, r.model): r for r in results}


def print_main_table(results: Sequence[AccuracyResult], digits: int) -> None:
    lookup = result_lookup(results)
    print("\nMAIN BASE-MODEL ACCURACY (unique generation-cache examples)")
    print("=" * 72)
    print(f"{'Dataset':<14} {'Llama-3.3-70B':>18} {'Qwen3-32B':>18}")
    for dataset in DATASET_ORDER:
        values = []
        for model in MODEL_ORDER:
            r = lookup.get(("main", dataset, model))
            values.append("MISSING" if r is None else f"{fmt(r.accuracy, digits)} ({r.correct}/{r.total})")
        print(f"{dataset:<14} {values[0]:>18} {values[1]:>18}")

    print("\nLaTeX rows for Table~\\ref{tab:model-accuracies}:")
    for dataset in DATASET_ORDER:
        llama = lookup.get(("main", dataset, "Llama-3.3-70B"))
        qwen = lookup.get(("main", dataset, "Qwen3-32B"))
        print(
            f"\\rowcolor{{{ROW_COLORS[dataset]}}} "
            f"\\mbox{{{dataset}}} & {fmt(llama.accuracy if llama else None, digits)} "
            f"& {fmt(qwen.accuracy if qwen else None, digits)} \\\\"
        )


def print_shift_table(results: Sequence[AccuracyResult], digits: int) -> None:
    lookup = result_lookup(results)
    print("\nDISTRIBUTION-SHIFT MODEL ACCURACY (exact sampled stream)")
    print("=" * 72)
    print(f"{'Dataset':<14} {'Llama-3.3-70B':>24}")
    for dataset in SHIFT_DATASETS:
        r = lookup.get(("distribution_shift", dataset, "Llama-3.3-70B"))
        text = "MISSING" if r is None else f"{fmt(r.accuracy, digits)} ({r.correct}/{r.total})"
        print(f"{dataset:<14} {text:>24}")

    print("\nLaTeX rows for Table~\\ref{tab:shift-model-accuracy}:")
    for dataset in SHIFT_DATASETS:
        r = lookup.get(("distribution_shift", dataset, "Llama-3.3-70B"))
        print(
            f"\\rowcolor{{{ROW_COLORS[dataset]}}} "
            f"{dataset} & {fmt(r.accuracy if r else None, digits)} \\\\"
        )


def latex_rows(results: Sequence[AccuracyResult], digits: int) -> str:
    lookup = result_lookup(results)
    lines = [
        "% Main base-model accuracy table",
    ]
    for dataset in DATASET_ORDER:
        llama = lookup.get(("main", dataset, "Llama-3.3-70B"))
        qwen = lookup.get(("main", dataset, "Qwen3-32B"))
        lines.append(
            f"\\rowcolor{{{ROW_COLORS[dataset]}}} "
            f"\\mbox{{{dataset}}} & {fmt(llama.accuracy if llama else None, digits)} "
            f"& {fmt(qwen.accuracy if qwen else None, digits)} \\\\"
        )
    lines.extend(["", "% Distribution-shift accuracy table"])
    for dataset in SHIFT_DATASETS:
        r = lookup.get(("distribution_shift", dataset, "Llama-3.3-70B"))
        lines.append(
            f"\\rowcolor{{{ROW_COLORS[dataset]}}} "
            f"{dataset} & {fmt(r.accuracy if r else None, digits)} \\\\"
        )
    return "\n".join(lines) + "\n"


def write_outputs(results: Sequence[AccuracyResult], digits: int) -> None:
    json_path = Path("stats_results.json")
    latex_path = Path("stats_latex_rows.txt")

    payload = {
        "accuracy_format_digits": digits,
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latex_path.write_text(latex_rows(results, digits), encoding="utf-8")
    print(f"\nWrote {json_path} and {latex_path}")


def main() -> int:
    args = parse_args()
    outputs_dir = args.outputs.resolve()
    if not outputs_dir.is_dir():
        print(f"error: outputs directory does not exist: {outputs_dir}", file=sys.stderr)
        return 2
    if args.digits < 0 or args.digits > 8:
        print("error: --digits must be between 0 and 8", file=sys.stderr)
        return 2

    try:
        results = collect_main_accuracy(outputs_dir)
        results.extend(collect_shift_accuracy(outputs_dir))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_main_table(results, args.digits)
    print_shift_table(results, args.digits)

    missing_main = [
        (dataset, model)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        if not any(
            r.experiment == "main" and r.dataset == dataset and r.model == model
            for r in results
        )
    ]
    missing_shift = [
        dataset
        for dataset in SHIFT_DATASETS
        if not any(
            r.experiment == "distribution_shift" and r.dataset == dataset
            for r in results
        )
    ]

    if missing_main:
        print(f"\n[warning] missing main results: {missing_main}", file=sys.stderr)
    if missing_shift:
        print(f"[warning] missing shift results: {missing_shift}", file=sys.stderr)

    if not args.no_write:
        write_outputs(results, args.digits)

    return 0 if not missing_main and not missing_shift else 1


if __name__ == "__main__":
    raise SystemExit(main())
