from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path


DEFAULT_METHODS = [
    "v2_8a_static_a064_d080",
    "v2_8a_static_a048_d096",
    "v2_8a_static_a032_d112",
    "v2_8a_static_a016_d128",
    "v2_8a_static_a000_d144",
]


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def _mean(values: list[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def _median(values: list[float]) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return math.nan
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, float(q))) * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def bootstrap_ci(deltas: list[float], *, iterations: int = 2000, seed: int = 20260621) -> dict:
    if not deltas:
        return {"mean": math.nan, "ci90_low": math.nan, "ci90_high": math.nan, "iterations": int(iterations)}
    rng = random.Random(int(seed))
    samples = []
    for _ in range(int(iterations)):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        samples.append(_mean(sample))
    samples.sort()
    return {
        "mean": _mean(deltas),
        "ci90_low": _percentile(samples, 0.05),
        "ci90_high": _percentile(samples, 0.95),
        "iterations": int(iterations),
    }


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(str(row["case_id"]), str(row["method"])): row for row in rows}


def _case_ids(rows: list[dict], baseline_method: str) -> list[str]:
    return sorted({str(row["case_id"]) for row in rows if str(row.get("method")) == baseline_method})


def summarize_deltas(dino: list[float], clip: list[float], copy: list[float]) -> dict:
    return {
        "case_count": len(dino),
        "dino_delta_mean": _mean(dino),
        "dino_delta_median": _median(dino),
        "dino_win_rate": _mean([1.0 if value > 0.0 else 0.0 for value in dino]),
        "dino_delta_min": min(dino) if dino else math.nan,
        "dino_delta_max": max(dino) if dino else math.nan,
        "clip_delta_mean": _mean(clip),
        "copy_ssim_delta_mean": _mean(copy),
        "copy_risk_count_delta_gt_0_015": sum(1 for value in copy if value > 0.015),
    }


def evaluate_predictions(
    rows: list[dict],
    predictions: dict[str, str],
    *,
    baseline_method: str,
    run_label: str = "",
) -> tuple[dict, list[dict]]:
    indexed = _index(rows)
    dino = []
    clip = []
    copy = []
    prediction_rows = []
    for case_id, method in sorted(predictions.items()):
        base = indexed[(case_id, baseline_method)]
        selected = indexed[(case_id, method)]
        dino_delta = _float(selected, "dino_sim_to_target") - _float(base, "dino_sim_to_target")
        clip_delta = _float(selected, "clip_text_image_sim") - _float(base, "clip_text_image_sim")
        copy_delta = _float(selected, "ref_copy_ssim_max") - _float(base, "ref_copy_ssim_max")
        dino.append(dino_delta)
        clip.append(clip_delta)
        copy.append(copy_delta)
        prediction_rows.append(
            {
                "run": run_label,
                "case_id": case_id,
                "subject_id": base.get("subject_id", ""),
                "category": base.get("category", ""),
                "selected_method": method,
                "dino_delta": dino_delta,
                "clip_delta": clip_delta,
                "copy_ssim_delta": copy_delta,
            }
        )
    summary = summarize_deltas(dino, clip, copy)
    return summary, prediction_rows


def _available_methods(indexed: dict[tuple[str, str], dict], case_id: str, candidate_methods: list[str]) -> list[str]:
    return [method for method in candidate_methods if (case_id, method) in indexed]


def fixed_method_predictions(rows: list[dict], *, baseline_method: str, method: str) -> dict[str, str]:
    indexed = _index(rows)
    return {case_id: method for case_id in _case_ids(rows, baseline_method) if (case_id, method) in indexed}


def reference_selector_predictions(rows: list[dict], *, baseline_method: str, candidate_methods: list[str]) -> dict[str, str]:
    indexed = _index(rows)
    predictions = {}
    for case_id in _case_ids(rows, baseline_method):
        available = _available_methods(indexed, case_id, candidate_methods)
        if available:
            predictions[case_id] = max(
                available,
                key=lambda method: _float(indexed[(case_id, method)], "dino_sim_to_multi_ref_mean"),
            )
    return predictions


def target_oracle_predictions(rows: list[dict], *, baseline_method: str, candidate_methods: list[str]) -> dict[str, str]:
    indexed = _index(rows)
    predictions = {}
    for case_id in _case_ids(rows, baseline_method):
        available = _available_methods(indexed, case_id, candidate_methods)
        if available:
            predictions[case_id] = max(
                available,
                key=lambda method: _float(indexed[(case_id, method)], "dino_sim_to_target"),
            )
    return predictions


def analyze_run(
    label: str,
    rows: list[dict],
    *,
    baseline_method: str,
    candidate_methods: list[str],
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    fixed = {}
    fixed_rows = []
    fixed_prediction_rows = {}
    for method in candidate_methods:
        predictions = fixed_method_predictions(rows, baseline_method=baseline_method, method=method)
        if not predictions:
            continue
        summary, prediction_rows = evaluate_predictions(rows, predictions, baseline_method=baseline_method, run_label=label)
        fixed_prediction_rows[method] = prediction_rows
        fixed[method] = {
            "summary": summary,
            "bootstrap_ci": bootstrap_ci(
                [float(row["dino_delta"]) for row in prediction_rows],
                iterations=bootstrap_iterations,
                seed=seed,
            ),
        }
        fixed_rows.append({"run": label, "method": method, **summary})
    best_fixed_method = max(fixed, key=lambda method: float(fixed[method]["summary"]["dino_delta_mean"]))

    selector_predictions = reference_selector_predictions(
        rows,
        baseline_method=baseline_method,
        candidate_methods=candidate_methods,
    )
    selector_summary, selector_rows = evaluate_predictions(
        rows,
        selector_predictions,
        baseline_method=baseline_method,
        run_label=label,
    )
    oracle_predictions = target_oracle_predictions(
        rows,
        baseline_method=baseline_method,
        candidate_methods=candidate_methods,
    )
    oracle_summary, oracle_rows = evaluate_predictions(
        rows,
        oracle_predictions,
        baseline_method=baseline_method,
        run_label=label,
    )

    best_fixed_summary = fixed[best_fixed_method]["summary"]
    return {
        "label": label,
        "case_count": int(selector_summary["case_count"]),
        "best_fixed_method": best_fixed_method,
        "fixed_methods": fixed,
        "fixed_method_rows": fixed_rows,
        "best_fixed": best_fixed_summary,
        "reference_selector": selector_summary,
        "reference_selector_ci": bootstrap_ci(
            [float(row["dino_delta"]) for row in selector_rows],
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "target_oracle": oracle_summary,
        "target_oracle_ci": bootstrap_ci(
            [float(row["dino_delta"]) for row in oracle_rows],
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "selector_method_counts": dict(Counter(selector_predictions.values())),
        "oracle_method_counts": dict(Counter(oracle_predictions.values())),
        "selection_gap_over_best_fixed": float(selector_summary["dino_delta_mean"])
        - float(best_fixed_summary["dino_delta_mean"]),
        "oracle_gap_over_best_fixed": float(oracle_summary["dino_delta_mean"])
        - float(best_fixed_summary["dino_delta_mean"]),
        "selector_retention_vs_oracle": (
            float(selector_summary["dino_delta_mean"]) / float(oracle_summary["dino_delta_mean"])
            if abs(float(oracle_summary["dino_delta_mean"])) > 1.0e-12
            else math.nan
        ),
        "selector_prediction_rows": selector_rows,
        "oracle_prediction_rows": oracle_rows,
        "best_fixed_prediction_rows": fixed_prediction_rows[best_fixed_method],
    }


def _aggregate_prediction_rows(rows: list[dict]) -> dict:
    dino = [float(row["dino_delta"]) for row in rows]
    clip = [float(row["clip_delta"]) for row in rows]
    copy = [float(row["copy_ssim_delta"]) for row in rows]
    return summarize_deltas(dino, clip, copy)


def analyze_compressibility(
    *,
    runs: list[tuple[str, str | Path]],
    baseline_method: str = "v2_8a_static_a064_d080",
    candidate_methods: list[str] | None = None,
    bootstrap_iterations: int = 2000,
    seed: int = 20260621,
) -> dict:
    candidate_methods = list(candidate_methods or DEFAULT_METHODS)
    run_results = []
    aggregate_selector_rows = []
    aggregate_oracle_rows = []
    aggregate_best_fixed_rows = []
    all_rows_with_run = []
    for label, case_metrics in runs:
        rows = read_csv(case_metrics)
        for row in rows:
            tagged = dict(row)
            tagged["case_id"] = f"{label}::{row['case_id']}"
            tagged["_original_case_id"] = row["case_id"]
            tagged["_run"] = label
            all_rows_with_run.append(tagged)
        result = analyze_run(
            str(label),
            rows,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        run_results.append(result)
        aggregate_selector_rows.extend(result["selector_prediction_rows"])
        aggregate_oracle_rows.extend(result["oracle_prediction_rows"])
        aggregate_best_fixed_rows.extend(result["best_fixed_prediction_rows"])
    aggregate_best_fixed = _aggregate_prediction_rows(aggregate_best_fixed_rows)
    aggregate_selector = _aggregate_prediction_rows(aggregate_selector_rows)
    aggregate_oracle = _aggregate_prediction_rows(aggregate_oracle_rows)
    global_fixed = {}
    for method in candidate_methods:
        predictions = fixed_method_predictions(all_rows_with_run, baseline_method=baseline_method, method=method)
        if not predictions:
            continue
        summary, prediction_rows = evaluate_predictions(
            all_rows_with_run,
            predictions,
            baseline_method=baseline_method,
            run_label="aggregate_global",
        )
        global_fixed[method] = {
            "summary": summary,
            "bootstrap_ci": bootstrap_ci(
                [float(row["dino_delta"]) for row in prediction_rows],
                iterations=bootstrap_iterations,
                seed=seed,
            ),
        }
    global_best_fixed_method = max(
        global_fixed,
        key=lambda method: float(global_fixed[method]["summary"]["dino_delta_mean"]),
    )
    global_best_fixed = global_fixed[global_best_fixed_method]["summary"]
    selection_gap = float(aggregate_selector["dino_delta_mean"]) - float(aggregate_best_fixed["dino_delta_mean"])
    global_selection_gap = float(aggregate_selector["dino_delta_mean"]) - float(global_best_fixed["dino_delta_mean"])
    single_path_ready = (
        float(global_best_fixed["dino_delta_mean"]) >= 0.010
        and global_selection_gap <= 0.005
        and float(global_best_fixed["dino_win_rate"]) >= 0.60
        and float(global_best_fixed["clip_delta_mean"]) >= -0.003
        and float(global_best_fixed["copy_ssim_delta_mean"]) <= 0.015
    )
    return {
        "baseline_method": baseline_method,
        "candidate_methods": candidate_methods,
        "runs": [
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "selector_prediction_rows",
                    "oracle_prediction_rows",
                    "fixed_method_rows",
                    "best_fixed_prediction_rows",
                }
            }
            for result in run_results
        ],
        "aggregate": {
            "best_fixed": aggregate_best_fixed,
            "global_fixed_methods": global_fixed,
            "global_best_fixed_method": global_best_fixed_method,
            "global_best_fixed": global_best_fixed,
            "reference_selector": aggregate_selector,
            "target_oracle": aggregate_oracle,
            "selection_gap_over_best_fixed": selection_gap,
            "selection_gap_over_global_best_fixed": global_selection_gap,
            "oracle_gap_over_best_fixed": float(aggregate_oracle["dino_delta_mean"])
            - float(aggregate_best_fixed["dino_delta_mean"]),
            "oracle_gap_over_global_best_fixed": float(aggregate_oracle["dino_delta_mean"])
            - float(global_best_fixed["dino_delta_mean"]),
            "selector_retention_vs_oracle": (
                float(aggregate_selector["dino_delta_mean"]) / float(aggregate_oracle["dino_delta_mean"])
                if abs(float(aggregate_oracle["dino_delta_mean"])) > 1.0e-12
                else math.nan
            ),
        },
        "decision": {
            "single_path_compression_ready": bool(single_path_ready),
            "reason": (
                "best fixed candidate retains most selector gain"
                if single_path_ready
                else "per-case selection gain is not explained by a single fixed path"
            ),
        },
    }


def analyze(args) -> dict:
    run_specs = args.run or []
    if not run_specs:
        raise ValueError("At least one --run is required.")
    result = analyze_compressibility(
        runs=[(label, case_metrics) for label, case_metrics in run_specs],
        baseline_method=str(args.baseline_method),
        candidate_methods=list(args.candidate_methods or DEFAULT_METHODS),
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v2_14_compressibility_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fixed_rows = []
    selector_rows = []
    oracle_rows = []
    for label, case_metrics in run_specs:
        rows = read_csv(case_metrics)
        run_result = analyze_run(
            label,
            rows,
            baseline_method=str(args.baseline_method),
            candidate_methods=list(args.candidate_methods or DEFAULT_METHODS),
            bootstrap_iterations=int(args.bootstrap_iterations),
            seed=int(args.seed),
        )
        fixed_rows.extend(run_result["fixed_method_rows"])
        selector_rows.extend(run_result["selector_prediction_rows"])
        oracle_rows.extend(run_result["oracle_prediction_rows"])
    write_csv(output_dir / "v2_14_fixed_method_summary.csv", fixed_rows)
    write_csv(output_dir / "v2_14_reference_selector_predictions.csv", selector_rows)
    write_csv(output_dir / "v2_14_target_oracle_predictions.csv", oracle_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze whether v2.14 selection gains can be compressed into a fixed single path.")
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "CASE_METRICS"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_method", default="v2_8a_static_a064_d080")
    parser.add_argument("--candidate_methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260621)
    return parser.parse_args(argv)


def main(argv=None):
    return analyze(parse_args(argv))


if __name__ == "__main__":
    main()
