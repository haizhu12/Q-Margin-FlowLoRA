from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


COPY_RISK_DELTA_THRESHOLD = 0.015
DEFAULT_BASELINE_METHOD = "v2_8a_static_a064_d080"

OFFICIAL_STATIC_METHODS = [
    "v2_8a_static_a064_d080",
    "v2_8a_static_a048_d096",
    "v2_8a_static_a032_d112",
    "v2_8a_static_a016_d128",
    "v2_8a_static_a000_d144",
]

A5_A064_SEEDS = [f"v2_14_a5_same_a064_seed{idx}" for idx in range(5)]
A5_A048_SEEDS = [f"v2_14_a5_same_a048_seed{idx}" for idx in range(5)]
G1_A064_SEEDS = [f"v2_14_g1_independent_noise_a064_seed{idx}" for idx in range(3)]

DIRECT_GROUPS = {
    "v2_8a_static_a064_d080": "official_static_route",
    "v2_8a_static_a048_d096": "official_static_route",
    "v2_8a_static_a032_d112": "official_static_route",
    "v2_8a_static_a016_d128": "official_static_route",
    "v2_8a_static_a000_d144": "official_static_route",
    **{method: "A5_same_route_sampling_individual" for method in A5_A064_SEEDS + A5_A048_SEEDS},
    "v2_14_b1_anchor_nearest_grid": "B1_anchor_selector",
    "v2_14_b1_anchor_uniform_stride": "B1_anchor_selector",
    "v2_14_b1_anchor_random": "B1_anchor_selector",
    "v2_14_b1_anchor_top_detail": "B1_anchor_selector",
    "v2_14_c1_detail_local_residual": "C1_detail_score",
    "v2_14_c1_detail_token_l2": "C1_detail_score",
    "v2_14_c1_detail_random": "C1_detail_score",
    "v2_14_c1_detail_lowest_residual": "C1_detail_score",
    "v2_14_c2_mask_on": "C2_anchor_mask",
    "v2_14_c2_mask_off": "C2_anchor_mask",
    "v2_14_d1_per_ref_quota": "D1_reference_quota",
    "v2_14_d1_global_quota": "D1_reference_quota",
    **{method: "G1_noise_independence_individual" for method in G1_A064_SEEDS},
}

AGGREGATE_SPECS = [
    {
        "method": "A5_structured_5route_oracle",
        "group": "A5_route_diversity",
        "mode": "best",
        "source_methods": OFFICIAL_STATIC_METHODS,
        "baseline_method": DEFAULT_BASELINE_METHOD,
    },
    {
        "method": "A5_same_a064_5seed_oracle",
        "group": "A5_route_diversity",
        "mode": "best",
        "source_methods": A5_A064_SEEDS,
        "baseline_method": "v2_14_a5_same_a064_seed0",
    },
    {
        "method": "A5_same_a048_5seed_oracle",
        "group": "A5_route_diversity",
        "mode": "best",
        "source_methods": A5_A048_SEEDS,
        "baseline_method": "v2_14_a5_same_a048_seed0",
    },
    {
        "method": "G1_independent_noise_a064_mean3",
        "group": "G1_noise_independence",
        "mode": "mean",
        "source_methods": G1_A064_SEEDS,
        "baseline_method": DEFAULT_BASELINE_METHOD,
    },
    {
        "method": "G1_independent_noise_a064_best3_oracle",
        "group": "G1_noise_independence",
        "mode": "best",
        "source_methods": G1_A064_SEEDS,
        "baseline_method": DEFAULT_BASELINE_METHOD,
    },
]

RELATIVE_SUMMARY_FIELDS = [
    "group",
    "method",
    "baseline_method",
    "mode",
    "source_methods",
    "n_cases",
    "dino_to_target_mean",
    "dino_delta_mean",
    "dino_delta_ci95_low",
    "dino_delta_ci95_high",
    "dino_delta_ci90_low",
    "dino_delta_ci90_high",
    "win_rate",
    "win_rate_ci95_low",
    "win_rate_ci95_high",
    "clip_delta_mean",
    "clip_delta_ci95_low",
    "clip_delta_ci95_high",
    "copy_ssim_delta_mean",
    "copy_ssim_delta_ci95_low",
    "copy_ssim_delta_ci95_high",
    "copy_risk_count",
    "copy_risk_rate",
    "copy_risk_rate_ci95_low",
    "copy_risk_rate_ci95_high",
]

CASE_DELTA_FIELDS = [
    "group",
    "method",
    "baseline_method",
    "case_id",
    "subject_id",
    "category",
    "source_method",
    "dino_to_target",
    "baseline_dino_to_target",
    "dino_delta",
    "clip_text_image_sim",
    "baseline_clip_text_image_sim",
    "clip_delta",
    "ref_copy_ssim_max",
    "baseline_ref_copy_ssim_max",
    "copy_ssim_delta",
    "copy_risk",
]


def _float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    if value is None or value == "":
        return float(default)
    return float(value)


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return value


def _read_case_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fieldnames})


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _mean(values: list[float]) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else math.nan


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    stat: Callable[[list[dict[str, Any]]], float],
    *,
    iterations: int,
    seed: int,
    low_q: float = 0.025,
    high_q: float = 0.975,
) -> dict[str, float]:
    estimate = stat(rows)
    if not rows or iterations <= 0:
        return {"estimate": estimate, "low": estimate, "high": estimate}
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["subject_id"])].append(row)
    keys = list(clusters)
    rng = random.Random(seed)
    samples = []
    for _ in range(int(iterations)):
        sampled_rows: list[dict[str, Any]] = []
        for _idx in keys:
            sampled_rows.extend(clusters[rng.choice(keys)])
        samples.append(stat(sampled_rows))
    return {"estimate": estimate, "low": _percentile(samples, low_q), "high": _percentile(samples, high_q)}


def _index_by_case_method(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["case_id"]), str(row["method"])): row for row in rows}


def _rows_for_direct_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row, source_method=row["method"], aggregate_mode="direct") for row in rows if row["method"] in DIRECT_GROUPS]


def _aggregate_best(
    rows_by_case_method: dict[tuple[str, str], dict[str, Any]],
    *,
    method: str,
    source_methods: list[str],
) -> list[dict[str, Any]]:
    case_ids = sorted({case_id for case_id, source in rows_by_case_method if source in source_methods})
    out = []
    for case_id in case_ids:
        candidates = [rows_by_case_method[(case_id, source)] for source in source_methods if (case_id, source) in rows_by_case_method]
        if len(candidates) != len(source_methods):
            continue
        selected = max(candidates, key=lambda row: _float(row, "dino_sim_to_target"))
        out.append(dict(selected, method=method, source_method=selected["method"], aggregate_mode="best"))
    return out


def _aggregate_mean(
    rows_by_case_method: dict[tuple[str, str], dict[str, Any]],
    *,
    method: str,
    source_methods: list[str],
) -> list[dict[str, Any]]:
    case_ids = sorted({case_id for case_id, source in rows_by_case_method if source in source_methods})
    metric_fields = ["dino_sim_to_target", "clip_text_image_sim", "ref_copy_ssim_max"]
    out = []
    for case_id in case_ids:
        candidates = [rows_by_case_method[(case_id, source)] for source in source_methods if (case_id, source) in rows_by_case_method]
        if len(candidates) != len(source_methods):
            continue
        base = dict(candidates[0])
        base["method"] = method
        base["source_method"] = "+".join(source_methods)
        base["aggregate_mode"] = "mean"
        for field in metric_fields:
            base[field] = _mean([_float(row, field) for row in candidates])
        out.append(base)
    return out


def _build_aggregate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows_by_case_method = _index_by_case_method(rows)
    aggregates = []
    spec_by_method = {}
    for spec in AGGREGATE_SPECS:
        source_methods = list(spec["source_methods"])
        if spec["mode"] == "best":
            built = _aggregate_best(rows_by_case_method, method=spec["method"], source_methods=source_methods)
        elif spec["mode"] == "mean":
            built = _aggregate_mean(rows_by_case_method, method=spec["method"], source_methods=source_methods)
        else:
            raise ValueError(f"Unsupported aggregate mode: {spec['mode']}")
        if built:
            aggregates.extend(built)
            spec_by_method[spec["method"]] = dict(spec)
    return aggregates, spec_by_method


def _baseline_method_for(method: str, aggregate_specs: dict[str, dict[str, Any]]) -> str:
    if method in aggregate_specs:
        return str(aggregate_specs[method]["baseline_method"])
    return DEFAULT_BASELINE_METHOD


def _group_for(method: str, aggregate_specs: dict[str, dict[str, Any]]) -> str:
    if method in aggregate_specs:
        return str(aggregate_specs[method]["group"])
    return DIRECT_GROUPS.get(method, "unclassified")


def _mode_for(method: str, aggregate_specs: dict[str, dict[str, Any]]) -> str:
    if method in aggregate_specs:
        return str(aggregate_specs[method]["mode"])
    return "direct"


def _source_methods_for(method: str, aggregate_specs: dict[str, dict[str, Any]]) -> str:
    if method in aggregate_specs:
        return "|".join(str(value) for value in aggregate_specs[method]["source_methods"])
    return method


def _case_delta_rows(rows: list[dict[str, Any]], aggregate_specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_case_method = _index_by_case_method(rows)
    deltas = []
    for row in rows:
        method = str(row["method"])
        baseline_method = _baseline_method_for(method, aggregate_specs)
        baseline = rows_by_case_method.get((str(row["case_id"]), baseline_method))
        if baseline is None:
            continue
        dino = _float(row, "dino_sim_to_target")
        base_dino = _float(baseline, "dino_sim_to_target")
        clip = _float(row, "clip_text_image_sim")
        base_clip = _float(baseline, "clip_text_image_sim")
        copy = _float(row, "ref_copy_ssim_max")
        base_copy = _float(baseline, "ref_copy_ssim_max")
        copy_delta = copy - base_copy
        deltas.append(
            {
                "group": _group_for(method, aggregate_specs),
                "method": method,
                "baseline_method": baseline_method,
                "case_id": row["case_id"],
                "subject_id": row.get("subject_id", row["case_id"]),
                "category": row.get("category", ""),
                "source_method": row.get("source_method", method),
                "dino_to_target": dino,
                "baseline_dino_to_target": base_dino,
                "dino_delta": dino - base_dino,
                "clip_text_image_sim": clip,
                "baseline_clip_text_image_sim": base_clip,
                "clip_delta": clip - base_clip,
                "ref_copy_ssim_max": copy,
                "baseline_ref_copy_ssim_max": base_copy,
                "copy_ssim_delta": copy_delta,
                "copy_risk": int(copy_delta > COPY_RISK_DELTA_THRESHOLD),
            }
        )
    return deltas


def _summarize_method(
    method_rows: list[dict[str, Any]],
    *,
    method: str,
    aggregate_specs: dict[str, dict[str, Any]],
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    dino = _cluster_bootstrap(
        method_rows,
        lambda rows: _mean([_float(row, "dino_delta") for row in rows]),
        iterations=bootstrap_iterations,
        seed=seed,
    )
    dino90 = _cluster_bootstrap(
        method_rows,
        lambda rows: _mean([_float(row, "dino_delta") for row in rows]),
        iterations=bootstrap_iterations,
        seed=seed + 17,
        low_q=0.05,
        high_q=0.95,
    )
    win = _cluster_bootstrap(
        method_rows,
        lambda rows: _mean([1.0 if _float(row, "dino_delta") > 0.0 else 0.0 for row in rows]),
        iterations=bootstrap_iterations,
        seed=seed + 31,
    )
    clip = _cluster_bootstrap(
        method_rows,
        lambda rows: _mean([_float(row, "clip_delta") for row in rows]),
        iterations=bootstrap_iterations,
        seed=seed + 47,
    )
    copy = _cluster_bootstrap(
        method_rows,
        lambda rows: _mean([_float(row, "copy_ssim_delta") for row in rows]),
        iterations=bootstrap_iterations,
        seed=seed + 61,
    )
    copy_risk = _cluster_bootstrap(
        method_rows,
        lambda rows: _mean([1.0 if int(row["copy_risk"]) else 0.0 for row in rows]),
        iterations=bootstrap_iterations,
        seed=seed + 73,
    )
    return {
        "group": _group_for(method, aggregate_specs),
        "method": method,
        "baseline_method": _baseline_method_for(method, aggregate_specs),
        "mode": _mode_for(method, aggregate_specs),
        "source_methods": _source_methods_for(method, aggregate_specs),
        "n_cases": len(method_rows),
        "dino_to_target_mean": _mean([_float(row, "dino_to_target") for row in method_rows]),
        "dino_delta_mean": dino["estimate"],
        "dino_delta_ci95_low": dino["low"],
        "dino_delta_ci95_high": dino["high"],
        "dino_delta_ci90_low": dino90["low"],
        "dino_delta_ci90_high": dino90["high"],
        "win_rate": win["estimate"],
        "win_rate_ci95_low": win["low"],
        "win_rate_ci95_high": win["high"],
        "clip_delta_mean": clip["estimate"],
        "clip_delta_ci95_low": clip["low"],
        "clip_delta_ci95_high": clip["high"],
        "copy_ssim_delta_mean": copy["estimate"],
        "copy_ssim_delta_ci95_low": copy["low"],
        "copy_ssim_delta_ci95_high": copy["high"],
        "copy_risk_count": sum(1 for row in method_rows if int(row["copy_risk"])),
        "copy_risk_rate": copy_risk["estimate"],
        "copy_risk_rate_ci95_low": copy_risk["low"],
        "copy_risk_rate_ci95_high": copy_risk["high"],
    }


def analyze_phase2_metrics(
    *,
    case_metrics_path: str | Path,
    output_dir: str | Path,
    bootstrap_iterations: int = 10000,
    seed: int = 20260624,
) -> dict[str, Any]:
    case_metrics_path = Path(case_metrics_path)
    output_dir = Path(output_dir)
    rows = _read_case_metrics(case_metrics_path)
    direct_rows = _rows_for_direct_methods(rows)
    aggregate_rows, aggregate_specs = _build_aggregate_rows(rows)
    analysis_rows = direct_rows + aggregate_rows
    case_delta_rows = _case_delta_rows(analysis_rows, aggregate_specs)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_delta_rows:
        by_method[str(row["method"])].append(row)

    summary_rows = []
    for idx, method in enumerate(sorted(by_method, key=lambda name: (_group_for(name, aggregate_specs), name))):
        summary_rows.append(
            _summarize_method(
                by_method[method],
                method=method,
                aggregate_specs=aggregate_specs,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed + idx * 101,
            )
        )

    relative_summary_csv = output_dir / "v2_14_ablation_phase2_test48_relative_summary.csv"
    case_deltas_csv = output_dir / "v2_14_ablation_phase2_test48_case_deltas.csv"
    summary_json = output_dir / "v2_14_ablation_phase2_test48_summary.json"
    manifest_json = output_dir / "v2_14_ablation_phase2_test48_output_manifest.json"
    _write_csv(relative_summary_csv, summary_rows, RELATIVE_SUMMARY_FIELDS)
    _write_csv(case_deltas_csv, case_delta_rows, CASE_DELTA_FIELDS)

    result = {
        "analysis": "v2_14_phase2_new_generation_ablation",
        "case_metrics_path": str(case_metrics_path),
        "case_count": len({row["case_id"] for row in rows}),
        "input_method_count": len({row["method"] for row in rows}),
        "analysis_method_count": len(summary_rows),
        "copy_risk_delta_threshold": COPY_RISK_DELTA_THRESHOLD,
        "bootstrap_iterations": int(bootstrap_iterations),
        "seed": int(seed),
        "relative_summary_csv": str(relative_summary_csv),
        "case_deltas_csv": str(case_deltas_csv),
        "summary_rows": summary_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "summary_json": str(summary_json),
        "relative_summary_csv": str(relative_summary_csv),
        "case_deltas_csv": str(case_deltas_csv),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["summary_json"] = str(summary_json)
    result["manifest_json"] = str(manifest_json)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze v2.14 Phase 2 new-generation ablation metrics.")
    parser.add_argument("--case_metrics", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = analyze_phase2_metrics(
        case_metrics_path=args.case_metrics,
        output_dir=args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    print(json.dumps({key: result[key] for key in ["summary_json", "relative_summary_csv", "case_deltas_csv"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
