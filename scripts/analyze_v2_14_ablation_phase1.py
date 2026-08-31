from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_v2_14_compressibility import DEFAULT_METHODS  # noqa: E402


MetricRows = list[dict[str, Any]]
PredictionRows = list[dict[str, Any]]
Statistic = Callable[[PredictionRows], float]


TAU_DINO_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, math.inf]
COPY_RISK_DELTA_THRESHOLD = 0.015
DEFAULT_BASELINE = "v2_8a_static_a064_d080"


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


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


def _trimmed_mean(values: list[float], fraction: float = 0.10) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return math.nan
    trim = int(len(clean) * float(fraction))
    if trim and len(clean) > 2 * trim:
        clean = clean[trim : len(clean) - trim]
    return _mean(clean)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, float(q))) * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    weight = pos - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return math.nan
    if abs(float(denominator)) <= 1.0e-12:
        return math.nan
    return float(numerator) / float(denominator)


def _index(rows: MetricRows) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["case_id"]), str(row["method"])): row for row in rows}


def _case_ids(rows: MetricRows, baseline_method: str) -> list[str]:
    return sorted({str(row["case_id"]) for row in rows if str(row.get("method")) == baseline_method})


def _available_methods(
    indexed: dict[tuple[str, str], dict[str, Any]],
    case_id: str,
    candidate_methods: list[str],
) -> list[str]:
    return [method for method in candidate_methods if (case_id, method) in indexed]


def _subject_key(row: dict[str, Any]) -> str:
    return str(row.get("subject_id") or row.get("case_id") or "")


def _clusters(rows: PredictionRows) -> dict[str, PredictionRows]:
    grouped: dict[str, PredictionRows] = defaultdict(list)
    for row in rows:
        grouped[_subject_key(row)].append(row)
    return dict(grouped)


def _cluster_bootstrap(
    rows: PredictionRows,
    statistic: Statistic,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    if not rows:
        return {
            "estimate": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "ci90_low": math.nan,
            "ci90_high": math.nan,
        }
    grouped = _clusters(rows)
    subject_ids = sorted(grouped)
    estimate = float(statistic(rows))
    if not subject_ids:
        return {
            "estimate": estimate,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "ci90_low": math.nan,
            "ci90_high": math.nan,
        }
    rng = random.Random(int(seed))
    samples: list[float] = []
    for _ in range(int(iterations)):
        sampled_rows: PredictionRows = []
        for _ in subject_ids:
            sampled_rows.extend(grouped[rng.choice(subject_ids)])
        samples.append(float(statistic(sampled_rows)))
    samples = sorted(value for value in samples if math.isfinite(float(value)))
    return {
        "estimate": estimate,
        "ci95_low": _percentile(samples, 0.025),
        "ci95_high": _percentile(samples, 0.975),
        "ci90_low": _percentile(samples, 0.05),
        "ci90_high": _percentile(samples, 0.95),
    }


def _case_bootstrap_ci90(
    rows: PredictionRows,
    statistic: Statistic,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    if not rows:
        return {"ci90_low": math.nan, "ci90_high": math.nan}
    rng = random.Random(int(seed))
    samples: list[float] = []
    for _ in range(int(iterations)):
        sampled = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        samples.append(float(statistic(sampled)))
    samples = sorted(value for value in samples if math.isfinite(float(value)))
    return {
        "ci90_low": _percentile(samples, 0.05),
        "ci90_high": _percentile(samples, 0.95),
    }


def _stat_mean(key: str) -> Statistic:
    return lambda rows: _mean([float(row[key]) for row in rows])


def _stat_rate(predicate: Callable[[dict[str, Any]], bool]) -> Statistic:
    return lambda rows: _mean([1.0 if predicate(row) else 0.0 for row in rows])


def _stat_ratio(numerator_key: str, denominator_key: str) -> Statistic:
    def _inner(rows: PredictionRows) -> float:
        return _safe_div(
            _mean([float(row[numerator_key]) for row in rows]),
            _mean([float(row[denominator_key]) for row in rows]),
        )

    return _inner


def _json_counts(values: list[str]) -> str:
    return json.dumps(dict(Counter(values)), ensure_ascii=False, sort_keys=True)


def summarize_prediction_rows(
    rows: PredictionRows,
    *,
    table: str,
    variant: str,
    baseline_variant: str,
    bootstrap_iterations: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = dict(extra or {})
    dino = [float(row["dino_delta"]) for row in rows]
    clip_ci = _cluster_bootstrap(rows, _stat_mean("clip_delta"), iterations=bootstrap_iterations, seed=seed + 11)
    copy_ci = _cluster_bootstrap(rows, _stat_mean("copy_ssim_delta"), iterations=bootstrap_iterations, seed=seed + 13)
    dino_ci = _cluster_bootstrap(rows, _stat_mean("dino_delta"), iterations=bootstrap_iterations, seed=seed)
    r14_case_ci90 = _case_bootstrap_ci90(
        rows,
        _stat_mean("dino_delta"),
        iterations=bootstrap_iterations,
        seed=seed,
    )
    win_ci = _cluster_bootstrap(
        rows,
        _stat_rate(lambda row: float(row["dino_delta"]) > 0.0),
        iterations=bootstrap_iterations,
        seed=seed + 17,
    )
    risk_ci = _cluster_bootstrap(
        rows,
        _stat_rate(lambda row: float(row["copy_ssim_delta"]) > COPY_RISK_DELTA_THRESHOLD),
        iterations=bootstrap_iterations,
        seed=seed + 19,
    )
    trigger_ci = _cluster_bootstrap(
        rows,
        _stat_rate(lambda row: bool(row.get("triggered"))),
        iterations=bootstrap_iterations,
        seed=seed + 23,
    )
    false_ci = _cluster_bootstrap(
        rows,
        _stat_rate(lambda row: bool(row.get("false_intervention"))),
        iterations=bootstrap_iterations,
        seed=seed + 29,
    )
    row = {
        "table": table,
        "variant": variant,
        "baseline_variant": baseline_variant,
        "n_cases": len(rows),
        "n_subjects": len(_clusters(rows)),
        "estimate": dino_ci["estimate"],
        "ci_level": 0.95,
        "ci95_low": dino_ci["ci95_low"],
        "ci95_high": dino_ci["ci95_high"],
        "ci90_low": dino_ci["ci90_low"],
        "ci90_high": dino_ci["ci90_high"],
        "interval_method": "subject_clustered_bootstrap",
        "r14_case_bootstrap_ci90_low": r14_case_ci90["ci90_low"],
        "r14_case_bootstrap_ci90_high": r14_case_ci90["ci90_high"],
        "r14_compatibility_interval_method": "case_bootstrap",
        "resampling_unit": "subject_id",
        "n_bootstrap": int(bootstrap_iterations),
        "bootstrap_seed": int(seed),
        "dino_delta_median": _median(dino),
        "dino_delta_trimmed_mean_10pct": _trimmed_mean(dino, 0.10),
        "win_rate": win_ci["estimate"],
        "win_rate_ci95_low": win_ci["ci95_low"],
        "win_rate_ci95_high": win_ci["ci95_high"],
        "clip_delta_mean": clip_ci["estimate"],
        "clip_delta_ci95_low": clip_ci["ci95_low"],
        "clip_delta_ci95_high": clip_ci["ci95_high"],
        "copy_ssim_delta_mean": copy_ci["estimate"],
        "copy_ssim_delta_ci95_low": copy_ci["ci95_low"],
        "copy_ssim_delta_ci95_high": copy_ci["ci95_high"],
        "copy_risk_count": sum(1 for row_ in rows if float(row_["copy_ssim_delta"]) > COPY_RISK_DELTA_THRESHOLD),
        "copy_risk_rate": risk_ci["estimate"],
        "copy_risk_rate_ci95_low": risk_ci["ci95_low"],
        "copy_risk_rate_ci95_high": risk_ci["ci95_high"],
        "trigger_count": sum(1 for row_ in rows if bool(row_.get("triggered"))),
        "trigger_rate": trigger_ci["estimate"],
        "trigger_rate_ci95_low": trigger_ci["ci95_low"],
        "trigger_rate_ci95_high": trigger_ci["ci95_high"],
        "false_intervention_count": sum(1 for row_ in rows if bool(row_.get("false_intervention"))),
        "false_intervention_rate": false_ci["estimate"],
        "false_intervention_rate_ci95_low": false_ci["ci95_low"],
        "false_intervention_rate_ci95_high": false_ci["ci95_high"],
        "selected_method_counts": _json_counts([str(row_["selected_method"]) for row_ in rows]),
    }
    row.update(extra)
    return row


def _dino_ref_mean(row: dict[str, Any]) -> float:
    return _float(row, "dino_sim_to_multi_ref_mean")


def _dino_ref_max(row: dict[str, Any]) -> float:
    return _float(row, "dino_ref_copy_sim_max")


def _dino_ref_min(row: dict[str, Any]) -> float:
    mean = _dino_ref_mean(row)
    max_value = _dino_ref_max(row)
    if math.isfinite(mean) and math.isfinite(max_value):
        return 2.0 * mean - max_value
    return mean


def _dino_ref_median(row: dict[str, Any]) -> float:
    return _dino_ref_mean(row)


def _copy_ssim(row: dict[str, Any], agg: str) -> float:
    if agg == "mean":
        return _float(row, "ssim_to_multi_ref_mean", _float(row, "ref_copy_ssim_max"))
    return _float(row, "ref_copy_ssim_max")


def _dino_copy(row: dict[str, Any], agg: str) -> float:
    if agg == "mean":
        return _dino_ref_mean(row)
    return _float(row, "dino_ref_copy_sim_max")


def _guard_score(row: dict[str, Any], *, copy_agg: str = "max", dino_copy_agg: str = "max") -> float:
    return _dino_ref_mean(row) - 0.5 * _copy_ssim(row, copy_agg) - 0.25 * _dino_copy(row, dino_copy_agg)


def _primary_score(row: dict[str, Any], primary: str) -> float:
    if primary == "clip":
        return _float(row, "clip_text_image_sim")
    if primary == "dino_ref_max":
        return _dino_ref_max(row)
    if primary == "dino_ref_min":
        return _dino_ref_min(row)
    if primary == "dino_ref_median":
        return _dino_ref_median(row)
    return _dino_ref_mean(row)


def _select_by_score(
    indexed: dict[tuple[str, str], dict[str, Any]],
    case_id: str,
    methods: list[str],
    score: Callable[[dict[str, Any]], float],
) -> str:
    return max(methods, key=lambda method: score(indexed[(case_id, method)]))


def _stable_random_choice(case_id: str, methods: list[str], *, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{case_id}:{'|'.join(methods)}".encode("utf-8")).hexdigest()
    return methods[int(digest[:12], 16) % len(methods)]


def _select_hybrid(
    indexed: dict[tuple[str, str], dict[str, Any]],
    case_id: str,
    methods: list[str],
    *,
    baseline_method: str,
    primary: str = "dino_ref_mean",
    trigger_kind: str = "relative_dino_copy",
    fallback_kind: str = "guard_rerank",
    copy_agg: str = "max",
    dino_copy_agg: str = "max",
    tau_dino: float = 0.10,
    tau_ssim: float = COPY_RISK_DELTA_THRESHOLD,
    absolute_dino_copy_threshold: float = 0.85,
) -> tuple[str, dict[str, Any]]:
    primary_selected = _select_by_score(indexed, case_id, methods, lambda row: _primary_score(row, primary))
    baseline = indexed[(case_id, baseline_method)]
    selected_row = indexed[(case_id, primary_selected)]
    dino_copy_delta = _dino_copy(selected_row, "max") - _dino_copy(baseline, "max")
    ssim_delta = _copy_ssim(selected_row, "max") - _copy_ssim(baseline, "max")
    triggered = False
    if trigger_kind == "none":
        triggered = False
    elif trigger_kind == "always":
        triggered = True
    elif trigger_kind == "absolute_dino_copy":
        triggered = _dino_copy(selected_row, "max") > float(absolute_dino_copy_threshold)
    elif trigger_kind == "relative_ssim":
        triggered = ssim_delta > float(tau_ssim)
    elif trigger_kind == "dino_or_ssim":
        triggered = dino_copy_delta > float(tau_dino) or ssim_delta > float(tau_ssim)
    else:
        triggered = dino_copy_delta > float(tau_dino)

    if not triggered:
        return primary_selected, {
            "triggered": False,
            "primary_selected": primary_selected,
            "fallback_selected": "",
            "false_intervention": False,
        }

    if fallback_kind == "baseline":
        selected = baseline_method
    else:
        selected = _select_by_score(
            indexed,
            case_id,
            methods,
            lambda row: _guard_score(row, copy_agg=copy_agg, dino_copy_agg=dino_copy_agg),
        )
    base_target = _float(baseline, "dino_sim_to_target")
    primary_delta = _float(indexed[(case_id, primary_selected)], "dino_sim_to_target") - base_target
    selected_delta = _float(indexed[(case_id, selected)], "dino_sim_to_target") - base_target
    return selected, {
        "triggered": True,
        "primary_selected": primary_selected,
        "fallback_selected": selected,
        "false_intervention": bool(selected_delta < primary_delta),
    }


def _make_prediction_rows(
    rows: MetricRows,
    *,
    run_label: str,
    role: str,
    baseline_method: str,
    candidate_methods: list[str],
    selector_name: str,
    selector: Callable[[dict[tuple[str, str], dict[str, Any]], str, list[str]], tuple[str, dict[str, Any]]],
) -> PredictionRows:
    indexed = _index(rows)
    output: PredictionRows = []
    for case_id in _case_ids(rows, baseline_method):
        methods = _available_methods(indexed, case_id, candidate_methods)
        if baseline_method not in methods or not methods:
            continue
        selected_method, meta = selector(indexed, case_id, methods)
        baseline = indexed[(case_id, baseline_method)]
        selected = indexed[(case_id, selected_method)]
        dino_delta = _float(selected, "dino_sim_to_target") - _float(baseline, "dino_sim_to_target")
        clip_delta = _float(selected, "clip_text_image_sim") - _float(baseline, "clip_text_image_sim")
        copy_delta = _float(selected, "ref_copy_ssim_max") - _float(baseline, "ref_copy_ssim_max")
        output.append(
            {
                "run": run_label,
                "role": role,
                "case_id": case_id,
                "subject_id": baseline.get("subject_id", ""),
                "category": baseline.get("category", ""),
                "selector": selector_name,
                "baseline_method": baseline_method,
                "selected_method": selected_method,
                "primary_selected": meta.get("primary_selected", ""),
                "fallback_selected": meta.get("fallback_selected", ""),
                "triggered": bool(meta.get("triggered", False)),
                "false_intervention": bool(meta.get("false_intervention", False)),
                "dino_delta": dino_delta,
                "clip_delta": clip_delta,
                "copy_ssim_delta": copy_delta,
                "baseline_image_path": baseline.get("image_path", ""),
                "selected_image_path": selected.get("image_path", ""),
            }
        )
    return output


def _fixed_selector(method: str) -> Callable[[dict[tuple[str, str], dict[str, Any]], str, list[str]], tuple[str, dict[str, Any]]]:
    def _selector(indexed: dict[tuple[str, str], dict[str, Any]], case_id: str, methods: list[str]) -> tuple[str, dict[str, Any]]:
        selected = method if method in methods else methods[0]
        return selected, {"triggered": False, "primary_selected": selected}

    return _selector


def _oracle_selector(
    indexed: dict[tuple[str, str], dict[str, Any]],
    case_id: str,
    methods: list[str],
) -> tuple[str, dict[str, Any]]:
    selected = _select_by_score(indexed, case_id, methods, lambda row: _float(row, "dino_sim_to_target"))
    return selected, {"triggered": False, "primary_selected": selected}


def _selector_by_primary(primary: str, *, random_seed: int = 0) -> Callable:
    def _selector(indexed: dict[tuple[str, str], dict[str, Any]], case_id: str, methods: list[str]) -> tuple[str, dict[str, Any]]:
        if primary == "random":
            selected = _stable_random_choice(case_id, methods, seed=random_seed)
        else:
            selected = _select_by_score(indexed, case_id, methods, lambda row: _primary_score(row, primary))
        return selected, {"triggered": False, "primary_selected": selected}

    return _selector


def _official_hybrid_selector(
    *,
    baseline_method: str,
    primary: str = "dino_ref_mean",
    trigger_kind: str = "relative_dino_copy",
    fallback_kind: str = "guard_rerank",
    copy_agg: str = "max",
    dino_copy_agg: str = "max",
    tau_dino: float = 0.10,
    tau_ssim: float = COPY_RISK_DELTA_THRESHOLD,
    absolute_dino_copy_threshold: float = 0.85,
) -> Callable:
    def _selector(indexed: dict[tuple[str, str], dict[str, Any]], case_id: str, methods: list[str]) -> tuple[str, dict[str, Any]]:
        return _select_hybrid(
            indexed,
            case_id,
            methods,
            baseline_method=baseline_method,
            primary=primary,
            trigger_kind=trigger_kind,
            fallback_kind=fallback_kind,
            copy_agg=copy_agg,
            dino_copy_agg=dino_copy_agg,
            tau_dino=tau_dino,
            tau_ssim=tau_ssim,
            absolute_dino_copy_threshold=absolute_dino_copy_threshold,
        )

    return _selector


def _load_runs(runs: list[tuple[str, str, str | Path]]) -> list[dict[str, Any]]:
    loaded = []
    for label, role, path in runs:
        loaded.append({"label": str(label), "role": str(role), "path": str(path), "rows": read_csv(path)})
    return loaded


def _rows_for_roles(loaded_runs: list[dict[str, Any]], roles: set[str] | None = None) -> list[dict[str, Any]]:
    selected = []
    for run in loaded_runs:
        if roles is not None and str(run["role"]) not in roles:
            continue
        selected.extend(run["rows"])
    return selected


def _predict_across_runs(
    loaded_runs: list[dict[str, Any]],
    *,
    baseline_method: str,
    candidate_methods: list[str],
    selector_name: str,
    selector: Callable,
    roles: set[str] | None = None,
) -> PredictionRows:
    rows: PredictionRows = []
    for run in loaded_runs:
        if roles is not None and str(run["role"]) not in roles:
            continue
        rows.extend(
            _make_prediction_rows(
                run["rows"],
                run_label=run["label"],
                role=run["role"],
                baseline_method=baseline_method,
                candidate_methods=candidate_methods,
                selector_name=selector_name,
                selector=selector,
            )
        )
    return rows


def _candidate_budget_subsets(candidate_methods: list[str], baseline_method: str) -> list[tuple[str, list[str]]]:
    methods = [method for method in candidate_methods if method != baseline_method]
    if len(candidate_methods) >= 5 and set(DEFAULT_METHODS).issubset(set(candidate_methods)):
        return [
            ("A3-1__a064_only", [baseline_method]),
            ("A3-2a__a064_a000", [baseline_method, "v2_8a_static_a000_d144"]),
            ("A3-2b__a064_a032", [baseline_method, "v2_8a_static_a032_d112"]),
            ("A3-3__a064_a032_a000", [baseline_method, "v2_8a_static_a032_d112", "v2_8a_static_a000_d144"]),
            (
                "A3-4__a064_a048_a016_a000",
                [
                    baseline_method,
                    "v2_8a_static_a048_d096",
                    "v2_8a_static_a016_d128",
                    "v2_8a_static_a000_d144",
                ],
            ),
            ("A3-5__full_five_way", list(candidate_methods)),
        ]
    subsets: list[tuple[str, list[str]]] = [("A3-1__baseline_only", [baseline_method])]
    for size in range(2, len(candidate_methods) + 1):
        for idx, combo in enumerate(__import__("itertools").combinations(methods, size - 1)):
            subsets.append((f"A3-{size}__subset{idx}", [baseline_method, *combo]))
    return subsets


def _ratio_summary(
    pair_rows: PredictionRows,
    numerator_key: str,
    denominator_key: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    return _cluster_bootstrap(pair_rows, _stat_ratio(numerator_key, denominator_key), iterations=iterations, seed=seed)


def _paired_diff_rows(left_rows: PredictionRows, right_rows: PredictionRows, *, selector: str) -> PredictionRows:
    right_by_key = {(row["run"], row["case_id"]): row for row in right_rows}
    paired: PredictionRows = []
    for left in left_rows:
        right = right_by_key.get((left["run"], left["case_id"]))
        if not right:
            continue
        paired.append(
            {
                **left,
                "selector": selector,
                "selected_method": selector,
                "dino_delta": float(left["dino_delta"]) - float(right["dino_delta"]),
                "clip_delta": float(left["clip_delta"]) - float(right["clip_delta"]),
                "copy_ssim_delta": float(left["copy_ssim_delta"]) - float(right["copy_ssim_delta"]),
                "triggered": False,
                "false_intervention": False,
            }
        )
    return paired


def analyze_phase1_offline(
    *,
    runs: list[tuple[str, str, str | Path]],
    baseline_method: str = DEFAULT_BASELINE,
    candidate_methods: list[str] | None = None,
    bootstrap_iterations: int = 10000,
    seed: int = 20260624,
    random_seed: int = 20260624,
) -> dict[str, Any]:
    candidate_methods = list(candidate_methods or DEFAULT_METHODS)
    loaded_runs = _load_runs(runs)
    validation_labels = [run["label"] for run in loaded_runs if run["role"] == "validation"]
    heldout_labels = [run["label"] for run in loaded_runs if run["role"] == "heldout"]
    if not validation_labels:
        validation_labels = [run["label"] for run in loaded_runs]
    if not heldout_labels:
        heldout_labels = [run["label"] for run in loaded_runs]

    all_roles = None
    validation_roles = {"validation"} if any(run["role"] == "validation" for run in loaded_runs) else None
    heldout_roles = {"heldout"} if any(run["role"] == "heldout" for run in loaded_runs) else None
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions: dict[str, PredictionRows] = {}

    for method in candidate_methods:
        rows = _predict_across_runs(
            loaded_runs,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            selector_name=f"A1_fixed__{method}",
            selector=_fixed_selector(method),
            roles=all_roles,
        )
        tables["static_method_summary"].append(
            summarize_prediction_rows(
                rows,
                table="A1_static_method_summary",
                variant=method,
                baseline_variant=baseline_method,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                extra={"method": method, "n_candidates": 1},
            )
        )
        predictions[f"A1_fixed__{method}"] = rows

    validation_static_rows = []
    for method in candidate_methods:
        rows = _predict_across_runs(
            loaded_runs,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            selector_name=f"A2_validation_fixed__{method}",
            selector=_fixed_selector(method),
            roles=validation_roles,
        )
        validation_static_rows.append(
            summarize_prediction_rows(
                rows,
                table="A2_validation_static",
                variant=method,
                baseline_variant=baseline_method,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                extra={"method": method},
            )
        )
    selected_global = max(validation_static_rows, key=lambda row: float(row["estimate"]))["method"]
    global_rows = _predict_across_runs(
        loaded_runs,
        baseline_method=baseline_method,
        candidate_methods=candidate_methods,
        selector_name=f"A2_global_best_static__{selected_global}",
        selector=_fixed_selector(str(selected_global)),
        roles=heldout_roles,
    )
    official_rows_heldout = _predict_across_runs(
        loaded_runs,
        baseline_method=baseline_method,
        candidate_methods=candidate_methods,
        selector_name="locked_hybrid",
        selector=_official_hybrid_selector(baseline_method=baseline_method),
        roles=heldout_roles,
    )
    tables["global_best_static_summary"].append(
        summarize_prediction_rows(
            global_rows,
            table="A2_global_best_static",
            variant="global_best_static_on_heldout",
            baseline_variant=baseline_method,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            extra={"selected_on_validation_method": selected_global},
        )
    )
    tables["global_best_static_summary"].append(
        summarize_prediction_rows(
            _paired_diff_rows(official_rows_heldout, global_rows, selector="locked_hybrid_minus_global_best_static"),
            table="A2_global_best_static",
            variant="locked_hybrid_minus_global_best_static",
            baseline_variant=str(selected_global),
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            extra={"selected_on_validation_method": selected_global},
        )
    )
    predictions["A2_global_best_static"] = global_rows

    full_hybrid_rows = _predict_across_runs(
        loaded_runs,
        baseline_method=baseline_method,
        candidate_methods=candidate_methods,
        selector_name="locked_hybrid_full",
        selector=_official_hybrid_selector(baseline_method=baseline_method),
        roles=all_roles,
    )
    full_oracle_rows = _predict_across_runs(
        loaded_runs,
        baseline_method=baseline_method,
        candidate_methods=candidate_methods,
        selector_name="target_oracle_full",
        selector=_oracle_selector,
        roles=all_roles,
    )
    full_hybrid_estimate = _mean([float(row["dino_delta"]) for row in full_hybrid_rows])
    full_oracle_estimate = _mean([float(row["dino_delta"]) for row in full_oracle_rows])
    predictions["locked_hybrid_full"] = full_hybrid_rows
    predictions["target_oracle_full"] = full_oracle_rows

    for subset_key, subset_methods in _candidate_budget_subsets(candidate_methods, baseline_method):
        subset_rows = _predict_across_runs(
            loaded_runs,
            baseline_method=baseline_method,
            candidate_methods=subset_methods,
            selector_name=f"A3_subset__{subset_key}",
            selector=_official_hybrid_selector(baseline_method=baseline_method),
            roles=all_roles,
        )
        subset_oracle = _predict_across_runs(
            loaded_runs,
            baseline_method=baseline_method,
            candidate_methods=subset_methods,
            selector_name=f"A3_oracle__{subset_key}",
            selector=_oracle_selector,
            roles=all_roles,
        )
        pairs = []
        oracle_by_key = {(row["run"], row["case_id"]): row for row in subset_oracle}
        for row in subset_rows:
            oracle = oracle_by_key.get((row["run"], row["case_id"]))
            if oracle:
                pairs.append({**row, "selected_delta": row["dino_delta"], "oracle_delta": oracle["dino_delta"]})
        oracle_recovery = _ratio_summary(
            pairs,
            "selected_delta",
            "oracle_delta",
            iterations=bootstrap_iterations,
            seed=seed + 101,
        )
        retained_rows = [{**row, "selected_delta": row["dino_delta"], "full_delta": full_hybrid_estimate} for row in subset_rows]
        retained = _ratio_summary(
            retained_rows,
            "selected_delta",
            "full_delta",
            iterations=bootstrap_iterations,
            seed=seed + 103,
        )
        row = summarize_prediction_rows(
            subset_rows,
            table="A3_candidate_subset",
            variant=subset_key,
            baseline_variant=baseline_method,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            extra={
                "candidate_methods": "|".join(subset_methods),
                "n_candidates": len(subset_methods),
                "oracle_estimate": _mean([float(row_["dino_delta"]) for row_ in subset_oracle]),
                "oracle_recovery_estimate": oracle_recovery["estimate"],
                "oracle_recovery_ci95_low": oracle_recovery["ci95_low"],
                "oracle_recovery_ci95_high": oracle_recovery["ci95_high"],
                "retained_fraction_vs_full_estimate": retained["estimate"],
                "retained_fraction_vs_full_ci95_low": retained["ci95_low"],
                "retained_fraction_vs_full_ci95_high": retained["ci95_high"],
                "full_hybrid_estimate": full_hybrid_estimate,
                "full_oracle_estimate": full_oracle_estimate,
            },
        )
        tables["candidate_subset_summary"].append(row)
        predictions[f"A3_subset__{subset_key}"] = subset_rows

    selector_variants = [
        ("E1-0__random_candidate", _selector_by_primary("random", random_seed=random_seed)),
        ("E1-1__clip_t_only", _selector_by_primary("clip")),
        ("E1-2__dino_ref_max", _selector_by_primary("dino_ref_max")),
        ("E1-3__dino_ref_mean_pure_refmax", _selector_by_primary("dino_ref_mean")),
        ("E1-4__dino_ref_min", _selector_by_primary("dino_ref_min")),
        ("E1-5__dino_ref_median", _selector_by_primary("dino_ref_median")),
    ]
    oracle_by_key = {(row["run"], row["case_id"]): row for row in full_oracle_rows}
    for name, selector in selector_variants:
        rows = _predict_across_runs(
            loaded_runs,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            selector_name=name,
            selector=selector,
            roles=all_roles,
        )
        regret_rows = []
        for row in rows:
            oracle = oracle_by_key.get((row["run"], row["case_id"]))
            if oracle:
                regret_rows.append({**row, "dino_delta": float(oracle["dino_delta"]) - float(row["dino_delta"])})
        regret = _cluster_bootstrap(regret_rows, _stat_mean("dino_delta"), iterations=bootstrap_iterations, seed=seed + 107)
        tables["selector_signal_summary"].append(
            summarize_prediction_rows(
                rows,
                table="E1_selector_signal",
                variant=name,
                baseline_variant=baseline_method,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                extra={
                    "regret_to_target_oracle_estimate": regret["estimate"],
                    "regret_to_target_oracle_ci95_low": regret["ci95_low"],
                    "regret_to_target_oracle_ci95_high": regret["ci95_high"],
                    "median_note": "dino_ref_median equals dino_ref_mean for the two-reference metric cache",
                },
            )
        )
        predictions[name] = rows

    aggregation_variants = [
        (
            "E2_formal_role_split_mean_primary_max_copy",
            _official_hybrid_selector(baseline_method=baseline_method, primary="dino_ref_mean", copy_agg="max", dino_copy_agg="max"),
        ),
        (
            "E2_max_primary_max_copy",
            _official_hybrid_selector(baseline_method=baseline_method, primary="dino_ref_max", copy_agg="max", dino_copy_agg="max"),
        ),
        (
            "E2_mean_primary_mean_copy",
            _official_hybrid_selector(baseline_method=baseline_method, primary="dino_ref_mean", copy_agg="mean", dino_copy_agg="mean"),
        ),
        (
            "E2_min_primary_max_copy",
            _official_hybrid_selector(baseline_method=baseline_method, primary="dino_ref_min", copy_agg="max", dino_copy_agg="max"),
        ),
    ]
    for name, selector in aggregation_variants:
        rows = _predict_across_runs(
            loaded_runs,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            selector_name=name,
            selector=selector,
            roles=all_roles,
        )
        tables["aggregation_role_summary"].append(
            summarize_prediction_rows(
                rows,
                table="E2_aggregation_roles",
                variant=name,
                baseline_variant=baseline_method,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
        )
        predictions[name] = rows

    guard_variants = [
        ("E3-0__pure_refmax_no_guard", _official_hybrid_selector(baseline_method=baseline_method, trigger_kind="none")),
        ("E3-1__always_guarded_score", _official_hybrid_selector(baseline_method=baseline_method, trigger_kind="always")),
        (
            "E3-2__absolute_dino_copy_trigger",
            _official_hybrid_selector(baseline_method=baseline_method, trigger_kind="absolute_dino_copy"),
        ),
        ("E3-3__relative_dino_copy_trigger_official", _official_hybrid_selector(baseline_method=baseline_method)),
        (
            "E3-4__relative_ssim_trigger_only",
            _official_hybrid_selector(baseline_method=baseline_method, trigger_kind="relative_ssim"),
        ),
        (
            "E3-5__dino_or_ssim_trigger",
            _official_hybrid_selector(baseline_method=baseline_method, trigger_kind="dino_or_ssim"),
        ),
        (
            "E3-6__trigger_then_fallback_baseline",
            _official_hybrid_selector(baseline_method=baseline_method, fallback_kind="baseline"),
        ),
        ("E3-7__trigger_then_guard_rerank_official", _official_hybrid_selector(baseline_method=baseline_method)),
    ]
    for name, selector in guard_variants:
        rows = _predict_across_runs(
            loaded_runs,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            selector_name=name,
            selector=selector,
            roles=all_roles,
        )
        tables["guard_strategy_summary"].append(
            summarize_prediction_rows(
                rows,
                table="E3_guard_strategy",
                variant=name,
                baseline_variant=baseline_method,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
        )
        predictions[name] = rows

    validation_threshold_rows = []
    heldout_threshold_rows = []
    for tau in TAU_DINO_GRID:
        label = "+inf" if math.isinf(tau) else f"{tau:.3f}"
        selector = _official_hybrid_selector(baseline_method=baseline_method, tau_dino=tau)
        rows_val = _predict_across_runs(
            loaded_runs,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            selector_name=f"E4_tau_{label}",
            selector=selector,
            roles=validation_roles,
        )
        rows_heldout = _predict_across_runs(
            loaded_runs,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            selector_name=f"E4_tau_{label}",
            selector=selector,
            roles=heldout_roles,
        )
        validation_threshold_rows.append(
            summarize_prediction_rows(
                rows_val,
                table="E4_threshold_validation",
                variant=f"tau_dino={label}",
                baseline_variant=baseline_method,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                extra={"tau_dino": label, "official_tau": label == "0.100"},
            )
        )
        heldout_threshold_rows.append(
            summarize_prediction_rows(
                rows_heldout,
                table="E4_threshold_heldout",
                variant=f"tau_dino={label}",
                baseline_variant=baseline_method,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
                extra={"tau_dino": label, "official_tau": label == "0.100"},
            )
        )
    for row in validation_threshold_rows:
        row["validation_score"] = float(row["estimate"]) - 0.50 * max(float(row["copy_ssim_delta_mean"]), 0.0) - 0.01 * float(
            row["copy_risk_rate"]
        )
    selected_tau = max(validation_threshold_rows, key=lambda row: float(row["validation_score"]))["tau_dino"]
    for row in validation_threshold_rows:
        row["validation_selected_tau"] = row["tau_dino"] == selected_tau
    for row in heldout_threshold_rows:
        row["validation_selected_tau"] = row["tau_dino"] == selected_tau
    tables["threshold_validation_summary"].extend(validation_threshold_rows)
    tables["threshold_heldout_summary"].extend(heldout_threshold_rows)

    official_rows = predictions["E3-7__trigger_then_guard_rerank_official"]
    tables["split_breakdown_summary"].extend(
        summarize_prediction_rows(
            [row for row in official_rows if row["run"] == label],
            table="split_breakdown",
            variant=label,
            baseline_variant=baseline_method,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        for label in sorted({str(row["run"]) for row in official_rows})
    )
    tables["category_breakdown_summary"].extend(
        summarize_prediction_rows(
            [row for row in official_rows if row["category"] == category],
            table="category_breakdown",
            variant=category,
            baseline_variant=baseline_method,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        for category in sorted({str(row["category"]) for row in official_rows})
    )
    tables["selected_split_breakdown_summary"].extend(
        summarize_prediction_rows(
            [row for row in official_rows if row["selected_method"] == method],
            table="selected_split_breakdown",
            variant=method,
            baseline_variant=baseline_method,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        for method in sorted({str(row["selected_method"]) for row in official_rows})
    )
    tables["guard_trigger_breakdown_summary"].extend(
        summarize_prediction_rows(
            [row for row in official_rows if bool(row["triggered"]) == triggered],
            table="guard_trigger_breakdown",
            variant=f"triggered={triggered}",
            baseline_variant=baseline_method,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        for triggered in [False, True]
    )

    sorted_worst = sorted(official_rows, key=lambda row: float(row["dino_delta"]))
    sorted_best = sorted(official_rows, key=lambda row: float(row["dino_delta"]), reverse=True)
    copy_risk = [row for row in official_rows if float(row["copy_ssim_delta"]) > COPY_RISK_DELTA_THRESHOLD]
    return {
        "metadata": {
            "version": "v2.14",
            "analysis": "phase1_offline_ablation",
            "baseline_method": baseline_method,
            "candidate_methods": candidate_methods,
            "runs": [{"label": run["label"], "role": run["role"], "case_metrics": run["path"]} for run in loaded_runs],
            "validation_labels": validation_labels,
            "heldout_labels": heldout_labels,
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(seed),
            "interval_method": "subject_clustered_bootstrap",
            "copy_risk_delta_threshold": COPY_RISK_DELTA_THRESHOLD,
            "selected_tau_from_validation": selected_tau,
        },
        "tables": {key: list(value) for key, value in tables.items()},
        "predictions": predictions,
        "case_lists": {
            "copy_risk_cases": copy_risk,
            "worst_cases": sorted_worst[:24],
            "best_cases": sorted_best[:24],
        },
    }


def _table_by_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return None


def _write_result_md(result: dict[str, Any], path: Path, written: dict[str, str]) -> None:
    meta = result["metadata"]
    guard_rows = result["tables"].get("guard_strategy_summary", [])
    official = _table_by_variant(guard_rows, "E3-7__trigger_then_guard_rerank_official") or {}
    pure = _table_by_variant(guard_rows, "E3-0__pure_refmax_no_guard") or {}
    global_rows = result["tables"].get("global_best_static_summary", [])
    global_static = global_rows[0] if global_rows else {}
    selected_tau = meta.get("selected_tau_from_validation", "")
    lines = [
        "# v2.14 Phase 1 Offline Ablation Result",
        "",
        f"Date: 2026-06-24",
        "",
        "## Scope",
        "",
        "This result uses existing R14 metric caches only. No new images were generated.",
        "",
        f"- Cases: {official.get('n_cases', '')}",
        f"- Subjects: {official.get('n_subjects', '')}",
        f"- Bootstrap: subject-clustered, n={meta['bootstrap_iterations']}, seed={meta['bootstrap_seed']}",
        f"- Default interval: 95% CI; 90% CI is retained for R14 compatibility.",
        "",
        "## Key Results",
        "",
        "| Item | Estimate | 95% CI | Notes |",
        "|---|---:|---:|---|",
        (
            "| Locked hybrid vs baseline | "
            f"{float(official.get('estimate', math.nan)):+.6f} | "
            f"[{float(official.get('ci95_low', math.nan)):+.6f}, {float(official.get('ci95_high', math.nan)):+.6f}] | "
            f"win={float(official.get('win_rate', math.nan)):.3f}, copy-risk={official.get('copy_risk_count', '')}/{official.get('n_cases', '')} |"
        ),
        (
            "| Pure RefMax no guard | "
            f"{float(pure.get('estimate', math.nan)):+.6f} | "
            f"[{float(pure.get('ci95_low', math.nan)):+.6f}, {float(pure.get('ci95_high', math.nan)):+.6f}] | "
            f"copy-risk={pure.get('copy_risk_count', '')}/{pure.get('n_cases', '')} |"
        ),
        (
            "| Global-Best-Static heldout | "
            f"{float(global_static.get('estimate', math.nan)):+.6f} | "
            f"[{float(global_static.get('ci95_low', math.nan)):+.6f}, {float(global_static.get('ci95_high', math.nan)):+.6f}] | "
            f"selected={global_static.get('selected_on_validation_method', '')} |"
        ),
        f"| Validation-selected tau |  |  | tau_dino={selected_tau}; official tau=0.100 is still reported separately |",
        "",
        "## Output Files",
        "",
    ]
    for key, value in sorted(written.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Treat variants whose paired 95% CI crosses zero as inconclusive.",
            "- Use the validation threshold curve for threshold rationale; heldout rows are report-only.",
            "- Phase 2 new-generation ablations should wait until this result document has been reviewed.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_outputs(result: dict[str, Any], figures_dir: Path) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local plotting stack
        written["plot_warning"] = f"matplotlib unavailable: {exc}"
        return written

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    subset_rows = sorted(result["tables"].get("candidate_subset_summary", []), key=lambda row: (int(row["n_candidates"]), str(row["variant"])))
    if subset_rows:
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        xs = [int(row["n_candidates"]) for row in subset_rows]
        ys = [float(row["estimate"]) for row in subset_rows]
        yerr = [
            [max(0.0, y - float(row["ci95_low"])) for y, row in zip(ys, subset_rows)],
            [max(0.0, float(row["ci95_high"]) - y) for y, row in zip(ys, subset_rows)],
        ]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o", color="#2A6FBB", ecolor="#91B7DE", elinewidth=0.8, capsize=2)
        ax.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--")
        ax.set_xlabel("# generated candidates")
        ax.set_ylabel("DINO-Tgt delta vs baseline")
        ax.set_title("Candidate budget Pareto")
        fig.tight_layout()
        out = figures_dir / "v2_14_ablation_candidate_budget_pareto_20260624.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written["candidate_budget_pareto_png"] = str(out)

    val_rows = result["tables"].get("threshold_validation_summary", [])
    heldout_rows = result["tables"].get("threshold_heldout_summary", [])
    if val_rows and heldout_rows:
        def _tau_value(row: dict[str, Any]) -> float:
            return 0.225 if str(row["tau_dino"]) == "+inf" else float(row["tau_dino"])

        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        for rows, label, color in [(val_rows, "validation", "#2A6FBB"), (heldout_rows, "heldout", "#C44E52")]:
            ordered = sorted(rows, key=_tau_value)
            ax.plot([_tau_value(row) for row in ordered], [float(row["estimate"]) for row in ordered], marker="o", linewidth=1.2, label=label, color=color)
        ax.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--")
        ax.set_xlabel("DINO-copy relative threshold")
        ax.set_ylabel("DINO-Tgt delta vs baseline")
        ax.set_title("Guard threshold sweep")
        ax.legend()
        fig.tight_layout()
        out = figures_dir / "v2_14_ablation_guard_threshold_curve_20260624.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written["guard_threshold_curve_png"] = str(out)
    return written


def write_phase1_outputs(result: dict[str, Any], *, output_dir: str | Path, figures_dir: str | Path | None = None) -> dict[str, str]:
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir) if figures_dir is not None else output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    summary = {key: value for key, value in result.items() if key not in {"predictions"}}
    summary_json = output_dir / "v2_14_ablation_phase1_offline_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["summary_json"] = str(summary_json)
    for table_name, rows in result["tables"].items():
        path = output_dir / f"v2_14_ablation_phase1_{table_name}.csv"
        write_csv(path, rows)
        written[f"{table_name}_csv"] = str(path)
    all_table_rows = []
    for rows in result["tables"].values():
        all_table_rows.extend(rows)
    all_tables = output_dir / "v2_14_ablation_phase1_offline_tables.csv"
    write_csv(all_tables, all_table_rows)
    written["all_tables_csv"] = str(all_tables)
    official_rows = result["predictions"].get("E3-7__trigger_then_guard_rerank_official", result["predictions"].get("locked_hybrid_full", []))
    predictions_csv = output_dir / "v2_14_ablation_phase1_official_locked_hybrid_predictions.csv"
    write_csv(predictions_csv, official_rows)
    written["official_predictions_csv"] = str(predictions_csv)
    for list_name, rows in result["case_lists"].items():
        path = output_dir / f"v2_14_ablation_phase1_{list_name}.csv"
        write_csv(path, rows)
        written[f"{list_name}_csv"] = str(path)
    written.update(_plot_outputs(result, figures_dir))
    result_md = output_dir / "v2_14_ablation_phase1_offline_result_20260624.md"
    _write_result_md(result, result_md, written)
    written["result_md"] = str(result_md)
    manifest = output_dir / "v2_14_ablation_phase1_output_manifest.json"
    manifest.write_text(json.dumps(written, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["manifest_json"] = str(manifest)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2.14 Phase 1 offline ablations with paper-grade intervals.")
    parser.add_argument("--run", nargs=3, action="append", metavar=("LABEL", "ROLE", "CASE_METRICS"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--figures_dir", default=None)
    parser.add_argument("--baseline_method", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate_methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--random_seed", type=int, default=20260624)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if not args.run:
        raise ValueError("At least one --run LABEL ROLE CASE_METRICS is required.")
    result = analyze_phase1_offline(
        runs=[(label, role, metrics) for label, role, metrics in args.run],
        baseline_method=args.baseline_method,
        candidate_methods=list(args.candidate_methods or DEFAULT_METHODS),
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
        random_seed=int(args.random_seed),
    )
    written = write_phase1_outputs(result, output_dir=args.output_dir, figures_dir=args.figures_dir)
    print(json.dumps({"summary_json": written["summary_json"], "result_md": written["result_md"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
