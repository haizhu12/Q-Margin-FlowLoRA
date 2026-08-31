from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_v2_14_compressibility import (  # noqa: E402
    DEFAULT_METHODS,
    _float,
    _index,
    bootstrap_ci,
    evaluate_predictions,
    read_csv,
    summarize_deltas,
    write_csv,
)


def _case_ids(rows: list[dict[str, Any]], baseline_method: str) -> list[str]:
    return sorted({str(row["case_id"]) for row in rows if str(row.get("method")) == baseline_method})


def _refmax_score(row: dict[str, Any]) -> float:
    return _float(row, "dino_sim_to_multi_ref_mean")


def _guard_score(row: dict[str, Any]) -> float:
    return (
        _float(row, "dino_sim_to_multi_ref_mean")
        - 0.5 * _float(row, "ref_copy_ssim_max")
        - 0.25 * _float(row, "dino_ref_copy_sim_max")
    )


def predict_with_refmax_guard_hybrid(
    rows: list[dict[str, Any]],
    *,
    baseline_method: str,
    candidate_methods: list[str],
    copy_delta_threshold: float,
    dino_copy_delta_threshold: float,
) -> dict[str, str]:
    indexed = _index(rows)
    predictions = {}
    for case_id in _case_ids(rows, baseline_method):
        available = [method for method in candidate_methods if (case_id, method) in indexed]
        if not available:
            continue
        baseline = indexed[(case_id, baseline_method)]
        selected = max(available, key=lambda method: _refmax_score(indexed[(case_id, method)]))
        selected_row = indexed[(case_id, selected)]
        copy_delta = _float(selected_row, "ref_copy_ssim_max") - _float(baseline, "ref_copy_ssim_max")
        dino_copy_delta = _float(selected_row, "dino_ref_copy_sim_max") - _float(
            baseline, "dino_ref_copy_sim_max"
        )
        if copy_delta > float(copy_delta_threshold) or dino_copy_delta > float(dino_copy_delta_threshold):
            selected = max(available, key=lambda method: _guard_score(indexed[(case_id, method)]))
        predictions[case_id] = selected
    return predictions


def _aggregate_prediction_rows(rows: list[dict[str, Any]], *, bootstrap_iterations: int, seed: int) -> dict[str, Any]:
    dino = [float(row["dino_delta"]) for row in rows]
    clip = [float(row["clip_delta"]) for row in rows]
    copy = [float(row["copy_ssim_delta"]) for row in rows]
    return {
        "summary": summarize_deltas(dino, clip, copy),
        "bootstrap_ci": bootstrap_ci(dino, iterations=bootstrap_iterations, seed=seed),
        "selected_method_counts": dict(Counter(str(row["selected_method"]) for row in rows)),
    }


def analyze_refmax_guard_hybrid(
    *,
    eval_runs: list[tuple[str, str | Path]],
    baseline_method: str = "v2_8a_static_a064_d080",
    candidate_methods: list[str] | None = None,
    copy_delta_threshold: float = 999.0,
    dino_copy_delta_threshold: float = 0.10,
    bootstrap_iterations: int = 2000,
    seed: int = 20260622,
) -> dict[str, Any]:
    candidate_methods = list(candidate_methods or DEFAULT_METHODS)
    per_run = []
    prediction_rows = []
    for label, path in eval_runs:
        rows = read_csv(path)
        predictions = predict_with_refmax_guard_hybrid(
            rows,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            copy_delta_threshold=copy_delta_threshold,
            dino_copy_delta_threshold=dino_copy_delta_threshold,
        )
        summary, run_prediction_rows = evaluate_predictions(
            rows,
            predictions,
            baseline_method=baseline_method,
            run_label=label,
        )
        ci = bootstrap_ci(
            [float(row["dino_delta"]) for row in run_prediction_rows],
            iterations=bootstrap_iterations,
            seed=seed,
        )
        per_run.append(
            {
                "label": label,
                "case_metrics": str(path),
                "summary": summary,
                "bootstrap_ci": ci,
                "selected_method_counts": dict(Counter(predictions.values())),
            }
        )
        prediction_rows.extend(run_prediction_rows)

    aggregate = _aggregate_prediction_rows(
        prediction_rows,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    passed = (
        aggregate["summary"]["dino_delta_mean"] >= 0.024
        and aggregate["bootstrap_ci"]["ci90_low"] >= 0.015
        and aggregate["summary"]["dino_win_rate"] >= 0.62
        and aggregate["summary"]["copy_ssim_delta_mean"] <= 0.003
    )
    return {
        "version": "v2.14",
        "analysis": "refmax_guard_hybrid_selector",
        "baseline_method": baseline_method,
        "candidate_methods": candidate_methods,
        "selector": {
            "kind": "refmax_with_dino_copy_guard_fallback",
            "copy_delta_threshold": float(copy_delta_threshold),
            "dino_copy_delta_threshold": float(dino_copy_delta_threshold),
            "guard_score": "dino_sim_to_multi_ref_mean - 0.5 * ref_copy_ssim_max - 0.25 * dino_ref_copy_sim_max",
            "target_free": True,
        },
        "eval_runs": [{"label": label, "case_metrics": str(path)} for label, path in eval_runs],
        "per_run": per_run,
        "aggregate": aggregate,
        "decision": {
            "hybrid_selector_ready": bool(passed),
            "selected_output_lora_longtrain_allowed": False,
        },
        "_prediction_rows": prediction_rows,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if not args.eval_run:
        raise ValueError("At least one --eval_run LABEL CASE_METRICS is required.")
    result = analyze_refmax_guard_hybrid(
        eval_runs=[(label, path) for label, path in args.eval_run],
        baseline_method=args.baseline_method,
        candidate_methods=list(args.candidate_methods or DEFAULT_METHODS),
        copy_delta_threshold=float(args.copy_delta_threshold),
        dino_copy_delta_threshold=float(args.dino_copy_delta_threshold),
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = result.pop("_prediction_rows")
    summary_json = output_dir / "v2_14_refmax_guard_hybrid_summary.json"
    predictions_csv = output_dir / "v2_14_refmax_guard_hybrid_predictions.csv"
    summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(predictions_csv, prediction_rows)
    print(json.dumps(result["decision"], ensure_ascii=False))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v2.14 RefMax + dino-copy guard hybrid selector.")
    parser.add_argument("--eval_run", nargs=2, action="append", metavar=("LABEL", "CASE_METRICS"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_method", default="v2_8a_static_a064_d080")
    parser.add_argument("--candidate_methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--copy_delta_threshold", type=float, default=999.0)
    parser.add_argument("--dino_copy_delta_threshold", type=float, default=0.10)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return analyze(parse_args(argv))


if __name__ == "__main__":
    main()
