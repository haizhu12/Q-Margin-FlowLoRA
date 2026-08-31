from __future__ import annotations

import argparse
import json
import math
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


def _case_ids(rows: list[dict], baseline_method: str) -> list[str]:
    return sorted({str(row["case_id"]) for row in rows if str(row.get("method")) == baseline_method})


def _score(row: dict, *, clip_weight: float, copy_penalty: float, dino_copy_penalty: float) -> float:
    return (
        _float(row, "dino_sim_to_multi_ref_mean")
        + float(clip_weight) * _float(row, "clip_text_image_sim")
        - float(copy_penalty) * _float(row, "ref_copy_ssim_max")
        - float(dino_copy_penalty) * _float(row, "dino_ref_copy_sim_max")
    )


def predict_with_policy(
    rows: list[dict],
    *,
    baseline_method: str,
    candidate_methods: list[str],
    clip_weight: float,
    copy_penalty: float,
    dino_copy_penalty: float,
    min_score_gain: float,
) -> dict[str, str]:
    indexed = _index(rows)
    predictions = {}
    for case_id in _case_ids(rows, baseline_method):
        available = [method for method in candidate_methods if (case_id, method) in indexed]
        if not available:
            continue
        best_method = max(
            available,
            key=lambda method: _score(
                indexed[(case_id, method)],
                clip_weight=clip_weight,
                copy_penalty=copy_penalty,
                dino_copy_penalty=dino_copy_penalty,
            ),
        )
        baseline_score = _score(
            indexed[(case_id, baseline_method)],
            clip_weight=clip_weight,
            copy_penalty=copy_penalty,
            dino_copy_penalty=dino_copy_penalty,
        )
        best_score = _score(
            indexed[(case_id, best_method)],
            clip_weight=clip_weight,
            copy_penalty=copy_penalty,
            dino_copy_penalty=dino_copy_penalty,
        )
        if best_score - baseline_score < float(min_score_gain):
            best_method = baseline_method
        predictions[case_id] = best_method
    return predictions


def _evaluate(
    rows: list[dict],
    predictions: dict[str, str],
    *,
    baseline_method: str,
    run_label: str,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    summary, prediction_rows = evaluate_predictions(
        rows,
        predictions,
        baseline_method=baseline_method,
        run_label=run_label,
    )
    return {
        "summary": summary,
        "bootstrap_ci": bootstrap_ci(
            [float(row["dino_delta"]) for row in prediction_rows],
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "prediction_rows": prediction_rows,
        "selected_method_counts": dict(Counter(predictions.values())),
    }


def _aggregate_prediction_rows(rows: list[dict], *, bootstrap_iterations: int, seed: int) -> dict:
    summary = summarize_deltas(
        [float(row["dino_delta"]) for row in rows],
        [float(row["clip_delta"]) for row in rows],
        [float(row["copy_ssim_delta"]) for row in rows],
    )
    return {
        "summary": summary,
        "bootstrap_ci": bootstrap_ci(
            [float(row["dino_delta"]) for row in rows],
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "selected_method_counts": dict(Counter(str(row["selected_method"]) for row in rows)),
    }


def _policy_grid(
    *,
    clip_weights: list[float],
    copy_penalties: list[float],
    dino_copy_penalties: list[float],
    min_score_gains: list[float],
) -> list[dict[str, float]]:
    policies = []
    for clip_weight in clip_weights:
        for copy_penalty in copy_penalties:
            for dino_copy_penalty in dino_copy_penalties:
                for min_score_gain in min_score_gains:
                    policies.append(
                        {
                            "clip_weight": float(clip_weight),
                            "copy_penalty": float(copy_penalty),
                            "dino_copy_penalty": float(dino_copy_penalty),
                            "min_score_gain": float(min_score_gain),
                        }
                    )
    return policies


def _policy_key(policy: dict[str, float]) -> str:
    return (
        f"clip{policy['clip_weight']:+.3f}_"
        f"copy{policy['copy_penalty']:+.3f}_"
        f"dinocopy{policy.get('dino_copy_penalty', 0.0):+.3f}_"
        f"gain{policy['min_score_gain']:+.3f}"
    )


def _policy_row(policy: dict[str, float], result: dict[str, Any], *, split: str) -> dict[str, Any]:
    summary = result["summary"]
    ci = result["bootstrap_ci"]
    return {
        "split": split,
        "policy_key": _policy_key(policy),
        "clip_weight": policy["clip_weight"],
        "copy_penalty": policy["copy_penalty"],
        "dino_copy_penalty": policy.get("dino_copy_penalty", 0.0),
        "min_score_gain": policy["min_score_gain"],
        "case_count": summary["case_count"],
        "dino_delta_mean": summary["dino_delta_mean"],
        "dino_delta_median": summary["dino_delta_median"],
        "dino_win_rate": summary["dino_win_rate"],
        "clip_delta_mean": summary["clip_delta_mean"],
        "copy_ssim_delta_mean": summary["copy_ssim_delta_mean"],
        "copy_risk_count_delta_gt_0_015": summary["copy_risk_count_delta_gt_0_015"],
        "ci90_low": ci["ci90_low"],
        "ci90_high": ci["ci90_high"],
    }


def _passes_entry(row: dict[str, Any]) -> bool:
    return (
        float(row["dino_delta_mean"]) >= 0.010
        and float(row["dino_delta_median"]) >= 0.0
        and float(row["dino_win_rate"]) >= 0.55
        and float(row["clip_delta_mean"]) >= -0.003
        and float(row["copy_ssim_delta_mean"]) <= 0.005
        and float(row["ci90_low"]) >= -0.003
    )


def _choose_policy(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [row for row in train_rows if _passes_entry(row)]
    candidates = passing or train_rows
    return max(
        candidates,
        key=lambda row: (
            float(row["dino_delta_mean"]),
            -float(row["copy_risk_count_delta_gt_0_015"]),
            float(row["dino_win_rate"]),
        ),
    )


def analyze_rerank_policy_sweep(
    *,
    train_metrics: str | Path,
    eval_runs: list[tuple[str, str | Path]],
    baseline_method: str = "v2_8a_static_a064_d080",
    candidate_methods: list[str] | None = None,
    clip_weights: list[float] | None = None,
    copy_penalties: list[float] | None = None,
    dino_copy_penalties: list[float] | None = None,
    min_score_gains: list[float] | None = None,
    bootstrap_iterations: int = 2000,
    seed: int = 20260622,
) -> dict[str, Any]:
    candidate_methods = list(candidate_methods or DEFAULT_METHODS)
    clip_weights = list(clip_weights if clip_weights is not None else [-0.5, 0.0, 0.5, 1.0])
    copy_penalties = list(copy_penalties if copy_penalties is not None else [0.0, 0.25, 0.5, 1.0])
    dino_copy_penalties = list(dino_copy_penalties if dino_copy_penalties is not None else [0.0])
    min_score_gains = list(min_score_gains if min_score_gains is not None else [-999.0, 0.0, 0.005, 0.010])
    policies = _policy_grid(
        clip_weights=clip_weights,
        copy_penalties=copy_penalties,
        dino_copy_penalties=dino_copy_penalties,
        min_score_gains=min_score_gains,
    )
    train_rows_raw = read_csv(train_metrics)
    train_policy_rows = []
    train_results = {}
    for policy in policies:
        predictions = predict_with_policy(
            train_rows_raw,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            **policy,
        )
        result = _evaluate(
            train_rows_raw,
            predictions,
            baseline_method=baseline_method,
            run_label="train",
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        key = _policy_key(policy)
        train_results[key] = result
        train_policy_rows.append(_policy_row(policy, result, split="train"))
    selected_row = _choose_policy(train_policy_rows)
    selected_policy = {
        "clip_weight": float(selected_row["clip_weight"]),
        "copy_penalty": float(selected_row["copy_penalty"]),
        "dino_copy_penalty": float(selected_row["dino_copy_penalty"]),
        "min_score_gain": float(selected_row["min_score_gain"]),
        "policy_key": str(selected_row["policy_key"]),
    }

    per_run = []
    eval_prediction_rows = []
    for label, path in eval_runs:
        rows = read_csv(path)
        predictions = predict_with_policy(
            rows,
            baseline_method=baseline_method,
            candidate_methods=candidate_methods,
            clip_weight=selected_policy["clip_weight"],
            copy_penalty=selected_policy["copy_penalty"],
            dino_copy_penalty=selected_policy["dino_copy_penalty"],
            min_score_gain=selected_policy["min_score_gain"],
        )
        result = _evaluate(
            rows,
            predictions,
            baseline_method=baseline_method,
            run_label=label,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        per_run.append(
            {
                "label": label,
                "case_metrics": str(path),
                "summary": result["summary"],
                "bootstrap_ci": result["bootstrap_ci"],
                "selected_method_counts": result["selected_method_counts"],
            }
        )
        eval_prediction_rows.extend(result["prediction_rows"])
    eval_aggregate = _aggregate_prediction_rows(
        eval_prediction_rows,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    train_selected_result = train_results[selected_policy["policy_key"]]
    selected_policy.update(
        {
            "train_summary": train_selected_result["summary"],
            "train_bootstrap_ci": train_selected_result["bootstrap_ci"],
        }
    )
    eval_passed = _passes_entry(
        {
            **eval_aggregate["summary"],
            "ci90_low": eval_aggregate["bootstrap_ci"]["ci90_low"],
            "ci90_high": eval_aggregate["bootstrap_ci"]["ci90_high"],
        }
    )
    return {
        "baseline_method": baseline_method,
        "candidate_methods": candidate_methods,
        "train_metrics": str(train_metrics),
        "eval_runs": [{"label": label, "case_metrics": str(path)} for label, path in eval_runs],
        "selected_policy": selected_policy,
        "train_policy_rows": train_policy_rows,
        "per_run": per_run,
        "eval_aggregate": eval_aggregate,
        "decision": {
            "rerank_policy_ready": bool(eval_passed),
            "reason": (
                "train-selected target-free rerank policy passed heldout gate"
                if eval_passed
                else "train-selected target-free rerank policy did not pass heldout gate"
            ),
        },
        "_eval_prediction_rows": eval_prediction_rows,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if not args.eval_run:
        raise ValueError("At least one --eval_run LABEL CASE_METRICS is required.")
    result = analyze_rerank_policy_sweep(
        train_metrics=args.train_metrics,
        eval_runs=[(label, path) for label, path in args.eval_run],
        baseline_method=args.baseline_method,
        candidate_methods=list(args.candidate_methods or DEFAULT_METHODS),
        clip_weights=list(args.clip_weights),
        copy_penalties=list(args.copy_penalties),
        dino_copy_penalties=list(args.dino_copy_penalties),
        min_score_gains=list(args.min_score_gains),
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = result.pop("_eval_prediction_rows")
    write_csv(output_dir / "v2_14_rerank_policy_sweep_train_summary.csv", result["train_policy_rows"])
    write_csv(output_dir / "v2_14_rerank_policy_sweep_eval_predictions.csv", prediction_rows)
    (output_dir / "v2_14_rerank_policy_sweep_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], ensure_ascii=False))
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-select and evaluate target-free v2.14 rerank score policies.")
    parser.add_argument("--train_metrics", required=True)
    parser.add_argument("--eval_run", nargs=2, action="append", metavar=("LABEL", "CASE_METRICS"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_method", default="v2_8a_static_a064_d080")
    parser.add_argument("--candidate_methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--clip_weights", nargs="*", type=float, default=[-0.5, 0.0, 0.5, 1.0])
    parser.add_argument("--copy_penalties", nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--dino_copy_penalties", nargs="*", type=float, default=[0.0])
    parser.add_argument("--min_score_gains", nargs="*", type=float, default=[-999.0, 0.0, 0.005, 0.010])
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args(argv)


def main(argv=None) -> dict[str, Any]:
    return analyze(parse_args(argv))


if __name__ == "__main__":
    main()
