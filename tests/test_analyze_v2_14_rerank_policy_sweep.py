import csv
import math
from pathlib import Path

from scripts.analyze_v2_14_rerank_policy_sweep import analyze_rerank_policy_sweep


def _write_metrics(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    case_id: str,
    method: str,
    *,
    target: float,
    ref: float,
    clip: float,
    copy: float,
    dino_copy: float | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "subject_id": f"subject_{case_id}",
        "category": "synthetic",
        "method": method,
        "prompt": "synthetic prompt",
        "dino_sim_to_target": str(target),
        "dino_sim_to_multi_ref_mean": str(ref),
        "clip_text_image_sim": str(clip),
        "ref_copy_ssim_max": str(copy),
        "dino_ref_copy_sim_max": str(copy if dino_copy is None else dino_copy),
    }


def test_train_selected_copy_penalty_policy_generalizes_to_eval(tmp_path):
    baseline = "base"
    methods = ["base", "copy", "safe"]
    train_rows = []
    for idx in range(3):
        train_rows.extend(
            [
                _row(f"train{idx}", "base", target=0.50, ref=0.40, clip=0.20, copy=0.10),
                _row(f"train{idx}", "copy", target=0.54, ref=0.80, clip=0.20, copy=0.70),
                _row(f"train{idx}", "safe", target=0.62, ref=0.70, clip=0.20, copy=0.12),
            ]
        )
    eval_rows = [
        _row("eval0", "base", target=0.50, ref=0.40, clip=0.20, copy=0.10),
        _row("eval0", "copy", target=0.52, ref=0.80, clip=0.20, copy=0.75),
        _row("eval0", "safe", target=0.64, ref=0.70, clip=0.20, copy=0.11),
    ]

    result = analyze_rerank_policy_sweep(
        train_metrics=_write_metrics(tmp_path / "train.csv", train_rows),
        eval_runs=[("eval", _write_metrics(tmp_path / "eval.csv", eval_rows))],
        baseline_method=baseline,
        candidate_methods=methods,
        copy_penalties=[0.0, 0.5, 1.0],
        clip_weights=[0.0],
        min_score_gains=[-999.0],
        bootstrap_iterations=20,
    )

    selected = result["selected_policy"]
    assert selected["copy_penalty"] > 0.0
    eval_summary = result["eval_aggregate"]["summary"]
    assert math.isclose(eval_summary["dino_delta_mean"], 0.14)
    assert eval_summary["copy_risk_count_delta_gt_0_015"] == 0


def test_min_score_gain_can_fallback_to_baseline(tmp_path):
    baseline = "base"
    methods = ["base", "weak"]
    train_rows = [
        _row("train0", "base", target=0.50, ref=0.40, clip=0.20, copy=0.10),
        _row("train0", "weak", target=0.51, ref=0.41, clip=0.20, copy=0.10),
    ]
    eval_rows = [
        _row("eval0", "base", target=0.50, ref=0.40, clip=0.20, copy=0.10),
        _row("eval0", "weak", target=0.51, ref=0.41, clip=0.20, copy=0.10),
    ]

    result = analyze_rerank_policy_sweep(
        train_metrics=_write_metrics(tmp_path / "train.csv", train_rows),
        eval_runs=[("eval", _write_metrics(tmp_path / "eval.csv", eval_rows))],
        baseline_method=baseline,
        candidate_methods=methods,
        copy_penalties=[0.0],
        clip_weights=[0.0],
        min_score_gains=[0.02],
        bootstrap_iterations=20,
    )

    assert result["eval_aggregate"]["selected_method_counts"] == {"base": 1}
    assert math.isclose(result["eval_aggregate"]["summary"]["dino_delta_mean"], 0.0)


def test_dino_copy_penalty_can_avoid_high_ref_copy_candidate(tmp_path):
    baseline = "base"
    methods = ["base", "copy_like", "safe"]
    train_rows = [
        _row("train0", "base", target=0.50, ref=0.40, clip=0.20, copy=0.10, dino_copy=0.20),
        _row("train0", "copy_like", target=0.47, ref=0.90, clip=0.20, copy=0.12, dino_copy=0.95),
        _row("train0", "safe", target=0.62, ref=0.65, clip=0.20, copy=0.11, dino_copy=0.25),
        _row("train1", "base", target=0.50, ref=0.40, clip=0.20, copy=0.10, dino_copy=0.20),
        _row("train1", "copy_like", target=0.48, ref=0.88, clip=0.20, copy=0.12, dino_copy=0.92),
        _row("train1", "safe", target=0.61, ref=0.66, clip=0.20, copy=0.11, dino_copy=0.24),
    ]
    eval_rows = [
        _row("eval0", "base", target=0.50, ref=0.40, clip=0.20, copy=0.10, dino_copy=0.20),
        _row("eval0", "copy_like", target=0.46, ref=0.90, clip=0.20, copy=0.12, dino_copy=0.96),
        _row("eval0", "safe", target=0.63, ref=0.64, clip=0.20, copy=0.11, dino_copy=0.25),
    ]

    result = analyze_rerank_policy_sweep(
        train_metrics=_write_metrics(tmp_path / "train.csv", train_rows),
        eval_runs=[("eval", _write_metrics(tmp_path / "eval.csv", eval_rows))],
        baseline_method=baseline,
        candidate_methods=methods,
        copy_penalties=[0.0],
        dino_copy_penalties=[0.0, 1.0],
        clip_weights=[0.0],
        min_score_gains=[-999.0],
        bootstrap_iterations=20,
    )

    assert result["selected_policy"]["dino_copy_penalty"] == 1.0
    assert result["eval_aggregate"]["selected_method_counts"] == {"safe": 1}
    assert math.isclose(result["eval_aggregate"]["summary"]["dino_delta_mean"], 0.13)
