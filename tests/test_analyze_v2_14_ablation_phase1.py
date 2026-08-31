import csv
import math
from pathlib import Path

from scripts.analyze_v2_14_ablation_phase1 import (
    analyze_phase1_offline,
    summarize_prediction_rows,
    write_phase1_outputs,
)


def _row(
    case_id: str,
    subject_id: str,
    method: str,
    *,
    target: float,
    ref_mean: float,
    ref_max: float,
    clip: float,
    copy: float,
    dino_copy: float,
) -> dict:
    return {
        "case_id": case_id,
        "subject_id": subject_id,
        "category": "synthetic",
        "method": method,
        "prompt": "synthetic prompt",
        "image_path": f"outputs/{case_id}/{method}/000000.png",
        "dino_sim_to_target": str(target),
        "dino_sim_to_multi_ref_mean": str(ref_mean),
        "dino_ref_copy_sim_max": str(ref_max),
        "clip_text_image_sim": str(clip),
        "ref_copy_ssim_max": str(copy),
        "ssim_to_multi_ref_mean": str(copy - 0.01),
    }


def _write_metrics(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _toy_rows() -> list[dict]:
    rows = []
    specs = [
        ("case0", "subject0", 0.50, 0.62, 0.58),
        ("case1", "subject0", 0.50, 0.45, 0.64),
        ("case2", "subject1", 0.50, 0.61, 0.56),
        ("case3", "subject1", 0.50, 0.46, 0.63),
    ]
    for case_id, subject_id, base_target, refmax_target, clip_target in specs:
        rows.extend(
            [
                _row(
                    case_id,
                    subject_id,
                    "base",
                    target=base_target,
                    ref_mean=0.40,
                    ref_max=0.42,
                    clip=0.20,
                    copy=0.10,
                    dino_copy=0.20,
                ),
                _row(
                    case_id,
                    subject_id,
                    "refmax",
                    target=refmax_target,
                    ref_mean=0.80,
                    ref_max=0.85,
                    clip=0.21,
                    copy=0.13,
                    dino_copy=0.30,
                ),
                _row(
                    case_id,
                    subject_id,
                    "clip",
                    target=clip_target,
                    ref_mean=0.55,
                    ref_max=0.60,
                    clip=0.35,
                    copy=0.11,
                    dino_copy=0.22,
                ),
            ]
        )
    return rows


def test_summarize_prediction_rows_reports_95_ci_and_subject_cluster_metadata():
    rows = [
        {
            "run": "toy",
            "case_id": "case0",
            "subject_id": "subject0",
            "selected_method": "refmax",
            "baseline_method": "base",
            "dino_delta": 0.10,
            "clip_delta": 0.02,
            "copy_ssim_delta": 0.02,
            "triggered": True,
            "false_intervention": False,
        },
        {
            "run": "toy",
            "case_id": "case1",
            "subject_id": "subject0",
            "selected_method": "refmax",
            "baseline_method": "base",
            "dino_delta": -0.02,
            "clip_delta": 0.00,
            "copy_ssim_delta": 0.00,
            "triggered": False,
            "false_intervention": False,
        },
        {
            "run": "toy",
            "case_id": "case2",
            "subject_id": "subject1",
            "selected_method": "refmax",
            "baseline_method": "base",
            "dino_delta": 0.04,
            "clip_delta": 0.01,
            "copy_ssim_delta": 0.03,
            "triggered": True,
            "false_intervention": True,
        },
    ]

    summary = summarize_prediction_rows(
        rows,
        table="unit",
        variant="refmax",
        baseline_variant="base",
        bootstrap_iterations=50,
        seed=7,
    )

    assert summary["n_cases"] == 3
    assert summary["n_subjects"] == 2
    assert summary["ci_level"] == 0.95
    assert summary["interval_method"] == "subject_clustered_bootstrap"
    assert {"ci95_low", "ci95_high", "ci90_low", "ci90_high"} <= set(summary)
    assert {"r14_case_bootstrap_ci90_low", "r14_case_bootstrap_ci90_high"} <= set(summary)
    assert summary["r14_compatibility_interval_method"] == "case_bootstrap"
    assert summary["win_rate"] == 2 / 3
    assert summary["copy_risk_count"] == 2
    assert summary["copy_risk_rate"] == 2 / 3
    assert summary["trigger_rate"] == 2 / 3
    assert summary["false_intervention_rate"] == 1 / 3


def test_phase1_offline_writes_core_ablation_tables_with_interval_columns(tmp_path):
    metrics = _write_metrics(tmp_path / "case_metrics.csv", _toy_rows())

    result = analyze_phase1_offline(
        runs=[
            ("toy_validation", "validation", metrics),
            ("toy_heldout", "heldout", metrics),
        ],
        baseline_method="base",
        candidate_methods=["base", "refmax", "clip"],
        bootstrap_iterations=50,
        seed=11,
        random_seed=13,
    )
    output_dir = tmp_path / "out"
    written = write_phase1_outputs(result, output_dir=output_dir, figures_dir=output_dir / "figures")

    required_tables = {
        "static_method_summary",
        "global_best_static_summary",
        "candidate_subset_summary",
        "selector_signal_summary",
        "aggregation_role_summary",
        "guard_strategy_summary",
        "threshold_validation_summary",
        "threshold_heldout_summary",
    }
    assert required_tables <= set(result["tables"])
    for table_name in required_tables:
        assert result["tables"][table_name], table_name
        row = result["tables"][table_name][0]
        assert {"estimate", "ci95_low", "ci95_high", "ci90_low", "ci90_high"} <= set(row)

    assert result["metadata"]["validation_labels"] == ["toy_validation"]
    assert result["metadata"]["heldout_labels"] == ["toy_heldout"]
    assert result["tables"]["global_best_static_summary"][0]["selected_on_validation_method"] in {
        "refmax",
        "clip",
    }
    assert math.isfinite(float(result["tables"]["candidate_subset_summary"][-1]["oracle_recovery_estimate"]))
    assert Path(written["summary_json"]).exists()
    assert Path(written["result_md"]).exists()
