import csv
from pathlib import Path

from scripts.analyze_v2_14_ablation_phase2 import analyze_phase2_metrics


FIELDNAMES = [
    "case_id",
    "subject_id",
    "category",
    "method",
    "dino_sim_to_target",
    "clip_text_image_sim",
    "ref_copy_ssim_max",
]


def _write_case_metrics(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(case_id, method, dino, clip=0.5, copy=0.1):
    return {
        "case_id": case_id,
        "subject_id": case_id.replace("case", "subject"),
        "category": "toy",
        "method": method,
        "dino_sim_to_target": str(dino),
        "clip_text_image_sim": str(clip),
        "ref_copy_ssim_max": str(copy),
    }


def _read_summary(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def test_phase2_analysis_builds_best_of_routes_and_paired_deltas(tmp_path):
    case_metrics = tmp_path / "case_metrics.csv"
    _write_case_metrics(
        case_metrics,
        [
            _row("case1", "v2_8a_static_a064_d080", 1.0, copy=0.10),
            _row("case1", "v2_8a_static_a048_d096", 1.3, copy=0.11),
            _row("case1", "v2_8a_static_a032_d112", 1.1, copy=0.12),
            _row("case1", "v2_8a_static_a016_d128", 0.9, copy=0.09),
            _row("case1", "v2_8a_static_a000_d144", 1.2, copy=0.08),
            _row("case1", "v2_14_a5_same_a064_seed0", 1.0, copy=0.10),
            _row("case1", "v2_14_a5_same_a064_seed1", 0.8, copy=0.13),
            _row("case1", "v2_14_a5_same_a064_seed2", 1.4, copy=0.16),
            _row("case1", "v2_14_a5_same_a064_seed3", 1.1, copy=0.12),
            _row("case1", "v2_14_a5_same_a064_seed4", 1.2, copy=0.14),
            _row("case1", "v2_14_c2_mask_off", 1.2, copy=0.13),
            _row("case2", "v2_8a_static_a064_d080", 2.0, copy=0.20),
            _row("case2", "v2_8a_static_a048_d096", 1.5, copy=0.19),
            _row("case2", "v2_8a_static_a032_d112", 2.4, copy=0.21),
            _row("case2", "v2_8a_static_a016_d128", 1.8, copy=0.18),
            _row("case2", "v2_8a_static_a000_d144", 2.1, copy=0.22),
            _row("case2", "v2_14_a5_same_a064_seed0", 2.0, copy=0.20),
            _row("case2", "v2_14_a5_same_a064_seed1", 2.2, copy=0.18),
            _row("case2", "v2_14_a5_same_a064_seed2", 1.7, copy=0.17),
            _row("case2", "v2_14_a5_same_a064_seed3", 2.3, copy=0.19),
            _row("case2", "v2_14_a5_same_a064_seed4", 1.9, copy=0.16),
            _row("case2", "v2_14_c2_mask_off", 1.9, copy=0.21),
        ],
    )

    result = analyze_phase2_metrics(
        case_metrics_path=case_metrics,
        output_dir=tmp_path / "out",
        bootstrap_iterations=200,
        seed=7,
    )

    summary = _read_summary(result["relative_summary_csv"])
    structured = summary["A5_structured_5route_oracle"]
    same_route = summary["A5_same_a064_5seed_oracle"]
    mask_off = summary["v2_14_c2_mask_off"]

    assert float(structured["dino_delta_mean"]) == 0.35
    assert float(same_route["dino_delta_mean"]) == 0.35
    assert int(mask_off["copy_risk_count"]) == 1
    assert float(mask_off["win_rate"]) == 0.5
