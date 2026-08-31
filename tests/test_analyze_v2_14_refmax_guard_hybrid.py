import csv
import math
from pathlib import Path

from scripts.analyze_v2_14_refmax_guard_hybrid import (
    analyze_refmax_guard_hybrid,
    predict_with_refmax_guard_hybrid,
)


FIELDNAMES = [
    "case_id",
    "subject_id",
    "category",
    "method",
    "dino_sim_to_target",
    "dino_sim_to_multi_ref_mean",
    "clip_text_image_sim",
    "ref_copy_ssim_max",
    "dino_ref_copy_sim_max",
]


def _row(
    case_id: str,
    method: str,
    *,
    target: float,
    ref: float,
    copy: float,
    dino_copy: float,
) -> dict:
    return {
        "case_id": case_id,
        "subject_id": f"subject_{case_id}",
        "category": "synthetic",
        "method": method,
        "dino_sim_to_target": str(target),
        "dino_sim_to_multi_ref_mean": str(ref),
        "clip_text_image_sim": "0.20",
        "ref_copy_ssim_max": str(copy),
        "dino_ref_copy_sim_max": str(dino_copy),
    }


def _write_metrics(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_hybrid_uses_refmax_when_copy_risk_is_below_threshold():
    rows = [
        _row("case0", "base", target=0.50, ref=0.40, copy=0.10, dino_copy=0.20),
        _row("case0", "refmax", target=0.62, ref=0.70, copy=0.11, dino_copy=0.25),
        _row("case0", "guard", target=0.58, ref=0.65, copy=0.10, dino_copy=0.22),
    ]

    predictions = predict_with_refmax_guard_hybrid(
        rows,
        baseline_method="base",
        candidate_methods=["base", "refmax", "guard"],
        copy_delta_threshold=0.05,
        dino_copy_delta_threshold=0.10,
    )

    assert predictions == {"case0": "refmax"}


def test_hybrid_falls_back_to_guard_when_refmax_has_high_dino_copy_delta(tmp_path):
    rows = [
        _row("case0", "base", target=0.50, ref=0.40, copy=0.10, dino_copy=0.20),
        _row("case0", "copy_like", target=0.47, ref=0.75, copy=0.11, dino_copy=0.95),
        _row("case0", "safe", target=0.63, ref=0.70, copy=0.10, dino_copy=0.20),
    ]
    metrics = _write_metrics(tmp_path / "metrics.csv", rows)

    result = analyze_refmax_guard_hybrid(
        eval_runs=[("eval", metrics)],
        baseline_method="base",
        candidate_methods=["base", "copy_like", "safe"],
        copy_delta_threshold=999.0,
        dino_copy_delta_threshold=0.10,
        bootstrap_iterations=20,
    )

    assert result["aggregate"]["selected_method_counts"] == {"safe": 1}
    assert math.isclose(result["aggregate"]["summary"]["dino_delta_mean"], 0.13)
