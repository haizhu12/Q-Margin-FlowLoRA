import csv
import math
from pathlib import Path

from scripts.analyze_v2_14_compressibility import analyze_compressibility


def _write_case_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_compressibility_separates_fixed_single_from_per_case_selection(tmp_path):
    baseline = "base"
    methods = ["base", "a", "b", "c"]
    rows = []
    # The best fixed method is weak: each candidate wins one case and loses the rest.
    # A target-free reference selector can recover the per-case winners.
    winners = {"case0": "a", "case1": "b", "case2": "c"}
    for case_id, winner in winners.items():
        rows.append(
            {
                "case_id": case_id,
                "subject_id": f"subj_{case_id}",
                "category": "cat",
                "method": baseline,
                "dino_sim_to_target": "0.50",
                "dino_sim_to_multi_ref_mean": "0.10",
                "clip_text_image_sim": "0.20",
                "ref_copy_ssim_max": "0.10",
            }
        )
        for method in methods[1:]:
            is_winner = method == winner
            rows.append(
                {
                    "case_id": case_id,
                    "subject_id": f"subj_{case_id}",
                    "category": "cat",
                    "method": method,
                    "dino_sim_to_target": "0.60" if is_winner else "0.48",
                    "dino_sim_to_multi_ref_mean": "0.90" if is_winner else "0.15",
                    "clip_text_image_sim": "0.21" if is_winner else "0.19",
                    "ref_copy_ssim_max": "0.11",
                }
            )
    metrics_path = tmp_path / "case_metrics.csv"
    _write_case_metrics(metrics_path, rows)

    result = analyze_compressibility(
        runs=[("synthetic", metrics_path)],
        baseline_method=baseline,
        candidate_methods=methods,
        bootstrap_iterations=50,
        seed=7,
    )

    run = result["runs"][0]
    assert run["best_fixed_method"] in {"a", "b", "c"}
    assert math.isclose(run["best_fixed"]["dino_delta_mean"], 0.01999999999999998)
    assert math.isclose(run["reference_selector"]["dino_delta_mean"], 0.10)
    assert math.isclose(run["target_oracle"]["dino_delta_mean"], 0.10)
    assert run["selector_method_counts"] == {"a": 1, "b": 1, "c": 1}
    assert result["aggregate"]["reference_selector"]["dino_delta_mean"] > 0.09
    assert result["aggregate"]["best_fixed"]["case_count"] == 3
    assert math.isclose(result["aggregate"]["best_fixed"]["dino_delta_mean"], 0.01999999999999998)
    assert result["aggregate"]["global_best_fixed_method"] in {"a", "b", "c"}
    assert math.isclose(result["aggregate"]["global_best_fixed"]["dino_delta_mean"], 0.01999999999999998)
    assert result["aggregate"]["selection_gap_over_best_fixed"] > 0.07
    assert result["decision"]["single_path_compression_ready"] is False
