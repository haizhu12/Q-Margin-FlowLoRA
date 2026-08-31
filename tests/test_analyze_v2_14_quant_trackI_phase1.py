import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.analyze_v2_14_quant_trackI_phase1 import run_quant_tracki_from_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_AGGREGATE_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "quant_main_v2_14_20260626"
    / "phase1_trackI"
    / "r14_replay"
    / "aggregate_table.csv"
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


def test_paper_row_field_order_matches_frozen_header_across_hash_seeds():
    with FROZEN_AGGREGATE_TABLE.open("r", newline="", encoding="utf-8") as handle:
        frozen_header = next(csv.reader(handle))

    source = {key: f"value-for-{key}" for key in frozen_header[5:]}
    probe = """
import json
import sys

from scripts.analyze_v2_14_quant_trackI_phase1 import _paper_row

source = json.loads(sys.argv[1])
row = _paper_row("comparison", "source_table", source, scope="scope")
print(json.dumps(list(row), separators=(",", ":")))
"""

    observed_headers = []
    for hash_seed in ("1", "20260624"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", probe, json.dumps(source)],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        observed_headers.append(json.loads(completed.stdout))

    assert observed_headers == [frozen_header, frozen_header]


def test_quant_tracki_manifest_run_writes_paper_grade_outputs(tmp_path):
    metrics = _write_metrics(tmp_path / "case_metrics.csv", _toy_rows())
    manifest = {
        "official_method": {
            "baseline_method": "base",
            "candidate_methods": ["base", "refmax", "clip"],
        },
        "runs": [
            {
                "label": "toy_validation",
                "role": "validation",
                "case_metrics": str(metrics),
                "case_count": 4,
            },
            {
                "label": "toy_heldout",
                "role": "heldout",
                "case_metrics": str(metrics),
                "case_count": 4,
            },
        ],
        "statistics": {
            "bootstrap_iterations": 40,
            "bootstrap_seed": 11,
        },
        "outputs": {
            "phase1a_output_dir": str(tmp_path / "quant_out"),
        },
    }
    manifest_path = tmp_path / "phase0_protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    written = run_quant_tracki_from_manifest(manifest_path)

    output_dir = Path(manifest["outputs"]["phase1a_output_dir"])
    expected_files = {
        "aggregate_table.csv",
        "case_level_predictions.csv",
        "split_breakdown.csv",
        "category_breakdown.csv",
        "guard_breakdown.csv",
        "copy_risk_cases.csv",
        "worst_cases.csv",
        "summary.json",
        "result.md",
        "run_command.txt",
        "environment.txt",
        "output_manifest.json",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    assert not any("ablation" in path.name for path in output_dir.iterdir())
    assert Path(written["aggregate_table_csv"]).exists()

    with (output_dir / "aggregate_table.csv").open("r", newline="", encoding="utf-8") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    comparisons = {row["comparison"] for row in aggregate_rows}
    assert {"Locked Hybrid", "Pure RefMax", "Target Oracle"} <= comparisons
    for row in aggregate_rows:
        assert {"dino_delta_mean", "ci95_low", "ci95_high", "win_rate"} <= set(row)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metadata"]["analysis"] == "quant_trackI_phase1a_r14_replay"
    assert summary["validation"]["missing_paths"] == []
    assert summary["validation"]["bad_candidate_case_count"] == 0
