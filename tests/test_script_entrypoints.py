import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script_name",
    [
        "run_v2_8_ctnr_oracle.py",
        "convert_customconcept101_manifest.py",
        "convert_sop_manifest.py",
        "prepare_sop_small_valid_split.py",
        "build_fixed_eval_from_manifest.py",
        "evaluate_fixed_reference_metrics.py",
        "analyze_v2_14_refmax_guard_hybrid.py",
        "run_background_replace.py",
        "create_v2_14_ablation_visual_figures.py",
    ],
)
def test_direct_script_help_runs_from_project_root(script_name):
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / script_name), "--help"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
