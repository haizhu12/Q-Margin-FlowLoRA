import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_flux2_utils_has_no_legacy_lora_gate_hook_or_import():
    source = (PROJECT_ROOT / "qmargin" / "flux2_utils.py").read_text(encoding="utf-8")

    assert "def install_timestep_gate_hook" not in source
    assert "qmargin.lora_gated" not in source


def test_ctnr_oracle_import_does_not_load_legacy_reference_training_stack(tmp_path):
    forbidden = [
        "scripts.infer_reference_guided",
        "scripts.train_qm_refflowlora",
        "qmargin.lora_gated",
        "qmargin.checkpointing",
        "qmargin.ref.losses",
        "qmargin.ref.q_nria_adapter",
        "qmargin.ref.reference_adapter",
        "qmargin.ref.reference_encoder",
        "qmargin.ref.selected_teacher_dataset",
        "qmargin.ref.semantic_native_interaction_adapter",
        "qmargin.ref.subject_dataset",
    ]
    probe = (
        "import sys\n"
        "import scripts.run_v2_8_ctnr_oracle\n"
        f"forbidden = {forbidden!r}\n"
        "loaded = sorted(name for name in forbidden if name in sys.modules)\n"
        "if loaded:\n"
        "    raise SystemExit('legacy modules imported: ' + ', '.join(loaded))\n"
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
