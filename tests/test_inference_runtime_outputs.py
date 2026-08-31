import json
from types import SimpleNamespace

from qmargin.inference_runtime import save_inference_outputs


def _output_args(tmp_path, checkpoint):
    return SimpleNamespace(
        out_dir=str(tmp_path / "outputs"),
        save_gate_trace=False,
        reference_conditioning_mode="joint_append_current_timestep",
        mode="stepwise_gate",
        prompt="test prompt",
        checkpoint=checkpoint,
        config=str(tmp_path / "config.yaml"),
        ref_images=[],
        num_images=1,
        num_inference_steps=1,
        seed=42,
    )


def test_save_inference_outputs_serializes_missing_checkpoint_as_json_null(tmp_path):
    args = _output_args(tmp_path, checkpoint=None)

    result = save_inference_outputs({"gate_trace": []}, args, {}, {}, {})

    metadata = json.loads((tmp_path / "outputs" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint"] is None
    assert result["metadata"]["checkpoint"] is None


def test_save_inference_outputs_keeps_real_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    args = _output_args(tmp_path, checkpoint=checkpoint)

    result = save_inference_outputs({"gate_trace": []}, args, {}, {}, {})

    metadata = json.loads((tmp_path / "outputs" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint"] == str(checkpoint)
    assert result["metadata"]["checkpoint"] == str(checkpoint)
