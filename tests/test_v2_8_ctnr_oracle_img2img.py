import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.run_v2_8_ctnr_oracle as runner


def test_flowmatch_img2img_uses_selected_timestep_and_scheduler_scaling(monkeypatch):
    class RecordingScheduler:
        def __init__(self):
            self.begin_indices = []
            self._step_index = 99
            self.scale_calls = []

        def set_begin_index(self, value):
            self.begin_indices.append(value)

        def scale_noise(self, clean, timestep, noise):
            self.scale_calls.append((clean.clone(), timestep.clone(), noise.clone(), self._step_index))
            return clean + 10 * noise

    scheduler = RecordingScheduler()
    pipe = SimpleNamespace(scheduler=scheduler)
    pure_noise = torch.full((1, 2, 3), 2.0)
    clean_reference = torch.full((1, 2, 3), 3.0)
    latent_inputs = {
        "latents_packed": pure_noise,
        "img_ids": torch.zeros(1, 2, 4),
        "timesteps": torch.tensor([1000.0, 750.0, 500.0, 250.0]),
        "latent_meta": {"height": 64, "width": 64},
    }
    ref_grids = torch.zeros(1, 1, 4, 8, 8)
    monkeypatch.setattr(
        runner,
        "pack_transformer_latents_compat",
        lambda actual_pipe, first_ref: (clean_reference, torch.zeros(1, 2, 4)),
    )

    result = runner._prepare_flowmatch_img2img_latents(
        pipe,
        latent_inputs,
        ref_grids,
        edit_strength=0.5,
    )

    assert scheduler.begin_indices == [2]
    assert scheduler.scale_calls[0][3] is None
    assert torch.equal(scheduler.scale_calls[0][0], clean_reference)
    assert torch.equal(scheduler.scale_calls[0][1], torch.tensor([500.0]))
    assert torch.equal(scheduler.scale_calls[0][2], pure_noise)
    assert torch.equal(result["latents_packed"], clean_reference + 10 * pure_noise)
    assert torch.equal(result["timesteps"], torch.tensor([500.0, 250.0]))
    assert result["latent_meta"] == {
        "height": 64,
        "width": 64,
        "edit_strength": 0.5,
        "edit_active": True,
        "t_start": 2,
        "actual_edit_steps": 2,
    }


def test_case_trajectory_resets_scheduler_to_edit_start(monkeypatch, tmp_path):
    class RecordingScheduler:
        def __init__(self):
            self.begin_indices = []
            self._step_index = 7

        def set_begin_index(self, value):
            self.begin_indices.append(value)

    scheduler = RecordingScheduler()
    pipe = SimpleNamespace(scheduler=scheduler)
    monkeypatch.setattr(runner.ref_infer, "decode_packed_latents_to_images", lambda *_args: [])
    monkeypatch.setattr(
        runner.ref_infer,
        "save_inference_outputs",
        lambda *_args: {"metadata_path": str(tmp_path / "metadata.json")},
    )

    runner._run_case_method(
        pipe=pipe,
        cfg={"reference_adapter": {}},
        row={"case_id": "case", "prompt": "edit it", "multi_reference_images": ["ref.png"]},
        method_name="v2_8a_static_a064_d080",
        bank=None,
        prompt_bundle=None,
        negative_prompt_bundle=None,
        latent_inputs={
            "latents_packed": torch.zeros(1, 2, 3),
            "img_ids": torch.zeros(1, 2, 4),
            "timesteps": torch.tensor([]),
            "latent_meta": {"t_start": 3},
        },
        latent_seed=1,
        output_root=tmp_path,
        height=64,
        width=64,
        num_inference_steps=4,
        guidance_scale=1.0,
        negative_ref_mode="same",
        ref_latent_scale=1.0,
        device="cpu",
        dtype=torch.float32,
    )

    assert scheduler.begin_indices == [3]
    assert scheduler._step_index is None


def test_zero_edit_strength_is_an_exact_preparation_noop(monkeypatch):
    class UntouchedScheduler:
        def set_begin_index(self, _value):
            raise AssertionError("zero strength must not reset scheduler during edit preparation")

        def scale_noise(self, *_args):
            raise AssertionError("zero strength must not add reference noise")

    pipe = SimpleNamespace(scheduler=UntouchedScheduler())
    latent_inputs = {
        "latents_packed": torch.randn(1, 2, 3),
        "img_ids": torch.zeros(1, 2, 4),
        "timesteps": torch.tensor([1000.0, 500.0]),
        "latent_meta": {"height": 64, "width": 64},
    }
    monkeypatch.setattr(
        runner,
        "pack_transformer_latents_compat",
        lambda *_args: (_ for _ in ()).throw(AssertionError("zero strength must not pack a reference")),
    )

    result = runner._prepare_flowmatch_img2img_latents(
        pipe,
        latent_inputs,
        torch.zeros(1, 1, 4, 8, 8),
        edit_strength=0.0,
    )

    assert result is latent_inputs


def test_flowmatch_img2img_rejects_clean_reference_noise_shape_mismatch(monkeypatch):
    class Scheduler:
        _step_index = None

        def set_begin_index(self, _value):
            pass

        def scale_noise(self, *_args):
            raise AssertionError("shape mismatch must fail before scheduler.scale_noise")

    pipe = SimpleNamespace(scheduler=Scheduler())
    latent_inputs = {
        "latents_packed": torch.zeros(1, 2, 3),
        "timesteps": torch.tensor([1000.0, 500.0]),
        "latent_meta": {},
    }
    monkeypatch.setattr(
        runner,
        "pack_transformer_latents_compat",
        lambda *_args: (torch.zeros(1, 3, 3), torch.zeros(1, 3, 4)),
    )

    with pytest.raises(
        ValueError,
        match=r"FlowMatch img2img latent shape mismatch: clean_reference=\(1, 3, 3\) pure_noise=\(1, 2, 3\)",
    ):
        runner._prepare_flowmatch_img2img_latents(
            pipe,
            latent_inputs,
            torch.zeros(1, 1, 4, 8, 8),
            edit_strength=0.5,
        )


@pytest.mark.parametrize("edit_strength", [-0.01, 1.01])
def test_edit_strength_rejects_values_outside_closed_unit_interval(edit_strength):
    with pytest.raises(ValueError, match=r"--edit_strength must be in \[0, 1\]"):
        runner._validate_edit_request(
            edit_strength=edit_strength,
            manifest_schema=runner.SINGLE_REFERENCE_NO_TARGET_SCHEMA,
            init_ref_latent_blend=0.0,
        )


def test_active_edit_requires_single_reference_no_target_schema():
    with pytest.raises(
        ValueError,
        match=r"--edit_strength > 0 requires --manifest_schema single_reference_no_target",
    ):
        runner._validate_edit_request(
            edit_strength=0.5,
            manifest_schema=runner.FIXED_REFERENCE_SCHEMA,
            init_ref_latent_blend=0.0,
        )


def test_active_edit_rejects_legacy_linear_reference_blend():
    with pytest.raises(
        ValueError,
        match=r"--edit_strength and --init_ref_latent_blend cannot both be greater than zero",
    ):
        runner._validate_edit_request(
            edit_strength=0.5,
            manifest_schema=runner.SINGLE_REFERENCE_NO_TARGET_SCHEMA,
            init_ref_latent_blend=0.1,
        )


def test_cli_edit_strength_defaults_to_disabled():
    args = runner.parse_args(
        [
            "--config",
            "config.yaml",
            "--eval_set",
            "cases.jsonl",
            "--output_root",
            "out",
        ]
    )

    assert args.edit_strength == 0.0


def test_run_rejects_invalid_edit_schema_before_loading_model(monkeypatch, tmp_path):
    cfg = {
        "model": {"inference_model_id": "unused"},
        "reference_adapter": {"source": "native_vae", "apply_native_adapter": False},
        "semantic_reference": {"enabled": False},
    }
    monkeypatch.setattr(runner.ref_infer, "load_config", lambda _path: cfg)
    monkeypatch.setattr(runner, "validate_config", lambda value: value)
    monkeypatch.setattr(runner.ref_infer, "reference_source", lambda _cfg: "native_vae")
    monkeypatch.setattr(runner, "_eval_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        runner,
        "load_flux2_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid edit request must fail before model loading")
        ),
    )
    args = SimpleNamespace(
        reference_conditioning_mode=runner.DEFAULT_REFERENCE_CONDITIONING_MODE,
        config="config.yaml",
        checkpoint=None,
        methods=["v2_8a_static_a064_d080"],
        device="cpu",
        root=tmp_path,
        output_root=tmp_path / "out",
        eval_set=tmp_path / "cases.jsonl",
        manifest_schema=runner.FIXED_REFERENCE_SCHEMA,
        guidance_scale=1.0,
        negative_ref_mode="same",
        ref_latent_scale=1.0,
        init_ref_latent_blend=0.0,
        edit_strength=0.5,
        height=64,
        width=64,
        num_inference_steps=4,
    )

    with pytest.raises(
        ValueError,
        match=r"--edit_strength > 0 requires --manifest_schema single_reference_no_target",
    ):
        runner.run(args)


def test_run_applies_img2img_after_hashing_pure_noise_and_reports_edit_metadata(monkeypatch, tmp_path):
    cfg = {
        "model": {"inference_model_id": "unused"},
        "reference_adapter": {
            "source": "native_vae",
            "apply_native_adapter": False,
            "tokens_per_image": 144,
            "coreset_anchor_tokens": 64,
        },
        "semantic_reference": {"enabled": False},
    }
    pure_noise = torch.full((1, 2, 3), 2.0)
    clean_reference = torch.full((1, 2, 3), 3.0)
    captured = {}

    class Scheduler:
        def __init__(self):
            self.begin_indices = []
            self._step_index = 9

        def set_begin_index(self, value):
            self.begin_indices.append(value)

        def scale_noise(self, clean, timestep, noise):
            return clean + noise

    pipe = SimpleNamespace(scheduler=Scheduler())
    monkeypatch.setattr(runner.ref_infer, "load_config", lambda _path: cfg)
    monkeypatch.setattr(runner, "validate_config", lambda value: value)
    monkeypatch.setattr(runner.ref_infer, "reference_source", lambda _cfg: "native_vae")
    monkeypatch.setattr(runner.ref_infer, "_dtype_from_cfg", lambda _cfg: torch.float32)
    monkeypatch.setattr(
        runner,
        "_eval_rows",
        lambda *_args, **_kwargs: [
            {
                "case_id": "case",
                "prompt": "put the object on a beach",
                "seed": 42,
                "multi_reference_images": ["ref.png"],
            }
        ],
    )
    monkeypatch.setattr(runner, "load_flux2_pipeline", lambda *_args, **_kwargs: pipe)
    monkeypatch.setattr(runner, "freeze_base_model", lambda _pipe: None)
    monkeypatch.setattr(runner.ref_infer, "encode_prompt_compat", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runner.ref_infer,
        "prepare_inference_latents",
        lambda *_args, **_kwargs: {
            "latents_packed": pure_noise.clone(),
            "img_ids": torch.zeros(1, 2, 4),
            "timesteps": torch.tensor([1000.0, 750.0, 500.0, 250.0]),
            "latent_meta": {"height": 64, "width": 64},
        },
    )
    monkeypatch.setattr(
        runner,
        "_load_ref_latent_grids",
        lambda *_args, **_kwargs: torch.zeros(1, 1, 4, 8, 8),
    )
    monkeypatch.setattr(
        runner,
        "pack_transformer_latents_compat",
        lambda *_args: (clean_reference.clone(), torch.zeros(1, 2, 4)),
    )
    monkeypatch.setattr(runner, "build_reference_token_bank", lambda *_args, **_kwargs: object())

    def capture_case(**kwargs):
        captured.update(kwargs)
        return {"metadata_path": "metadata.json", "image_paths": [], "gate_trace_path": "trace.json"}

    monkeypatch.setattr(runner, "_run_case_method", capture_case)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(
        reference_conditioning_mode=runner.DEFAULT_REFERENCE_CONDITIONING_MODE,
        config="actual-demo-config.yaml",
        checkpoint=None,
        methods=["v2_8a_static_a064_d080"],
        device="cpu",
        root=tmp_path,
        output_root=tmp_path / "out",
        eval_set=tmp_path / "cases.jsonl",
        manifest_schema=runner.SINGLE_REFERENCE_NO_TARGET_SCHEMA,
        guidance_scale=1.0,
        negative_ref_mode="same",
        ref_latent_scale=1.0,
        init_ref_latent_blend=0.0,
        edit_strength=0.5,
        height=64,
        width=64,
        num_inference_steps=4,
    )

    summary = runner.run(args)

    edited = captured["latent_inputs"]
    assert edited["initial_noise_sha256"] == runner._packed_latent_sha256(pure_noise)
    assert torch.equal(edited["latents_packed"], clean_reference + pure_noise)
    assert torch.equal(edited["timesteps"], torch.tensor([500.0, 250.0]))
    assert edited["latent_meta"]["edit_strength"] == 0.5
    assert edited["latent_meta"]["edit_active"] is True
    assert edited["latent_meta"]["t_start"] == 2
    assert edited["latent_meta"]["actual_edit_steps"] == 2
    assert summary["edit_strength"] == 0.5
    assert summary["edit_active"] is True
    assert summary["t_start"] == 2
    assert summary["actual_edit_steps"] == 2
    assert captured["config_path"] == "actual-demo-config.yaml"


def test_case_metadata_records_the_actual_requested_config(monkeypatch, tmp_path):
    class Scheduler:
        _step_index = 3

        def set_begin_index(self, _value):
            pass

    monkeypatch.setattr(runner.ref_infer, "decode_packed_latents_to_images", lambda *_args: [])
    result = runner._run_case_method(
        pipe=SimpleNamespace(scheduler=Scheduler()),
        cfg={"reference_adapter": {}, "adapter_injection": {}},
        row={"case_id": "case", "prompt": "edit it", "multi_reference_images": ["ref.png"]},
        method_name="v2_8a_static_a064_d080",
        bank=None,
        prompt_bundle=None,
        negative_prompt_bundle=None,
        latent_inputs={
            "latents_packed": torch.zeros(1, 2, 3),
            "img_ids": torch.zeros(1, 2, 4),
            "timesteps": torch.tensor([]),
            "latent_meta": {"height": 64, "width": 64},
        },
        latent_seed=1,
        output_root=tmp_path,
        height=64,
        width=64,
        num_inference_steps=4,
        guidance_scale=1.0,
        negative_ref_mode="same",
        ref_latent_scale=1.0,
        config_path="actual-demo-config.yaml",
        device="cpu",
        dtype=torch.float32,
    )

    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["config"] == "actual-demo-config.yaml"
    assert metadata["latent_meta"] == {
        "height": 64,
        "width": 64,
        "edit_strength": 0.0,
        "edit_active": False,
        "t_start": 0,
        "actual_edit_steps": 0,
    }


def test_legacy_blend_still_fingerprints_original_pure_noise(monkeypatch, tmp_path):
    cfg = {
        "model": {"inference_model_id": "unused"},
        "reference_adapter": {
            "source": "native_vae",
            "apply_native_adapter": False,
            "tokens_per_image": 144,
            "coreset_anchor_tokens": 64,
        },
        "semantic_reference": {"enabled": False},
    }
    pure_noise = torch.full((1, 2, 3), 2.0)
    clean_reference = torch.full((1, 2, 3), 4.0)
    captured = {}
    pipe = SimpleNamespace(scheduler=SimpleNamespace())
    monkeypatch.setattr(runner.ref_infer, "load_config", lambda _path: cfg)
    monkeypatch.setattr(runner, "validate_config", lambda value: value)
    monkeypatch.setattr(runner.ref_infer, "reference_source", lambda _cfg: "native_vae")
    monkeypatch.setattr(runner.ref_infer, "_dtype_from_cfg", lambda _cfg: torch.float32)
    monkeypatch.setattr(
        runner,
        "_eval_rows",
        lambda *_args, **_kwargs: [
            {"case_id": "case", "prompt": "prompt", "seed": 42, "multi_reference_images": ["ref.png"]}
        ],
    )
    monkeypatch.setattr(runner, "load_flux2_pipeline", lambda *_args, **_kwargs: pipe)
    monkeypatch.setattr(runner, "freeze_base_model", lambda _pipe: None)
    monkeypatch.setattr(runner.ref_infer, "encode_prompt_compat", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runner.ref_infer,
        "prepare_inference_latents",
        lambda *_args, **_kwargs: {
            "latents_packed": pure_noise.clone(),
            "img_ids": torch.zeros(1, 2, 4),
            "timesteps": torch.tensor([1000.0]),
            "latent_meta": {"height": 64, "width": 64},
        },
    )
    monkeypatch.setattr(
        runner,
        "_load_ref_latent_grids",
        lambda *_args, **_kwargs: torch.zeros(1, 1, 4, 8, 8),
    )
    monkeypatch.setattr(
        runner,
        "pack_transformer_latents_compat",
        lambda *_args: (clean_reference.clone(), torch.zeros(1, 2, 4)),
    )
    monkeypatch.setattr(runner, "build_reference_token_bank", lambda *_args, **_kwargs: object())

    def capture_case(**kwargs):
        captured.update(kwargs)
        return {"metadata_path": "metadata.json", "image_paths": [], "gate_trace_path": "trace.json"}

    monkeypatch.setattr(runner, "_run_case_method", capture_case)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runner.run(
        SimpleNamespace(
            reference_conditioning_mode=runner.DEFAULT_REFERENCE_CONDITIONING_MODE,
            config="config.yaml",
            checkpoint=None,
            methods=["v2_8a_static_a064_d080"],
            device="cpu",
            root=tmp_path,
            output_root=tmp_path / "out",
            eval_set=tmp_path / "cases.jsonl",
            manifest_schema=runner.FIXED_REFERENCE_SCHEMA,
            guidance_scale=1.0,
            negative_ref_mode="same",
            ref_latent_scale=1.0,
            init_ref_latent_blend=0.5,
            edit_strength=0.0,
            height=64,
            width=64,
            num_inference_steps=1,
        )
    )

    blended = captured["latent_inputs"]
    assert torch.equal(blended["latents_packed"], (pure_noise + clean_reference) / 2)
    assert blended["initial_noise_sha256"] == runner._packed_latent_sha256(pure_noise)
