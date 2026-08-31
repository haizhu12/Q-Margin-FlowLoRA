import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import scripts.run_v2_8_ctnr_oracle as ctnr_runner
from qmargin.flux2_utils import (
    DEFAULT_REFERENCE_CONDITIONING_MODE,
    REFERENCE_CONDITIONING_MODES,
    ConditionBundle,
    predict_velocity_compat,
)


QMARGIN_ROOT = Path(__file__).resolve().parents[1]


def _load_external_wrapper(module_name: str):
    configured_path = os.environ.get("QMARGIN_EXTERNAL_WRAPPER")
    if not configured_path:
        pytest.skip(
            "QMARGIN_EXTERNAL_WRAPPER is unset; external wrapper integration is unavailable"
        )
    wrapper_path = Path(configured_path).expanduser()
    if not wrapper_path.is_file():
        pytest.skip(
            "QMARGIN_EXTERNAL_WRAPPER does not point to a valid wrapper file: "
            f"{wrapper_path}"
        )
    spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
    if spec is None or spec.loader is None:
        pytest.skip(
            "QMARGIN_EXTERNAL_WRAPPER could not be imported as a Python module: "
            f"{wrapper_path}"
        )
    wrapper = importlib.util.module_from_spec(spec)
    missing = object()
    previous_module = sys.modules.get(module_name, missing)
    sys.modules[module_name] = wrapper
    try:
        spec.loader.exec_module(wrapper)
    finally:
        if previous_module is missing:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    return wrapper


def test_load_external_wrapper_registers_module_during_execution(
    monkeypatch, tmp_path
):
    wrapper_path = tmp_path / "dataclass_wrapper.py"
    wrapper_path.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class WrapperConfig:\n"
        "    value: int = 1\n",
        encoding="utf-8",
    )
    module_name = "qmargin_external_dataclass_wrapper_test"
    monkeypatch.setenv("QMARGIN_EXTERNAL_WRAPPER", str(wrapper_path))

    wrapper = _load_external_wrapper(module_name)

    assert wrapper.WrapperConfig().value == 1
    assert module_name not in sys.modules


class RecordingTransformer:
    def __init__(self, *, supports_extract=False, returns_combined=True):
        self.supports_extract = supports_extract
        self.returns_combined = returns_combined
        self.calls = []

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        return_dict=True,
        num_ref_tokens=0,
        ref_fixed_timestep=0.0,
        **kwargs,
    ):
        if not self.supports_extract and "kv_cache_mode" in kwargs:
            raise AssertionError("joint mode must not forward kv_cache_mode")
        self.calls.append(
            {
                "hidden_states": hidden_states.clone(),
                "img_ids": img_ids.clone(),
                "num_ref_tokens": num_ref_tokens,
                "ref_fixed_timestep": ref_fixed_timestep,
                **kwargs,
            }
        )
        if self.returns_combined:
            values = torch.arange(hidden_states.shape[1], device=hidden_states.device).view(1, -1, 1)
            return (values.expand_as(hidden_states).float(),)
        return (hidden_states[:, -2:] + 10.0,)


class NoExtractTransformer:
    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, img_ids=None, return_dict=True):
        return (hidden_states,)


class InternalExtractTransformer:
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        return_dict=True,
        num_ref_tokens=0,
        ref_fixed_timestep=0.0,
        **kwargs,
    ):
        return (hidden_states,)


InternalExtractTransformer.forward.__qmargin_internal_kwargs__ = {"kv_cache_mode"}


class SampleOutput:
    def __init__(self, sample):
        self.sample = sample


class FakePipe:
    def __init__(self, transformer):
        self.transformer = transformer


def make_condition(*, ref_tokens=2, fixed_timestep=0.375):
    ref_latents = torch.tensor([[[100.0, 101.0], [102.0, 103.0]]])[:, :ref_tokens]
    ref_img_ids = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 1.0, 0.0]]])[:, :ref_tokens]
    return ConditionBundle(
        prompt_embeds=torch.zeros(1, 3, 2),
        combined_prompt_embeds=torch.zeros(1, 3, 2),
        txt_ids=torch.zeros(3, 4),
        img_ids=torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]),
        ref_latents=ref_latents,
        ref_img_ids=ref_img_ids,
        num_ref_tokens=ref_tokens,
        ref_fixed_timestep=fixed_timestep,
    )


def test_joint_append_current_timestep_orders_tokens_and_slices_leading_prediction():
    transformer = RecordingTransformer()
    latents = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    prediction = predict_velocity_compat(
        FakePipe(transformer),
        latents,
        torch.tensor([0.75]),
        make_condition(),
        kv_cache_mode="extract",
    )

    call = transformer.calls[0]
    assert call["hidden_states"].tolist() == [[[1.0, 2.0], [3.0, 4.0], [100.0, 101.0], [102.0, 103.0]]]
    assert call["img_ids"][0, :, 0].tolist() == [0.0, 0.0, 10.0, 10.0]
    assert "kv_cache_mode" not in call
    assert prediction[:, :, 0].tolist() == [[0.0, 1.0]]


def test_kv_extract_accepts_transformer_declared_internal_cache_keyword():
    prediction = predict_velocity_compat(
        FakePipe(InternalExtractTransformer()),
        torch.zeros(1, 2, 2),
        torch.tensor([0.5]),
        make_condition(),
        reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
    )

    assert prediction.shape == (1, 2, 2)


def test_kv_extract_prepend_fixed_timestep_orders_tokens_and_slices_trailing_prediction():
    transformer = RecordingTransformer(supports_extract=True)
    latents = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    prediction = predict_velocity_compat(
        FakePipe(transformer),
        latents,
        torch.tensor([0.75]),
        make_condition(fixed_timestep=0.125),
        reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
        kv_cache_mode="ignored-by-contract",
    )

    call = transformer.calls[0]
    assert call["hidden_states"].tolist() == [[[100.0, 101.0], [102.0, 103.0], [1.0, 2.0], [3.0, 4.0]]]
    assert call["img_ids"][0, :, 0].tolist() == [10.0, 10.0, 0.0, 0.0]
    assert call["kv_cache_mode"] == "extract"
    assert call["num_ref_tokens"] == 2
    assert call["ref_fixed_timestep"] == pytest.approx(0.125)
    assert prediction[:, :, 0].tolist() == [[2.0, 3.0]]


def test_extract_accepts_two_element_transformer_tuple():
    transformer = RecordingTransformer(supports_extract=True)
    original_forward = transformer.forward

    def forward_with_kv(*args, **kwargs):
        return (*original_forward(*args, **kwargs), {"kv": "cache"})

    transformer.forward = forward_with_kv
    prediction = predict_velocity_compat(
        FakePipe(transformer),
        torch.zeros(1, 2, 2),
        torch.tensor([0.5]),
        make_condition(),
        reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
    )

    assert prediction[:, :, 0].tolist() == [[2.0, 3.0]]


def test_extract_accepts_sample_bearing_output_object():
    transformer = RecordingTransformer(supports_extract=True)
    original_forward = transformer.forward
    transformer.forward = lambda *args, **kwargs: SampleOutput(original_forward(*args, **kwargs)[0])

    prediction = predict_velocity_compat(
        FakePipe(transformer),
        torch.zeros(1, 2, 2),
        torch.tensor([0.5]),
        make_condition(),
        reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
    )

    assert prediction[:, :, 0].tolist() == [[2.0, 3.0]]


def test_kv_extract_requires_transformer_capability_instead_of_silently_filtering():
    with pytest.raises(RuntimeError, match="kv_cache_mode"):
        predict_velocity_compat(
            FakePipe(NoExtractTransformer()),
            torch.zeros(1, 2, 2),
            torch.tensor([0.5]),
            make_condition(),
            reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
        )


def test_kv_extract_rejects_native_reference_count_without_native_latents():
    condition = make_condition()
    condition.ref_latents = None

    with pytest.raises(RuntimeError, match="num_ref_tokens.*ref_latents"):
        predict_velocity_compat(
            FakePipe(RecordingTransformer(supports_extract=True)),
            torch.zeros(1, 2, 2),
            torch.tensor([0.5]),
            condition,
            reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
        )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda condition: setattr(condition, "img_ids", None), "generated img_ids"),
        (lambda condition: setattr(condition, "ref_img_ids", None), "ref_img_ids"),
        (lambda condition: setattr(condition, "num_ref_tokens", 1), "num_ref_tokens"),
        (lambda condition: setattr(condition, "ref_img_ids", condition.ref_img_ids[:, :1]), "ref_img_ids sequence"),
        (lambda condition: setattr(condition, "ref_img_ids", torch.cat([condition.ref_img_ids, condition.ref_img_ids], dim=0)), "ref_img_ids batch"),
        (lambda condition: setattr(condition, "img_ids", torch.cat([condition.img_ids, condition.img_ids], dim=0)), "generated img_ids batch"),
    ],
)
def test_reference_concatenation_rejects_missing_or_mismatched_ids(mutate, error):
    condition = make_condition()
    mutate(condition)

    with pytest.raises((RuntimeError, ValueError), match=error):
        predict_velocity_compat(
            FakePipe(RecordingTransformer()),
            torch.zeros(1, 2, 2),
            torch.tensor([0.5]),
            condition,
        )


def test_zero_reference_tokens_use_standard_forward_without_extract_mode():
    transformer = RecordingTransformer(supports_extract=True, returns_combined=False)
    latents = torch.zeros(1, 2, 2)

    prediction = predict_velocity_compat(
        FakePipe(transformer),
        latents,
        torch.tensor([0.5]),
        make_condition(ref_tokens=0),
        reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
    )

    call = transformer.calls[0]
    assert call["hidden_states"].shape[1] == 2
    assert "kv_cache_mode" not in call
    assert prediction.shape == latents.shape


def test_runner_rejects_unknown_mode_before_loading_config_or_model(monkeypatch):
    args = SimpleNamespace(reference_conditioning_mode="not-a-mode")
    monkeypatch.setattr(ctnr_runner.ref_infer, "load_config", lambda *_args: pytest.fail("config must not load"))

    with pytest.raises(ValueError, match="reference_conditioning_mode"):
        ctnr_runner.run(args)


def test_runner_case_threads_mode_to_positive_negative_calls_and_trace(monkeypatch, tmp_path):
    class Route:
        policy = "test"
        num_ref_tokens = 2
        per_ref_counts = [2]
        unique_source_count = 2
        flat_indices = torch.tensor([[0, 1]])
        tokens = torch.ones(1, 2, 2)
        ids = torch.ones(1, 2, 4)

    class Scheduler:
        def step(self, model_output, timestep, sample, return_dict=False):
            return (sample,)

    pipe = SimpleNamespace(scheduler=Scheduler())
    calls = []
    saved = {}
    real_save_inference_outputs = ctnr_runner.ref_infer.save_inference_outputs
    monkeypatch.setattr(ctnr_runner, "_route_for_method", lambda *args, **kwargs: Route())
    monkeypatch.setattr(
        ctnr_runner,
        "build_native_vae_reference_condition_bundle",
        lambda **kwargs: make_condition(ref_tokens=kwargs["ref_latents"].shape[1]),
    )
    monkeypatch.setattr(ctnr_runner.ref_infer, "transformer_timestep_from_scheduler", lambda *_args: torch.tensor(0.5))
    monkeypatch.setattr(ctnr_runner.ref_infer, "normalize_timestep_from_schedule", lambda *_args: torch.tensor(0.5))
    monkeypatch.setattr(
        ctnr_runner.ref_infer,
        "predict_velocity_compat",
        lambda pipe, latents, timestep, condition, **kwargs: calls.append(
            (condition.num_ref_tokens, kwargs["reference_conditioning_mode"])
        )
        or torch.zeros_like(latents),
    )
    monkeypatch.setattr(ctnr_runner.ref_infer, "decode_packed_latents_to_images", lambda *_args: [])
    def save_inference_outputs(output, args, *extra_args):
        saved.update(output=output, args=args)
        result = real_save_inference_outputs(output, args, *extra_args)
        saved["result"] = result
        return result

    monkeypatch.setattr(ctnr_runner.ref_infer, "save_inference_outputs", save_inference_outputs)

    ctnr_runner._run_case_method(
        pipe=pipe,
        cfg={"reference_adapter": {}},
        row={"case_id": "case", "prompt": "prompt", "multi_reference_images": []},
        method_name="v2_8_static_coreset",
        bank=SimpleNamespace(num_ref_images=1, tokens_per_image=2),
        prompt_bundle=object(),
        negative_prompt_bundle=object(),
        latent_inputs={"latents_packed": torch.zeros(1, 2, 2), "img_ids": torch.zeros(1, 2, 4), "timesteps": torch.tensor([1.0]), "latent_meta": {}},
        latent_seed=1,
        output_root=tmp_path,
        height=8,
        width=8,
        num_inference_steps=1,
        guidance_scale=2.0,
        negative_ref_mode="drop",
        ref_latent_scale=1.0,
        reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
        device="cpu",
        dtype=torch.float32,
    )

    assert calls == [
        (2, "kv_extract_prepend_fixed_timestep"),
        (0, "kv_extract_prepend_fixed_timestep"),
    ]
    assert saved["args"].checkpoint is None
    metadata = json.loads(Path(saved["result"]["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["checkpoint"] is None
    assert saved["args"].reference_conditioning_mode == "kv_extract_prepend_fixed_timestep"
    assert saved["output"]["gate_trace"][0]["reference_conditioning_mode"] == "kv_extract_prepend_fixed_timestep"


def test_wrapper_propagates_mode_into_runner_summary_status_and_candidate_rows(monkeypatch, tmp_path):
    wrapper = _load_external_wrapper("qmargin_single_ref_wrapper_test")

    manifest = tmp_path / "cases.jsonl"
    manifest.write_text(json.dumps({"case_id": "case", "ref_paths": [], "prompt": "prompt"}) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        manifest=manifest,
        run_dir=tmp_path / "run",
        qmargin_root=QMARGIN_ROOT,
        config="cfg.yaml",
        height=8,
        width=8,
        num_inference_steps=1,
        guidance_scale=1.0,
        negative_ref_mode="same",
        ref_latent_scale=1.0,
        init_ref_latent_blend=0.0,
        reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
        checkpoint=None,
        allow_base_mismatch=False,
        allow_gate_override=False,
        device="cpu",
        methods=["route"],
        comparison_mode="smoke",
        notes="test",
        allow_diagnostic_formal_budget=False,
    )
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    import scripts.run_v2_8_ctnr_oracle as actual_runner

    monkeypatch.setattr(actual_runner, "validate_reference_conditioning_mode", lambda mode: mode)
    monkeypatch.setattr(
        actual_runner,
        "run",
        lambda runner_args: {
            "case_count": 1,
            "checkpoint_loaded": False,
            "checkpoint": None,
            "outputs": [{"case_id": "case", "method": "route", "image_paths": [], "metadata_path": "m", "gate_trace_path": "t"}],
            "reference_conditioning_mode": runner_args.reference_conditioning_mode,
        },
    )

    wrapper.main()

    status = json.loads((args.run_dir / "smoke_status.json").read_text(encoding="utf-8"))
    row = json.loads((args.run_dir / "manifests" / "candidate_manifest_all_routes.jsonl").read_text(encoding="utf-8"))
    summary = json.loads((args.run_dir / "ctnr_summary.json").read_text(encoding="utf-8"))
    assert summary["reference_conditioning_mode"] == "kv_extract_prepend_fixed_timestep"
    assert status["reference_conditioning_mode"] == "kv_extract_prepend_fixed_timestep"
    assert status["protocol"]["reference_conditioning_mode"] == "kv_extract_prepend_fixed_timestep"
    assert row["reference_conditioning_mode"] == "kv_extract_prepend_fixed_timestep"
    assert DEFAULT_REFERENCE_CONDITIONING_MODE == "joint_append_current_timestep"
    assert wrapper.REFERENCE_CONDITIONING_MODES == REFERENCE_CONDITIONING_MODES
    assert wrapper.DEFAULT_REFERENCE_CONDITIONING_MODE == DEFAULT_REFERENCE_CONDITIONING_MODE


def test_wrapper_formal_preflight_requires_exact_routes_and_budget_before_side_effects(tmp_path):
    wrapper = _load_external_wrapper("qmargin_single_ref_wrapper_formal_test")

    args = SimpleNamespace(
        run_dir=tmp_path / "formal_run",
        comparison_mode="formal",
        notes="formal candidate contract",
        allow_diagnostic_formal_budget=False,
        height=512,
        width=512,
        num_inference_steps=31,
        methods=list(wrapper.DEFAULT_METHODS) + ["v2_8a_static_a064_d080"],
    )

    with pytest.raises(ValueError, match="exactly"):
        wrapper.validate_run_budget(args)
    assert not args.run_dir.exists()


def test_wrapper_rejects_formal_target_input_before_creating_run_dir(monkeypatch, tmp_path):
    wrapper = _load_external_wrapper("qmargin_single_ref_wrapper_target_test")

    manifest = tmp_path / "target_input.jsonl"
    manifest.write_text(json.dumps({"case_id": "case", "seed": 42, "target_path": "forbidden.png"}) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        manifest=manifest,
        run_dir=tmp_path / "formal_run",
        comparison_mode="formal",
        notes="target-free output requested",
        allow_diagnostic_formal_budget=False,
        height=512,
        width=512,
        num_inference_steps=30,
        methods=list(wrapper.DEFAULT_METHODS),
    )
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="target/oracle"):
        wrapper.main()
    assert not args.run_dir.exists()


def test_wrapper_diagnostic_override_still_rejects_bad_method_and_target_before_output(monkeypatch, tmp_path):
    wrapper = _load_external_wrapper("qmargin_single_ref_wrapper_diagnostic_preflight_test")

    manifest = tmp_path / "diagnostic_target_input.jsonl"
    manifest.write_text(json.dumps({"case_id": "case", "target_path": "forbidden.png"}) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        manifest=manifest,
        run_dir=tmp_path / "formal_run",
        comparison_mode="formal",
        notes="formal diagnostic",
        allow_diagnostic_formal_budget=True,
        height=128,
        width=128,
        num_inference_steps=4,
        methods=["v2_11_high_late"],
    )
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(wrapper, "collect_env", lambda: (_ for _ in ()).throw(AssertionError("preflight bypassed")))

    with pytest.raises(ValueError, match="route IDs|target/oracle"):
        wrapper.main()
    assert not args.run_dir.exists()


@pytest.mark.parametrize("rows", [
    [{"case_id": "case", "seed": "random"}],
    [{"case_id": "case", "seed": True}],
    [{"case_id": "case", "seed": 42.0}],
    [{"case_id": "case", "seed": 42}, {"case_id": "case", "seed": 43}],
    [{"case_id": "case", "seed": 42, "target_assisted": True}],
    [{"case_id": "case", "seed": 42, "oracle_selection": "fixed"}],
    [{"case_id": "case", "seed": 42, "selector": {"method": "oracle"}}],
    [{"case_id": "case", "seed": 42, "selection_method": "oracle_based"}],
])
def test_wrapper_formal_manifest_contract_rejects_before_output_or_runner_import(monkeypatch, tmp_path, rows):
    wrapper = _load_external_wrapper("qmargin_single_ref_wrapper_manifest_contract_test")

    manifest = tmp_path / "cases.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        manifest=manifest,
        run_dir=tmp_path / "formal_run",
        qmargin_root=QMARGIN_ROOT,
        comparison_mode="formal",
        notes="formal run",
        allow_diagnostic_formal_budget=False,
        height=512,
        width=512,
        num_inference_steps=30,
        methods=list(wrapper.DEFAULT_METHODS),
    )
    calls = {"env": 0, "runner_import": 0}
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(wrapper, "collect_env", lambda: calls.__setitem__("env", calls["env"] + 1) or {})
    import builtins

    real_import = builtins.__import__

    def spy_import(name, *args, **kwargs):
        if name == "scripts.run_v2_8_ctnr_oracle":
            calls["runner_import"] += 1
            raise AssertionError("runner import must not occur")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", spy_import)
    with pytest.raises(ValueError):
        wrapper.main()
    assert not args.run_dir.exists()
    assert calls == {"env": 0, "runner_import": 0}


def test_wrapper_formal_manifest_rejects_malformed_json_with_line_context(monkeypatch, tmp_path):
    wrapper = _load_external_wrapper("qmargin_single_ref_wrapper_manifest_json_test")
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text('{"case_id": "case", "seed": 42\n', encoding="utf-8")
    args = SimpleNamespace(
        manifest=manifest,
        run_dir=tmp_path / "formal_run",
        comparison_mode="formal",
        notes="formal run",
        allow_diagnostic_formal_budget=False,
        height=512,
        width=512,
        num_inference_steps=30,
        methods=list(wrapper.DEFAULT_METHODS),
    )
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    with pytest.raises(ValueError, match=r"line 1"):
        wrapper.main()
    assert not args.run_dir.exists()
