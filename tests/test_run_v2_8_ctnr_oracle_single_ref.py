import json
from pathlib import Path

from scripts.run_v2_8_ctnr_oracle import (
    SINGLE_REFERENCE_NO_TARGET_SCHEMA,
    _eval_rows,
    _expected_ref_tokens_for_bank,
    _trace_entry,
)


def test_single_reference_no_target_rows_convert_to_internal_multi_ref(tmp_path):
    ref_path = tmp_path / "ref.png"
    ref_path.write_bytes(b"not-an-image-but-existing")
    manifest = tmp_path / "dreambench_single_ref.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "case_id": "case_001",
                "prompt": "A photo of the object",
                "seed": 42,
                "ref_paths": [ref_path.name],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _eval_rows(manifest, tmp_path, manifest_schema=SINGLE_REFERENCE_NO_TARGET_SCHEMA)

    assert len(rows) == 1
    assert rows[0]["multi_reference_images"] == [str(ref_path)]
    assert rows[0]["num_refs"] == 1
    assert "target_image" not in rows[0]


def test_expected_ref_tokens_uses_actual_reference_count():
    class FakeBank:
        def __init__(self, num_ref_images, tokens_per_image):
            self.num_ref_images = num_ref_images
            self.tokens_per_image = tokens_per_image

    assert _expected_ref_tokens_for_bank(FakeBank(num_ref_images=1, tokens_per_image=144)) == 144
    assert _expected_ref_tokens_for_bank(FakeBank(num_ref_images=2, tokens_per_image=144)) == 288


def test_trace_entry_keeps_reference_injection_diagnostics():
    class FakeRoute:
        policy = "coreset"
        num_ref_tokens = 64
        per_ref_counts = [64]
        unique_source_count = 64

    entry = _trace_entry(
        method_name="v2_8_static_coreset",
        route=FakeRoute(),
        step=3,
        timestep=0.5,
        overlap=0.75,
        reference_conditioning_mode="kv_extract_prepend_fixed_timestep",
        condition_stats={"native_ref_l2_to_text_l2": 1.25},
        positive_interaction_stats={"native_target_update_ratio_mean": 0.02},
        negative_interaction_stats={"native_target_update_ratio_mean": 0.0},
    )

    assert entry["condition_stats"]["native_ref_l2_to_text_l2"] == 1.25
    assert entry["positive_interaction_stats"]["native_target_update_ratio_mean"] == 0.02
    assert entry["negative_interaction_stats"]["native_target_update_ratio_mean"] == 0.0
    assert entry["reference_conditioning_mode"] == "kv_extract_prepend_fixed_timestep"


def test_formal_static_trace_entry_records_exact_route_policy_and_split():
    class FakeRoute:
        policy = "v2_8a_static_a064_d080"
        num_ref_tokens = 144
        per_ref_counts = [144]
        unique_source_count = 144

    entry = _trace_entry(
        method_name="v2_8a_static_a064_d080",
        route=FakeRoute(),
        step=0,
        timestep=1.0,
        overlap=None,
    )

    assert entry["method"] == "v2_8a_static_a064_d080"
    assert entry["policy"] == "v2_8a_static_a064_d080"
    assert entry["route_split"] == {
        "anchor_tokens_per_ref": 64,
        "detail_tokens_per_ref": 80,
    }
