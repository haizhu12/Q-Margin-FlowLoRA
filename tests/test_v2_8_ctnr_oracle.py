import re

import torch

import scripts.run_v2_8_ctnr_oracle as runner
from scripts.run_v2_8_ctnr_oracle import (
    generation_seed_for_method,
    planned_method_names,
    reset_scheduler_for_new_trajectory,
    supported_method_names,
    temporal_high_token_uses_full_bank,
)


def test_initial_packed_latent_fingerprint_is_stable_and_cached_once_per_seed(monkeypatch):
    latent_inputs = {"latents_packed": torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)}
    calls = []
    real_hash = runner._packed_latent_sha256

    def recording_hash(value):
        calls.append(value)
        return real_hash(value)

    monkeypatch.setattr(runner, "_packed_latent_sha256", recording_hash)

    first = runner._ensure_initial_noise_sha256(latent_inputs)
    second = runner._ensure_initial_noise_sha256(latent_inputs)

    assert first is latent_inputs
    assert second is latent_inputs
    assert len(calls) == 1
    assert re.fullmatch(r"[0-9a-f]{64}", latent_inputs["initial_noise_sha256"])
    same_content = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    changed_content = same_content.clone()
    changed_content[0, 0, 0] += 1
    assert runner._packed_latent_sha256(same_content) == latent_inputs["initial_noise_sha256"]
    assert runner._packed_latent_sha256(changed_content) != latent_inputs["initial_noise_sha256"]


def test_v2_8a_ctnr_oracle_default_methods_match_validation_design():
    assert planned_method_names() == [
        "v2_8a_static_a064_d080",
        "v2_8a_static_a048_d096",
        "v2_8a_static_a032_d112",
        "v2_8a_static_a016_d128",
        "v2_8a_static_a000_d144",
        "v2_8_high_token_direct",
        "v2_8_duplicate512_cluster",
    ]


def test_v2_8_legacy_methods_remain_supported_for_reproducibility():
    methods = supported_method_names()
    assert "v2_8_static_coreset" in methods
    assert "v2_8_static_detail_heavy" in methods
    assert "v2_8_dyn_balanced_late_detail" in methods
    assert methods.index("v2_8a_static_a064_d080") < methods.index("v2_8_static_coreset")


def test_v2_8a_static_sweep_method_names_are_ordered_by_lower_anchor_budget():
    assert planned_method_names()[:5] == [
        "v2_8a_static_a064_d080",
        "v2_8a_static_a048_d096",
        "v2_8a_static_a032_d112",
        "v2_8a_static_a016_d128",
        "v2_8a_static_a000_d144",
    ]


def test_v2_8_legacy_default_from_minimal_validation_remains_supported():
    assert all(
        method in supported_method_names()
        for method in [
            "v2_8_static_coreset",
            "v2_8_high_token_direct",
            "v2_8_duplicate512_cluster",
            "v2_8_static_coverage",
            "v2_8_static_detail_heavy",
            "v2_8_static_novelty",
            "v2_8_static_ref1_heavy",
            "v2_8_static_ref2_heavy",
            "v2_8_static_random",
            "v2_8_dyn_balanced_late_detail",
            "v2_8_dyn_coverage_balanced_detail",
            "v2_8_dyn_balanced_novelty_detail",
            "v2_8_dyn_ref_alternate_detail",
        ]
    )


def test_v2_8a_default_keeps_high_token_and_duplicate_controls_after_sweep():
    assert planned_method_names()[5:] == [
        "v2_8_high_token_direct",
        "v2_8_duplicate512_cluster",
    ]


def test_v2_10_query_conditioned_methods_are_supported_but_not_default():
    methods = supported_method_names()
    for method in [
        "v2_10_qc_all_a064_q080",
        "v2_10_qc_late_a064_q080",
        "v2_10_qc_all_a032_q112",
        "v2_10_qc_late_a032_q112",
    ]:
        assert method in methods
        assert method not in planned_method_names()


def test_v2_11_temporal_high_token_methods_are_supported_but_not_default():
    methods = supported_method_names()
    for method in [
        "v2_11_high_late",
        "v2_11_high_last",
        "v2_11_high_early",
        "v2_11_high_mid",
    ]:
        assert method in methods
        assert method not in planned_method_names()


def test_v2_14_phase2_ablation_methods_are_supported_but_not_default():
    methods = supported_method_names()
    expected = [
        "v2_14_a5_same_a064_seed0",
        "v2_14_a5_same_a064_seed1",
        "v2_14_a5_same_a064_seed2",
        "v2_14_a5_same_a064_seed3",
        "v2_14_a5_same_a064_seed4",
        "v2_14_a5_same_a048_seed0",
        "v2_14_a5_same_a048_seed1",
        "v2_14_a5_same_a048_seed2",
        "v2_14_a5_same_a048_seed3",
        "v2_14_a5_same_a048_seed4",
        "v2_14_b1_anchor_nearest_grid",
        "v2_14_b1_anchor_uniform_stride",
        "v2_14_b1_anchor_random",
        "v2_14_b1_anchor_top_detail",
        "v2_14_c1_detail_local_residual",
        "v2_14_c1_detail_token_l2",
        "v2_14_c1_detail_random",
        "v2_14_c1_detail_lowest_residual",
        "v2_14_c2_mask_on",
        "v2_14_c2_mask_off",
        "v2_14_d1_per_ref_quota",
        "v2_14_d1_global_quota",
        "v2_14_g1_independent_noise_a064_seed0",
        "v2_14_g1_independent_noise_a064_seed1",
        "v2_14_g1_independent_noise_a064_seed2",
    ]
    for method in expected:
        assert method in methods
        assert method not in planned_method_names()


def test_v2_14_seeded_ablation_methods_change_generation_seed_only_when_requested():
    assert generation_seed_for_method(20260624, "v2_14_a5_same_a064_seed0") == 20260624
    assert generation_seed_for_method(20260624, "v2_8a_static_a064_d080") == 20260624
    assert generation_seed_for_method(20260624, "v2_14_a5_same_a064_seed1") != 20260624
    assert generation_seed_for_method(20260624, "v2_14_a5_same_a064_seed2") != generation_seed_for_method(
        20260624,
        "v2_14_a5_same_a064_seed1",
    )
    assert generation_seed_for_method(20260624, "v2_14_g1_independent_noise_a064_seed0") != 20260624


def test_v2_11_temporal_high_token_schedule_for_four_step_eval():
    assert [temporal_high_token_uses_full_bank("v2_11_high_late", step=i, num_steps=4) for i in range(4)] == [
        False,
        False,
        True,
        True,
    ]
    assert [temporal_high_token_uses_full_bank("v2_11_high_last", step=i, num_steps=4) for i in range(4)] == [
        False,
        False,
        False,
        True,
    ]
    assert [temporal_high_token_uses_full_bank("v2_11_high_early", step=i, num_steps=4) for i in range(4)] == [
        True,
        True,
        False,
        False,
    ]
    assert [temporal_high_token_uses_full_bank("v2_11_high_mid", step=i, num_steps=4) for i in range(4)] == [
        False,
        True,
        True,
        False,
    ]


def test_reset_scheduler_for_new_trajectory_clears_diffusers_step_state():
    class FakeScheduler:
        def __init__(self):
            self.begin_index = None
            self._step_index = 4

        def set_begin_index(self, value):
            self.begin_index = value

    scheduler = FakeScheduler()

    reset_scheduler_for_new_trajectory(scheduler)

    assert scheduler.begin_index == 0
    assert scheduler._step_index is None
