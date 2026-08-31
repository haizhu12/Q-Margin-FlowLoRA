import torch

from qmargin.flux2_utils import select_reference_native_latent_tokens_2d
from qmargin.ref.ctnr_diagnostics import (
    build_reference_token_bank,
    duplicate_coreset_to_full_tokens,
    policy_for_schedule_step,
    select_anchor_detail_ablation_reference_tokens,
    select_anchor_detail_reference_tokens,
    select_ctnr_reference_tokens,
    select_query_conditioned_reference_tokens,
)


class TinyPackPipe:
    def _pack_latents(self, latents):
        batch, channels, height, width = latents.shape
        return latents.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

    def _prepare_latent_ids(self, latents):
        batch, _channels, height, width = latents.shape
        rows = []
        for y in range(height):
            for x in range(width):
                rows.append((0.0, float(y), float(x), 0.0))
        return torch.tensor(rows, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1).contiguous()


def _tiny_bank():
    pipe = TinyPackPipe()
    ref_latent_grids = torch.arange(1 * 2 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 2, 3, 4, 4)
    bank = build_reference_token_bank(
        pipe,
        ref_latent_grids,
        tokens_per_image=4,
        pool_grid=[2, 2],
        anchor_tokens_per_ref=2,
    )
    return pipe, ref_latent_grids, bank


def test_ctnr_coreset_policy_matches_existing_native_coreset_selection():
    pipe, ref_latent_grids, bank = _tiny_bank()

    route = select_ctnr_reference_tokens(bank, policy="coreset")
    expected_tokens, expected_ids = select_reference_native_latent_tokens_2d(
        pipe,
        ref_latent_grids,
        tokens_per_image=4,
        mode="native_coreset",
        pool_grid=[2, 2],
        coreset_anchor_tokens=2,
    )

    assert torch.equal(route.tokens, expected_tokens)
    assert torch.equal(route.ids, expected_ids)
    assert route.num_ref_tokens == 8
    assert route.per_ref_counts == [4, 4]


def test_ctnr_static_policies_return_expected_budget_and_sorted_ids():
    _pipe, _ref_latent_grids, bank = _tiny_bank()

    for policy in ["coverage", "detail_heavy", "novelty", "random"]:
        route = select_ctnr_reference_tokens(bank, policy=policy, random_seed=7)

        assert route.tokens.shape == (1, 8, 3)
        assert route.ids.shape == (1, 8, 4)
        assert route.num_ref_tokens == 8
        assert sum(route.per_ref_counts) == 8
        keys = route.ids[0, :, 0] * 1.0e6 + route.ids[0, :, 1] * 1.0e3 + route.ids[0, :, 2]
        assert torch.equal(keys, torch.sort(keys).values)


def test_anchor_detail_selector_matches_coreset_at_baseline_anchor_budget():
    _pipe, _ref_latent_grids, bank = _tiny_bank()

    baseline = select_ctnr_reference_tokens(bank, policy="coreset")
    anchor_detail = select_anchor_detail_reference_tokens(
        bank,
        anchor_tokens_per_ref=2,
        policy_name="tiny_a002_d002",
    )

    assert torch.equal(anchor_detail.tokens, baseline.tokens)
    assert torch.equal(anchor_detail.ids, baseline.ids)
    assert torch.equal(anchor_detail.flat_indices, baseline.flat_indices)
    assert anchor_detail.policy == "tiny_a002_d002"
    assert anchor_detail.per_ref_counts == [4, 4]


def test_anchor_detail_selector_supports_detail_only_budget():
    _pipe, _ref_latent_grids, bank = _tiny_bank()

    route = select_anchor_detail_reference_tokens(bank, anchor_tokens_per_ref=0)

    assert route.tokens.shape == (1, 8, 3)
    assert route.ids.shape == (1, 8, 4)
    assert route.policy == "anchor_detail_a000_d004"
    assert route.num_ref_tokens == 8
    assert route.unique_source_count == 8
    keys = route.ids[0, :, 0] * 1.0e6 + route.ids[0, :, 1] * 1.0e3 + route.ids[0, :, 2]
    assert torch.equal(keys, torch.sort(keys).values)


def test_anchor_detail_ablation_matches_official_route_with_default_policies():
    _pipe, _ref_latent_grids, bank = _tiny_bank()

    official = select_anchor_detail_reference_tokens(
        bank,
        anchor_tokens_per_ref=2,
        tokens_per_image=4,
        policy_name="official",
    )
    ablation = select_anchor_detail_ablation_reference_tokens(
        bank,
        anchor_tokens_per_ref=2,
        tokens_per_image=4,
        anchor_policy="nearest_grid",
        detail_policy="local_residual",
        mask_anchor_tokens=True,
        quota_policy="per_ref",
        policy_name="ablation_default",
    )

    assert torch.equal(ablation.tokens, official.tokens)
    assert torch.equal(ablation.ids, official.ids)
    assert torch.equal(ablation.flat_indices, official.flat_indices)
    assert ablation.policy == "ablation_default"


def test_anchor_detail_ablation_mask_off_allows_anchor_reselection():
    _pipe, _ref_latent_grids, bank = _tiny_bank()

    masked = select_anchor_detail_ablation_reference_tokens(
        bank,
        anchor_tokens_per_ref=2,
        tokens_per_image=4,
        anchor_policy="nearest_grid",
        detail_policy="local_residual",
        mask_anchor_tokens=True,
    )
    unmasked = select_anchor_detail_ablation_reference_tokens(
        bank,
        anchor_tokens_per_ref=2,
        tokens_per_image=4,
        anchor_policy="nearest_grid",
        detail_policy="local_residual",
        mask_anchor_tokens=False,
    )

    assert unmasked.num_ref_tokens == masked.num_ref_tokens == 8
    assert unmasked.unique_source_count < masked.unique_source_count


def test_anchor_detail_ablation_global_quota_changes_per_reference_counts():
    _pipe, _ref_latent_grids, bank = _tiny_bank()
    bank.detail_scores[0, 0] = torch.arange(bank.full_tokens_per_image, dtype=torch.float32)
    bank.detail_scores[0, 1] = torch.arange(bank.full_tokens_per_image, dtype=torch.float32) + 100.0

    route = select_anchor_detail_ablation_reference_tokens(
        bank,
        anchor_tokens_per_ref=1,
        tokens_per_image=4,
        anchor_policy="nearest_grid",
        detail_policy="local_residual",
        mask_anchor_tokens=True,
        quota_policy="global",
    )

    assert route.num_ref_tokens == 8
    assert route.per_ref_counts != [4, 4]
    assert route.per_ref_counts[1] > route.per_ref_counts[0]


def test_anchor_detail_ablation_supports_uniform_and_random_anchor_controls():
    _pipe, _ref_latent_grids, bank = _tiny_bank()

    uniform = select_anchor_detail_ablation_reference_tokens(
        bank,
        anchor_tokens_per_ref=2,
        tokens_per_image=4,
        anchor_policy="uniform_stride",
        detail_policy="local_residual",
        random_seed=7,
    )
    random_a = select_anchor_detail_ablation_reference_tokens(
        bank,
        anchor_tokens_per_ref=2,
        tokens_per_image=4,
        anchor_policy="random",
        detail_policy="local_residual",
        random_seed=7,
    )
    random_b = select_anchor_detail_ablation_reference_tokens(
        bank,
        anchor_tokens_per_ref=2,
        tokens_per_image=4,
        anchor_policy="random",
        detail_policy="local_residual",
        random_seed=8,
    )

    assert uniform.num_ref_tokens == 8
    assert random_a.num_ref_tokens == 8
    assert not torch.equal(random_a.flat_indices, random_b.flat_indices)


def test_query_conditioned_selector_retrieves_query_similar_reference_tokens():
    _pipe, _ref_latent_grids, bank = _tiny_bank()
    query = bank.full_tokens[:, :, 15, :].reshape(1, 2, 3)

    route = select_query_conditioned_reference_tokens(
        bank,
        query_tokens=query,
        anchor_tokens_per_ref=0,
        tokens_per_image=2,
        policy_name="tiny_qc",
    )

    assert route.policy == "tiny_qc"
    assert route.num_ref_tokens == 4
    assert route.per_ref_counts == [2, 2]
    assert int(bank.full_tokens_per_image - 1) in route.flat_indices[0].tolist()
    assert int(bank.full_tokens_per_image * 2 - 1) in route.flat_indices[0].tolist()


def test_query_conditioned_selector_keeps_anchor_budget_and_sorted_ids():
    _pipe, _ref_latent_grids, bank = _tiny_bank()
    query = bank.full_tokens[:, 0, :1, :]

    route = select_query_conditioned_reference_tokens(
        bank,
        query_tokens=query,
        anchor_tokens_per_ref=1,
        tokens_per_image=3,
    )

    assert route.tokens.shape == (1, 6, 3)
    assert route.ids.shape == (1, 6, 4)
    assert route.num_ref_tokens == 6
    keys = route.ids[0, :, 0] * 1.0e6 + route.ids[0, :, 1] * 1.0e3 + route.ids[0, :, 2]
    assert torch.equal(keys, torch.sort(keys).values)


def test_ctnr_ref_heavy_policies_jointly_allocate_dynamic_slots():
    _pipe, _ref_latent_grids, bank = _tiny_bank()

    ref1_route = select_ctnr_reference_tokens(bank, policy="ref1_heavy")
    ref2_route = select_ctnr_reference_tokens(bank, policy="ref2_heavy")

    assert ref1_route.per_ref_counts == [5, 3]
    assert ref2_route.per_ref_counts == [3, 5]
    assert ref1_route.num_ref_tokens == 8
    assert ref2_route.num_ref_tokens == 8


def test_duplicate_coreset_to_full_tokens_uses_full_layout_without_new_unique_support():
    _pipe, _ref_latent_grids, bank = _tiny_bank()

    duplicate = duplicate_coreset_to_full_tokens(bank)

    assert duplicate.tokens.shape == (1, 32, 3)
    assert duplicate.ids.shape == (1, 32, 4)
    assert torch.equal(duplicate.ids, bank.full_ids_flat)
    assert duplicate.num_ref_tokens == 32
    assert duplicate.unique_source_count <= 8
    assert sorted(set(duplicate.source_indices[0].tolist())) == sorted(bank.coreset_flat_indices[0].tolist())


def test_ctnr_schedule_step_mapping_for_four_step_eval():
    assert policy_for_schedule_step("v2_8_dyn_balanced_late_detail", step=0, num_steps=4) == "coreset"
    assert policy_for_schedule_step("v2_8_dyn_balanced_late_detail", step=2, num_steps=4) == "detail_heavy"
    assert policy_for_schedule_step("v2_8_dyn_coverage_balanced_detail", step=0, num_steps=4) == "coverage"
    assert policy_for_schedule_step("v2_8_dyn_balanced_novelty_detail", step=1, num_steps=4) == "novelty"
    assert policy_for_schedule_step("v2_8_dyn_ref_alternate_detail", step=1, num_steps=4) == "ref1_heavy"
    assert policy_for_schedule_step("v2_8_dyn_ref_alternate_detail", step=2, num_steps=4) == "ref2_heavy"
