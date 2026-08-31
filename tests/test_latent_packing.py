import pytest
import torch

from qmargin.flux2_utils import (
    pack_transformer_latents_compat,
    pool_reference_latent_grids_2d,
    select_reference_native_latent_tokens_2d,
)


class FakePipe:
    @staticmethod
    def _pack_latents(x):
        b, c, h, w = x.shape
        return x.reshape(b, c, h * w).permute(0, 2, 1)

    @staticmethod
    def _prepare_latent_ids(x):
        b, _, h, w = x.shape
        return torch.zeros(b, h * w, 4, device=x.device)


class BadIdPipe(FakePipe):
    @staticmethod
    def _prepare_latent_ids(x):
        b, _, h, w = x.shape
        return torch.zeros(b, h * w, 3, device=x.device)


class GridIdPipe(FakePipe):
    @staticmethod
    def _prepare_latent_ids(x):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
        ids = torch.stack([torch.zeros_like(yy), yy, xx, torch.zeros_like(yy)], dim=-1).float()
        return ids.reshape(1, h * w, 4).expand(b, -1, -1).contiguous()


def test_pack_transformer_latents_uses_pipeline_helpers_without_repatchifying():
    z = torch.randn(2, 16, 8, 8)

    packed, img_ids = pack_transformer_latents_compat(FakePipe(), z)

    assert packed.shape == (2, 64, 16)
    assert img_ids.shape == (2, 64, 4)


def test_pack_transformer_latents_rejects_non_flux2_4d_ids():
    z = torch.randn(2, 16, 8, 8)

    with pytest.raises(AssertionError, match="4 coordinates"):
        pack_transformer_latents_compat(BadIdPipe(), z)


def test_pool_reference_latent_grids_2d_returns_exact_tokens_and_ref_ids():
    grids = torch.arange(1 * 2 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 2, 1, 4, 4)

    pooled, pooled_ids = pool_reference_latent_grids_2d(
        FakePipe(),
        grids,
        tokens_per_image=4,
        pool_grid=(2, 2),
        img_id_device=grids.device,
        img_id_dtype=torch.float32,
    )

    expected_first = torch.tensor([2.5, 4.5, 10.5, 12.5]).view(1, 4, 1)
    expected_second = torch.tensor([18.5, 20.5, 26.5, 28.5]).view(1, 4, 1)
    assert pooled.shape == (1, 8, 1)
    assert torch.allclose(pooled[:, :4], expected_first)
    assert torch.allclose(pooled[:, 4:], expected_second)
    assert pooled_ids.shape == (1, 8, 4)
    assert pooled_ids[0, :, 0].tolist() == [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]
    assert pooled_ids[0, :, 1].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert pooled_ids[0, :, 2].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_pool_reference_latent_grids_2d_rejects_mismatched_token_budget():
    grids = torch.zeros(1, 1, 2, 4, 4)

    with pytest.raises(ValueError, match="tokens_per_image"):
        pool_reference_latent_grids_2d(FakePipe(), grids, tokens_per_image=5, pool_grid=(2, 2))


def test_select_reference_native_nearest_keeps_original_tokens_and_ids():
    grids = torch.arange(1 * 1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 1, 4, 4)

    selected, selected_ids = select_reference_native_latent_tokens_2d(
        GridIdPipe(),
        grids,
        tokens_per_image=4,
        mode="native_nearest_grid_2d",
        pool_grid=(2, 2),
    )

    assert selected.shape == (1, 4, 1)
    assert selected[0, :, 0].tolist() == [0.0, 3.0, 12.0, 15.0]
    assert selected_ids.shape == (1, 4, 4)
    assert selected_ids[0, :, 0].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert selected_ids[0, :, 1].tolist() == [0.0, 0.0, 3.0, 3.0]
    assert selected_ids[0, :, 2].tolist() == [0.0, 3.0, 0.0, 3.0]


def test_select_reference_native_coreset_includes_detail_spike():
    grids = torch.zeros(1, 1, 1, 4, 4)
    grids[0, 0, 0, 1, 1] = 100.0

    selected, selected_ids = select_reference_native_latent_tokens_2d(
        GridIdPipe(),
        grids,
        tokens_per_image=6,
        mode="native_coreset",
        pool_grid=(2, 3),
        coreset_anchor_tokens=4,
    )

    assert selected.shape == (1, 6, 1)
    assert 100.0 in selected[0, :, 0].tolist()
    assert selected_ids.shape == (1, 6, 4)
