from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from qmargin.flux2_utils import (
    _native_coreset_indices,
    _native_detail_scores,
    _native_nearest_grid_indices,
    _native_random_indices,
    _sort_indices_by_native_ids,
    pack_transformer_latents_compat,
)


@dataclass
class ReferenceTokenBank:
    full_tokens: torch.Tensor
    full_ids: torch.Tensor
    full_tokens_flat: torch.Tensor
    full_ids_flat: torch.Tensor
    coreset_indices: torch.Tensor
    coreset_flat_indices: torch.Tensor
    anchor_indices: torch.Tensor
    detail_scores: torch.Tensor
    tokens_per_image: int
    anchor_tokens_per_ref: int
    pool_grid: tuple[int, int] | list[int] | None

    @property
    def batch_size(self) -> int:
        return int(self.full_tokens.shape[0])

    @property
    def num_ref_images(self) -> int:
        return int(self.full_tokens.shape[1])

    @property
    def full_tokens_per_image(self) -> int:
        return int(self.full_tokens.shape[2])


@dataclass
class RoutedReferenceTokens:
    tokens: torch.Tensor
    ids: torch.Tensor
    flat_indices: torch.Tensor
    source_indices: torch.Tensor
    policy: str
    per_ref_counts: list[int]
    num_ref_tokens: int
    unique_source_count: int


_SCHEDULES: dict[str, list[str]] = {
    "v2_8_dyn_balanced_late_detail": ["coreset", "coreset", "detail_heavy", "detail_heavy"],
    "v2_8_dyn_coverage_balanced_detail": ["coverage", "coreset", "detail_heavy", "detail_heavy"],
    "v2_8_dyn_balanced_novelty_detail": ["coreset", "novelty", "novelty", "detail_heavy"],
    "v2_8_dyn_ref_alternate_detail": ["coreset", "ref1_heavy", "ref2_heavy", "detail_heavy"],
}


def _layer_norm_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(tokens.float(), (tokens.shape[-1],))


def _zscore(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1.0e-6)


def _flat_offset(ref_idx: int, tokens_per_ref: int) -> int:
    return int(ref_idx) * int(tokens_per_ref)


def _sort_flat_indices_by_reference_ids(full_ids_flat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    selected_ids = full_ids_flat[indices]
    keys = (
        selected_ids[:, 0].float() * 1.0e9
        + selected_ids[:, 1].float() * 1.0e6
        + selected_ids[:, 2].float() * 1.0e3
        + selected_ids[:, 3].float()
    )
    return indices[torch.argsort(keys)]


def build_reference_token_bank(
    pipe,
    ref_latent_grids: torch.Tensor,
    *,
    tokens_per_image: int,
    pool_grid: tuple[int, int] | list[int] | None = None,
    anchor_tokens_per_ref: int = 64,
) -> ReferenceTokenBank:
    if ref_latent_grids.ndim != 5:
        raise ValueError(f"ref_latent_grids must be [B,N,C,H,W], got {tuple(ref_latent_grids.shape)}")
    tokens_per_image = int(tokens_per_image)
    anchor_tokens_per_ref = int(anchor_tokens_per_ref)
    if tokens_per_image < 1:
        raise ValueError("tokens_per_image must be >= 1")
    if anchor_tokens_per_ref < 0:
        raise ValueError("anchor_tokens_per_ref must be >= 0")

    batch_size, num_ref_images = ref_latent_grids.shape[:2]
    flat = ref_latent_grids.reshape(batch_size * num_ref_images, *ref_latent_grids.shape[2:])
    packed, packed_ids = pack_transformer_latents_compat(pipe, flat)
    packed = packed.reshape(batch_size, num_ref_images, packed.shape[1], packed.shape[2])
    packed_ids = packed_ids.reshape(batch_size, num_ref_images, packed_ids.shape[1], packed_ids.shape[2]).to(
        device=packed.device
    )

    full_ids = packed_ids.clone()
    coreset_by_batch = []
    coreset_flat_by_batch = []
    anchors_by_batch = []
    detail_by_batch = []
    for batch_idx in range(batch_size):
        coreset_refs = []
        coreset_flat_refs = []
        anchor_refs = []
        detail_refs = []
        for ref_idx in range(num_ref_images):
            tokens = packed[batch_idx, ref_idx]
            native_ids = full_ids[batch_idx, ref_idx].clone()
            native_ids[:, 0] = float(10 * (ref_idx + 1))
            full_ids[batch_idx, ref_idx] = native_ids
            coreset = _native_coreset_indices(
                tokens,
                native_ids,
                tokens_per_image,
                pool_grid,
                coreset_anchor_tokens=anchor_tokens_per_ref,
            )
            anchor_count = min(anchor_tokens_per_ref, tokens_per_image, tokens.shape[0])
            if anchor_count > 0:
                anchors = _native_nearest_grid_indices(native_ids, anchor_count, None)
            else:
                anchors = torch.empty(0, device=tokens.device, dtype=torch.long)
            detail_scores = _native_detail_scores(tokens, native_ids)
            coreset_refs.append(coreset)
            coreset_flat_refs.append(coreset + _flat_offset(ref_idx, packed.shape[2]))
            anchor_refs.append(anchors)
            detail_refs.append(detail_scores)
        coreset_by_batch.append(torch.stack(coreset_refs, dim=0))
        coreset_flat_by_batch.append(torch.cat(coreset_flat_refs, dim=0))
        anchors_by_batch.append(torch.stack(anchor_refs, dim=0))
        detail_by_batch.append(torch.stack(detail_refs, dim=0))

    full_tokens_flat = packed.reshape(batch_size, num_ref_images * packed.shape[2], packed.shape[-1])
    full_ids_flat = full_ids.reshape(batch_size, num_ref_images * packed.shape[2], full_ids.shape[-1])
    return ReferenceTokenBank(
        full_tokens=packed,
        full_ids=full_ids,
        full_tokens_flat=full_tokens_flat,
        full_ids_flat=full_ids_flat,
        coreset_indices=torch.stack(coreset_by_batch, dim=0),
        coreset_flat_indices=torch.stack(coreset_flat_by_batch, dim=0),
        anchor_indices=torch.stack(anchors_by_batch, dim=0),
        detail_scores=torch.stack(detail_by_batch, dim=0),
        tokens_per_image=tokens_per_image,
        anchor_tokens_per_ref=anchor_tokens_per_ref,
        pool_grid=pool_grid,
    )


def _top_detail_indices(
    tokens: torch.Tensor,
    native_ids: torch.Tensor,
    detail_scores: torch.Tensor,
    *,
    total_count: int,
    anchor_count: int,
) -> torch.Tensor:
    total_count = int(total_count)
    anchor_count = max(0, min(int(anchor_count), total_count, tokens.shape[0]))
    if anchor_count > 0:
        anchors = _native_nearest_grid_indices(native_ids, anchor_count, None)
    else:
        anchors = torch.empty(0, device=tokens.device, dtype=torch.long)
    detail_count = total_count - int(anchors.numel())
    if detail_count <= 0:
        return _sort_indices_by_native_ids(native_ids, anchors[:total_count])
    scores = detail_scores.float().clone()
    if anchors.numel() > 0:
        scores[anchors] = -torch.inf
    details = torch.topk(scores, k=detail_count, largest=True).indices
    return _sort_indices_by_native_ids(native_ids, torch.cat([anchors, details], dim=0))


def _uniform_stride_indices(native_ids: torch.Tensor, count: int) -> torch.Tensor:
    count = int(count)
    if count <= 0:
        return torch.empty(0, device=native_ids.device, dtype=torch.long)
    if native_ids.shape[0] < count:
        raise ValueError(f"Cannot select {count} tokens from only {native_ids.shape[0]} native tokens.")
    ordered = _sort_indices_by_native_ids(
        native_ids,
        torch.arange(native_ids.shape[0], device=native_ids.device, dtype=torch.long),
    )
    positions = torch.linspace(0, ordered.numel() - 1, count, device=native_ids.device).round().long()
    selected = ordered[positions]
    if torch.unique(selected).numel() == selected.numel():
        return _sort_indices_by_native_ids(native_ids, selected)
    used = set(int(value) for value in selected.detach().cpu().tolist())
    filled = [int(value) for value in selected.detach().cpu().tolist()]
    for candidate in ordered.detach().cpu().tolist():
        candidate = int(candidate)
        if candidate not in used:
            filled.append(candidate)
            used.add(candidate)
            if len(filled) >= count:
                break
    return _sort_indices_by_native_ids(
        native_ids,
        torch.tensor(filled[:count], device=native_ids.device, dtype=torch.long),
    )


def _anchor_indices_for_policy(
    bank: ReferenceTokenBank,
    *,
    batch_idx: int,
    ref_idx: int,
    anchor_count: int,
    anchor_policy: str,
    random_seed: int,
) -> torch.Tensor:
    tokens = bank.full_tokens[batch_idx, ref_idx]
    native_ids = bank.full_ids[batch_idx, ref_idx]
    anchor_count = max(0, min(int(anchor_count), tokens.shape[0]))
    if anchor_count <= 0:
        return torch.empty(0, device=tokens.device, dtype=torch.long)
    anchor_policy = str(anchor_policy)
    if anchor_policy == "nearest_grid":
        return _native_nearest_grid_indices(native_ids, anchor_count, None)
    if anchor_policy == "uniform_stride":
        return _uniform_stride_indices(native_ids, anchor_count)
    if anchor_policy == "random":
        return _native_random_indices(
            native_ids,
            anchor_count,
            seed=int(random_seed) + int(batch_idx) * 1009 + int(ref_idx) * 9176,
        )
    if anchor_policy == "top_detail":
        anchors = torch.topk(bank.detail_scores[batch_idx, ref_idx].float(), k=anchor_count, largest=True).indices
        return _sort_indices_by_native_ids(native_ids, anchors)
    raise ValueError(f"Unsupported anchor ablation policy: {anchor_policy}")


def _detail_scores_for_policy(
    bank: ReferenceTokenBank,
    *,
    batch_idx: int,
    ref_idx: int,
    detail_policy: str,
    random_seed: int,
) -> torch.Tensor:
    tokens = bank.full_tokens[batch_idx, ref_idx]
    detail_policy = str(detail_policy)
    if detail_policy == "local_residual":
        return bank.detail_scores[batch_idx, ref_idx].float()
    if detail_policy == "token_l2":
        return torch.linalg.vector_norm(tokens.float(), dim=-1)
    if detail_policy == "lowest_residual":
        return -bank.detail_scores[batch_idx, ref_idx].float()
    if detail_policy == "random":
        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(int(random_seed) + int(batch_idx) * 1009 + int(ref_idx) * 9176 + 7919)
        return torch.rand(tokens.shape[0], device=tokens.device, generator=generator)
    raise ValueError(f"Unsupported detail ablation policy: {detail_policy}")


def _detail_indices_for_policy(
    bank: ReferenceTokenBank,
    *,
    batch_idx: int,
    ref_idx: int,
    detail_count: int,
    anchors: torch.Tensor,
    detail_policy: str,
    mask_anchor_tokens: bool,
    random_seed: int,
) -> torch.Tensor:
    detail_count = int(detail_count)
    if detail_count <= 0:
        return torch.empty(0, device=bank.full_tokens.device, dtype=torch.long)
    scores = _detail_scores_for_policy(
        bank,
        batch_idx=batch_idx,
        ref_idx=ref_idx,
        detail_policy=detail_policy,
        random_seed=random_seed,
    ).clone()
    if bool(mask_anchor_tokens) and anchors.numel() > 0:
        scores[anchors] = -torch.inf
    finite = torch.isfinite(scores)
    if int(finite.sum().item()) < detail_count:
        raise ValueError(
            f"Cannot select {detail_count} detail tokens with mask_anchor_tokens={mask_anchor_tokens}; "
            f"only {int(finite.sum().item())} candidates are available."
        )
    return torch.topk(scores, k=detail_count, largest=True).indices


def _route_from_flat_indices(
    bank: ReferenceTokenBank,
    *,
    flat_by_batch: list[torch.Tensor],
    policy: str,
) -> RoutedReferenceTokens:
    tokens_by_batch = []
    ids_by_batch = []
    sorted_flat_by_batch = []
    for batch_idx, flat_indices in enumerate(flat_by_batch):
        flat_indices = _sort_flat_indices_by_reference_ids(bank.full_ids_flat[batch_idx], flat_indices)
        sorted_flat_by_batch.append(flat_indices)
        tokens_by_batch.append(bank.full_tokens_flat[batch_idx, flat_indices])
        ids_by_batch.append(bank.full_ids_flat[batch_idx, flat_indices])

    flat = torch.stack(sorted_flat_by_batch, dim=0)
    per_ref_counts = []
    if flat.shape[0] > 0:
        first = flat[0] // bank.full_tokens_per_image
        per_ref_counts = [int((first == ref_idx).sum().item()) for ref_idx in range(bank.num_ref_images)]
    return RoutedReferenceTokens(
        tokens=torch.stack(tokens_by_batch, dim=0),
        ids=torch.stack(ids_by_batch, dim=0),
        flat_indices=flat,
        source_indices=flat,
        policy=policy,
        per_ref_counts=per_ref_counts,
        num_ref_tokens=int(flat.shape[1]),
        unique_source_count=int(torch.unique(flat[0]).numel()),
    )


def _novelty_scores(bank: ReferenceTokenBank, batch_idx: int, ref_idx: int) -> torch.Tensor:
    tokens = _layer_norm_tokens(bank.full_tokens[batch_idx, ref_idx])
    other_refs = [idx for idx in range(bank.num_ref_images) if idx != ref_idx]
    if not other_refs:
        return torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.float32)
    other = torch.cat([_layer_norm_tokens(bank.full_tokens[batch_idx, idx]) for idx in other_refs], dim=0)
    return torch.cdist(tokens, other).min(dim=1).values.float()


def _novelty_indices(
    bank: ReferenceTokenBank,
    batch_idx: int,
    ref_idx: int,
    *,
    total_count: int,
    anchor_count: int,
) -> torch.Tensor:
    tokens = bank.full_tokens[batch_idx, ref_idx]
    native_ids = bank.full_ids[batch_idx, ref_idx]
    anchor_count = max(0, min(int(anchor_count), int(total_count), tokens.shape[0]))
    if anchor_count > 0:
        anchors = _native_nearest_grid_indices(native_ids, anchor_count, None)
    else:
        anchors = torch.empty(0, device=tokens.device, dtype=torch.long)
    remaining = int(total_count) - int(anchors.numel())
    if remaining <= 0:
        return _sort_indices_by_native_ids(native_ids, anchors[:total_count])
    score = _zscore(bank.detail_scores[batch_idx, ref_idx]) + _zscore(_novelty_scores(bank, batch_idx, ref_idx))
    if anchors.numel() > 0:
        score = score.clone()
        score[anchors] = -torch.inf
    details = torch.topk(score, k=remaining, largest=True).indices
    return _sort_indices_by_native_ids(native_ids, torch.cat([anchors, details], dim=0))


def _ref_heavy_dynamic_counts(total_dynamic: int, heavy_ref_idx: int, num_refs: int) -> list[int]:
    if num_refs != 2:
        base = total_dynamic // num_refs
        counts = [base for _ in range(num_refs)]
        for idx in range(total_dynamic - base * num_refs):
            counts[idx] += 1
        return counts
    heavy = int(math.ceil(float(total_dynamic) * 0.70))
    heavy = max(0, min(heavy, total_dynamic))
    counts = [total_dynamic - heavy, total_dynamic - heavy]
    counts[int(heavy_ref_idx)] = heavy
    counts[1 - int(heavy_ref_idx)] = total_dynamic - heavy
    return counts


def _local_indices_for_policy(
    bank: ReferenceTokenBank,
    *,
    batch_idx: int,
    ref_idx: int,
    policy: str,
    total_count: int,
    random_seed: int,
) -> torch.Tensor:
    tokens = bank.full_tokens[batch_idx, ref_idx]
    native_ids = bank.full_ids[batch_idx, ref_idx]
    detail_scores = bank.detail_scores[batch_idx, ref_idx]
    total_count = int(total_count)
    if total_count < 1:
        return torch.empty(0, device=tokens.device, dtype=torch.long)
    if total_count > tokens.shape[0]:
        raise ValueError(f"Cannot select {total_count} tokens from only {tokens.shape[0]} native tokens.")

    if policy == "coreset" and total_count == bank.tokens_per_image:
        return bank.coreset_indices[batch_idx, ref_idx]
    if policy == "coverage":
        pool_grid = bank.pool_grid if total_count == bank.tokens_per_image else None
        return _native_nearest_grid_indices(native_ids, total_count, pool_grid)
    if policy == "detail_heavy":
        detail_anchor_count = min(max(bank.anchor_tokens_per_ref // 2, 0), total_count)
        return _top_detail_indices(
            tokens,
            native_ids,
            detail_scores,
            total_count=total_count,
            anchor_count=detail_anchor_count,
        )
    if policy == "novelty":
        return _novelty_indices(
            bank,
            batch_idx,
            ref_idx,
            total_count=total_count,
            anchor_count=min(bank.anchor_tokens_per_ref, total_count),
        )
    if policy in {"ref1_heavy", "ref2_heavy", "random"}:
        if policy == "random":
            return _native_random_indices(
                native_ids,
                total_count,
                seed=int(random_seed) + int(batch_idx) * 1009 + int(ref_idx) * 9176,
            )
        return _top_detail_indices(
            tokens,
            native_ids,
            detail_scores,
            total_count=total_count,
            anchor_count=min(bank.anchor_tokens_per_ref, total_count),
        )
    if policy == "coreset":
        return _top_detail_indices(
            tokens,
            native_ids,
            detail_scores,
            total_count=total_count,
            anchor_count=min(bank.anchor_tokens_per_ref, total_count),
        )
    raise ValueError(f"Unsupported CTNR policy: {policy}")


def _target_counts_for_policy(bank: ReferenceTokenBank, policy: str) -> list[int]:
    if policy == "ref1_heavy":
        total_dynamic = bank.tokens_per_image * bank.num_ref_images - bank.anchor_tokens_per_ref * bank.num_ref_images
        return [bank.anchor_tokens_per_ref + value for value in _ref_heavy_dynamic_counts(total_dynamic, 0, bank.num_ref_images)]
    if policy == "ref2_heavy":
        total_dynamic = bank.tokens_per_image * bank.num_ref_images - bank.anchor_tokens_per_ref * bank.num_ref_images
        return [bank.anchor_tokens_per_ref + value for value in _ref_heavy_dynamic_counts(total_dynamic, 1, bank.num_ref_images)]
    return [bank.tokens_per_image for _ in range(bank.num_ref_images)]


def select_ctnr_reference_tokens(
    bank: ReferenceTokenBank,
    *,
    policy: str,
    random_seed: int = 0,
) -> RoutedReferenceTokens:
    policy = str(policy)
    counts = _target_counts_for_policy(bank, policy)
    tokens_by_batch = []
    ids_by_batch = []
    flat_by_batch = []
    for batch_idx in range(bank.batch_size):
        flat_indices_refs = []
        for ref_idx, count in enumerate(counts):
            local = _local_indices_for_policy(
                bank,
                batch_idx=batch_idx,
                ref_idx=ref_idx,
                policy=policy,
                total_count=int(count),
                random_seed=random_seed,
            )
            flat_indices_refs.append(local + _flat_offset(ref_idx, bank.full_tokens_per_image))
        flat_indices = torch.cat(flat_indices_refs, dim=0)
        flat_by_batch.append(flat_indices)
        tokens_by_batch.append(bank.full_tokens_flat[batch_idx, flat_indices])
        ids_by_batch.append(bank.full_ids_flat[batch_idx, flat_indices])

    flat = torch.stack(flat_by_batch, dim=0)
    return RoutedReferenceTokens(
        tokens=torch.stack(tokens_by_batch, dim=0),
        ids=torch.stack(ids_by_batch, dim=0),
        flat_indices=flat,
        source_indices=flat,
        policy=policy,
        per_ref_counts=list(counts),
        num_ref_tokens=int(flat.shape[1]),
        unique_source_count=int(torch.unique(flat[0]).numel()),
    )


def select_anchor_detail_reference_tokens(
    bank: ReferenceTokenBank,
    *,
    anchor_tokens_per_ref: int,
    tokens_per_image: int | None = None,
    policy_name: str | None = None,
) -> RoutedReferenceTokens:
    tokens_per_image = bank.tokens_per_image if tokens_per_image is None else int(tokens_per_image)
    anchor_tokens_per_ref = int(anchor_tokens_per_ref)
    if tokens_per_image < 1:
        raise ValueError("tokens_per_image must be >= 1")
    if tokens_per_image > bank.full_tokens_per_image:
        raise ValueError(
            f"Cannot select {tokens_per_image} tokens from only {bank.full_tokens_per_image} native tokens."
        )
    if anchor_tokens_per_ref < 0:
        raise ValueError("anchor_tokens_per_ref must be >= 0")
    if anchor_tokens_per_ref > tokens_per_image:
        raise ValueError("anchor_tokens_per_ref must be <= tokens_per_image")

    detail_tokens_per_ref = tokens_per_image - anchor_tokens_per_ref
    policy = policy_name or f"anchor_detail_a{anchor_tokens_per_ref:03d}_d{detail_tokens_per_ref:03d}"
    counts = [tokens_per_image for _ in range(bank.num_ref_images)]
    tokens_by_batch = []
    ids_by_batch = []
    flat_by_batch = []
    for batch_idx in range(bank.batch_size):
        flat_indices_refs = []
        for ref_idx in range(bank.num_ref_images):
            local = _top_detail_indices(
                bank.full_tokens[batch_idx, ref_idx],
                bank.full_ids[batch_idx, ref_idx],
                bank.detail_scores[batch_idx, ref_idx],
                total_count=tokens_per_image,
                anchor_count=anchor_tokens_per_ref,
            )
            flat_indices_refs.append(local + _flat_offset(ref_idx, bank.full_tokens_per_image))
        flat_indices = torch.cat(flat_indices_refs, dim=0)
        flat_by_batch.append(flat_indices)
        tokens_by_batch.append(bank.full_tokens_flat[batch_idx, flat_indices])
        ids_by_batch.append(bank.full_ids_flat[batch_idx, flat_indices])

    flat = torch.stack(flat_by_batch, dim=0)
    return RoutedReferenceTokens(
        tokens=torch.stack(tokens_by_batch, dim=0),
        ids=torch.stack(ids_by_batch, dim=0),
        flat_indices=flat,
        source_indices=flat,
        policy=policy,
        per_ref_counts=counts,
        num_ref_tokens=int(flat.shape[1]),
        unique_source_count=int(torch.unique(flat[0]).numel()),
    )


def select_anchor_detail_ablation_reference_tokens(
    bank: ReferenceTokenBank,
    *,
    anchor_tokens_per_ref: int,
    tokens_per_image: int | None = None,
    anchor_policy: str = "nearest_grid",
    detail_policy: str = "local_residual",
    mask_anchor_tokens: bool = True,
    quota_policy: str = "per_ref",
    random_seed: int = 0,
    policy_name: str | None = None,
) -> RoutedReferenceTokens:
    tokens_per_image = bank.tokens_per_image if tokens_per_image is None else int(tokens_per_image)
    anchor_tokens_per_ref = int(anchor_tokens_per_ref)
    if tokens_per_image < 1:
        raise ValueError("tokens_per_image must be >= 1")
    if tokens_per_image > bank.full_tokens_per_image:
        raise ValueError(
            f"Cannot select {tokens_per_image} tokens from only {bank.full_tokens_per_image} native tokens."
        )
    if anchor_tokens_per_ref < 0:
        raise ValueError("anchor_tokens_per_ref must be >= 0")
    if anchor_tokens_per_ref > tokens_per_image:
        raise ValueError("anchor_tokens_per_ref must be <= tokens_per_image")

    detail_tokens_per_ref = tokens_per_image - anchor_tokens_per_ref
    quota_policy = str(quota_policy)
    policy = policy_name or (
        "anchor_detail_ablation_"
        f"a{anchor_tokens_per_ref:03d}_d{detail_tokens_per_ref:03d}_"
        f"anchor-{anchor_policy}_detail-{detail_policy}_"
        f"mask-{int(bool(mask_anchor_tokens))}_quota-{quota_policy}"
    )

    if quota_policy == "per_ref":
        flat_by_batch = []
        for batch_idx in range(bank.batch_size):
            flat_indices_refs = []
            for ref_idx in range(bank.num_ref_images):
                anchors = _anchor_indices_for_policy(
                    bank,
                    batch_idx=batch_idx,
                    ref_idx=ref_idx,
                    anchor_count=anchor_tokens_per_ref,
                    anchor_policy=anchor_policy,
                    random_seed=random_seed,
                )
                details = _detail_indices_for_policy(
                    bank,
                    batch_idx=batch_idx,
                    ref_idx=ref_idx,
                    detail_count=detail_tokens_per_ref,
                    anchors=anchors,
                    detail_policy=detail_policy,
                    mask_anchor_tokens=mask_anchor_tokens,
                    random_seed=random_seed,
                )
                local = _sort_indices_by_native_ids(
                    bank.full_ids[batch_idx, ref_idx],
                    torch.cat([anchors, details], dim=0),
                )
                flat_indices_refs.append(local + _flat_offset(ref_idx, bank.full_tokens_per_image))
            flat_by_batch.append(torch.cat(flat_indices_refs, dim=0))
        return _route_from_flat_indices(bank, flat_by_batch=flat_by_batch, policy=policy)

    if quota_policy == "global":
        total_budget = tokens_per_image * bank.num_ref_images
        flat_by_batch = []
        for batch_idx in range(bank.batch_size):
            anchor_flat_refs = []
            global_candidate_scores = []
            global_candidate_flat = []
            for ref_idx in range(bank.num_ref_images):
                anchors = _anchor_indices_for_policy(
                    bank,
                    batch_idx=batch_idx,
                    ref_idx=ref_idx,
                    anchor_count=anchor_tokens_per_ref,
                    anchor_policy=anchor_policy,
                    random_seed=random_seed,
                )
                anchor_flat = anchors + _flat_offset(ref_idx, bank.full_tokens_per_image)
                anchor_flat_refs.append(anchor_flat)
                scores = _detail_scores_for_policy(
                    bank,
                    batch_idx=batch_idx,
                    ref_idx=ref_idx,
                    detail_policy=detail_policy,
                    random_seed=random_seed,
                ).clone()
                if bool(mask_anchor_tokens) and anchors.numel() > 0:
                    scores[anchors] = -torch.inf
                local = torch.arange(bank.full_tokens_per_image, device=scores.device, dtype=torch.long)
                finite = torch.isfinite(scores)
                global_candidate_scores.append(scores[finite])
                global_candidate_flat.append(local[finite] + _flat_offset(ref_idx, bank.full_tokens_per_image))
            anchor_flat = torch.cat(anchor_flat_refs, dim=0)
            detail_count = total_budget - int(anchor_flat.numel())
            if detail_count < 0:
                raise ValueError("Global quota anchor count exceeds total route budget.")
            if detail_count == 0:
                flat_by_batch.append(anchor_flat)
                continue
            scores = torch.cat(global_candidate_scores, dim=0)
            candidates = torch.cat(global_candidate_flat, dim=0)
            if scores.numel() < detail_count:
                raise ValueError(f"Cannot select {detail_count} global detail tokens from {scores.numel()} candidates.")
            details = candidates[torch.topk(scores, k=detail_count, largest=True).indices]
            flat_by_batch.append(torch.cat([anchor_flat, details], dim=0))
        return _route_from_flat_indices(bank, flat_by_batch=flat_by_batch, policy=policy)

    raise ValueError(f"Unsupported quota ablation policy: {quota_policy}")


def _query_conditioned_indices(
    bank: ReferenceTokenBank,
    *,
    batch_idx: int,
    ref_idx: int,
    query_tokens: torch.Tensor,
    total_count: int,
    anchor_count: int,
    detail_weight: float,
) -> torch.Tensor:
    tokens = bank.full_tokens[batch_idx, ref_idx]
    native_ids = bank.full_ids[batch_idx, ref_idx]
    total_count = int(total_count)
    anchor_count = max(0, min(int(anchor_count), total_count, tokens.shape[0]))
    if total_count < 1:
        return torch.empty(0, device=tokens.device, dtype=torch.long)
    if total_count > tokens.shape[0]:
        raise ValueError(f"Cannot select {total_count} tokens from only {tokens.shape[0]} native tokens.")
    if anchor_count > 0:
        anchors = _native_nearest_grid_indices(native_ids, anchor_count, None)
    else:
        anchors = torch.empty(0, device=tokens.device, dtype=torch.long)
    remaining = total_count - int(anchors.numel())
    if remaining <= 0:
        return _sort_indices_by_native_ids(native_ids, anchors[:total_count])

    ref_norm = F.normalize(tokens.float(), dim=-1)
    query = query_tokens[batch_idx].to(device=tokens.device).float()
    query_norm = F.normalize(query, dim=-1)
    query_score = torch.matmul(ref_norm, query_norm.transpose(0, 1)).max(dim=1).values
    if float(detail_weight) != 0.0:
        query_score = query_score + float(detail_weight) * _zscore(bank.detail_scores[batch_idx, ref_idx])
    if anchors.numel() > 0:
        query_score = query_score.clone()
        query_score[anchors] = -torch.inf
    selected = torch.topk(query_score, k=remaining, largest=True).indices
    return _sort_indices_by_native_ids(native_ids, torch.cat([anchors, selected], dim=0))


def select_query_conditioned_reference_tokens(
    bank: ReferenceTokenBank,
    *,
    query_tokens: torch.Tensor,
    anchor_tokens_per_ref: int,
    tokens_per_image: int | None = None,
    detail_weight: float = 0.0,
    policy_name: str | None = None,
) -> RoutedReferenceTokens:
    if query_tokens.ndim != 3:
        raise ValueError(f"query_tokens must be [B,L,D], got {tuple(query_tokens.shape)}")
    if int(query_tokens.shape[0]) != bank.batch_size:
        raise ValueError("query_tokens batch size must match bank batch size.")
    if int(query_tokens.shape[-1]) != int(bank.full_tokens.shape[-1]):
        raise ValueError("query_tokens feature dimension must match reference token dimension.")

    tokens_per_image = bank.tokens_per_image if tokens_per_image is None else int(tokens_per_image)
    anchor_tokens_per_ref = int(anchor_tokens_per_ref)
    if tokens_per_image < 1:
        raise ValueError("tokens_per_image must be >= 1")
    if anchor_tokens_per_ref < 0:
        raise ValueError("anchor_tokens_per_ref must be >= 0")
    if anchor_tokens_per_ref > tokens_per_image:
        raise ValueError("anchor_tokens_per_ref must be <= tokens_per_image")

    query_count = tokens_per_image - anchor_tokens_per_ref
    policy = policy_name or f"query_conditioned_a{anchor_tokens_per_ref:03d}_q{query_count:03d}"
    counts = [tokens_per_image for _ in range(bank.num_ref_images)]
    tokens_by_batch = []
    ids_by_batch = []
    flat_by_batch = []
    for batch_idx in range(bank.batch_size):
        flat_indices_refs = []
        for ref_idx in range(bank.num_ref_images):
            local = _query_conditioned_indices(
                bank,
                batch_idx=batch_idx,
                ref_idx=ref_idx,
                query_tokens=query_tokens,
                total_count=tokens_per_image,
                anchor_count=anchor_tokens_per_ref,
                detail_weight=float(detail_weight),
            )
            flat_indices_refs.append(local + _flat_offset(ref_idx, bank.full_tokens_per_image))
        flat_indices = torch.cat(flat_indices_refs, dim=0)
        flat_by_batch.append(flat_indices)
        tokens_by_batch.append(bank.full_tokens_flat[batch_idx, flat_indices])
        ids_by_batch.append(bank.full_ids_flat[batch_idx, flat_indices])

    flat = torch.stack(flat_by_batch, dim=0)
    return RoutedReferenceTokens(
        tokens=torch.stack(tokens_by_batch, dim=0),
        ids=torch.stack(ids_by_batch, dim=0),
        flat_indices=flat,
        source_indices=flat,
        policy=policy,
        per_ref_counts=counts,
        num_ref_tokens=int(flat.shape[1]),
        unique_source_count=int(torch.unique(flat[0]).numel()),
    )


def full_reference_tokens(bank: ReferenceTokenBank) -> RoutedReferenceTokens:
    flat = torch.arange(bank.full_tokens_flat.shape[1], device=bank.full_tokens_flat.device, dtype=torch.long)
    flat = flat.unsqueeze(0).expand(bank.batch_size, -1).contiguous()
    return RoutedReferenceTokens(
        tokens=bank.full_tokens_flat,
        ids=bank.full_ids_flat,
        flat_indices=flat,
        source_indices=flat,
        policy="high_token_direct",
        per_ref_counts=[bank.full_tokens_per_image for _ in range(bank.num_ref_images)],
        num_ref_tokens=int(flat.shape[1]),
        unique_source_count=int(flat.shape[1]),
    )


def duplicate_coreset_to_full_tokens(bank: ReferenceTokenBank) -> RoutedReferenceTokens:
    tokens_by_batch = []
    source_by_batch = []
    for batch_idx in range(bank.batch_size):
        dup_refs = []
        source_refs = []
        for ref_idx in range(bank.num_ref_images):
            tokens = bank.full_tokens[batch_idx, ref_idx]
            native_ids = bank.full_ids[batch_idx, ref_idx]
            selected = bank.coreset_indices[batch_idx, ref_idx]
            norm_tokens = _layer_norm_tokens(tokens)
            selected_tokens = norm_tokens[selected]
            feature_dist = torch.cdist(norm_tokens, selected_tokens).square()
            coords = native_ids[:, 1:3].float()
            selected_coords = coords[selected]
            spatial_dist = torch.cdist(coords, selected_coords).square() * 0.25
            assignment = torch.argmin(feature_dist + spatial_dist, dim=1)
            assignment[selected] = torch.arange(selected.numel(), device=tokens.device)
            source_local = selected[assignment]
            dup_refs.append(tokens[source_local])
            source_refs.append(source_local + _flat_offset(ref_idx, bank.full_tokens_per_image))
        tokens_by_batch.append(torch.cat(dup_refs, dim=0))
        source_by_batch.append(torch.cat(source_refs, dim=0))
    source = torch.stack(source_by_batch, dim=0)
    flat = torch.arange(bank.full_tokens_flat.shape[1], device=bank.full_tokens_flat.device, dtype=torch.long)
    flat = flat.unsqueeze(0).expand(bank.batch_size, -1).contiguous()
    return RoutedReferenceTokens(
        tokens=torch.stack(tokens_by_batch, dim=0),
        ids=bank.full_ids_flat,
        flat_indices=flat,
        source_indices=source,
        policy="duplicate512_cluster",
        per_ref_counts=[bank.full_tokens_per_image for _ in range(bank.num_ref_images)],
        num_ref_tokens=int(flat.shape[1]),
        unique_source_count=int(torch.unique(source[0]).numel()),
    )


def policy_for_schedule_step(schedule_name: str, *, step: int, num_steps: int) -> str:
    schedule_name = str(schedule_name)
    if schedule_name not in _SCHEDULES:
        raise ValueError(f"Unsupported CTNR schedule: {schedule_name}")
    policies = _SCHEDULES[schedule_name]
    step = int(step)
    num_steps = max(int(num_steps), 1)
    if num_steps == len(policies):
        idx = max(0, min(step, len(policies) - 1))
        return policies[idx]
    fraction = step / max(num_steps - 1, 1)
    idx = min(int(math.floor(fraction * len(policies))), len(policies) - 1)
    return policies[idx]


def schedule_names() -> list[str]:
    return list(_SCHEDULES.keys())
