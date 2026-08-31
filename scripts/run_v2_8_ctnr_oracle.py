from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qmargin import inference_runtime as ref_infer
from qmargin.config_utils import validate_config
from qmargin.flux2_utils import (
    DEFAULT_REFERENCE_CONDITIONING_MODE,
    REFERENCE_CONDITIONING_MODES,
    build_native_vae_reference_condition_bundle,
    freeze_base_model,
    load_flux2_pipeline,
    pack_transformer_latents_compat,
    validate_reference_conditioning_mode,
)
from qmargin.ref.ctnr_diagnostics import (
    RoutedReferenceTokens,
    build_reference_token_bank,
    duplicate_coreset_to_full_tokens,
    full_reference_tokens,
    policy_for_schedule_step,
    schedule_names,
    select_anchor_detail_ablation_reference_tokens,
    select_anchor_detail_reference_tokens,
    select_ctnr_reference_tokens,
    select_query_conditioned_reference_tokens,
)
from scripts.validate_fixed_reference_eval import validate_fixed_reference_eval

FIXED_REFERENCE_SCHEMA = "fixed_reference"
SINGLE_REFERENCE_NO_TARGET_SCHEMA = "single_reference_no_target"
MANIFEST_SCHEMAS = (FIXED_REFERENCE_SCHEMA, SINGLE_REFERENCE_NO_TARGET_SCHEMA)


STATIC_POLICY_METHODS = {
    "v2_8_static_coreset": "coreset",
    "v2_8_static_coverage": "coverage",
    "v2_8_static_detail_heavy": "detail_heavy",
    "v2_8_static_novelty": "novelty",
    "v2_8_static_ref1_heavy": "ref1_heavy",
    "v2_8_static_ref2_heavy": "ref2_heavy",
    "v2_8_static_random": "random",
}

ANCHOR_DETAIL_METHODS = {
    "v2_8a_static_a064_d080": (64, 80),
    "v2_8a_static_a048_d096": (48, 96),
    "v2_8a_static_a032_d112": (32, 112),
    "v2_8a_static_a016_d128": (16, 128),
    "v2_8a_static_a000_d144": (0, 144),
}

V2_8A_STATIC_SWEEP_METHODS = list(ANCHOR_DETAIL_METHODS.keys())

QUERY_CONDITIONED_METHODS = {
    "v2_10_qc_all_a064_q080": {"anchor": 64, "query": 80, "schedule": "all"},
    "v2_10_qc_late_a064_q080": {"anchor": 64, "query": 80, "schedule": "late"},
    "v2_10_qc_all_a032_q112": {"anchor": 32, "query": 112, "schedule": "all"},
    "v2_10_qc_late_a032_q112": {"anchor": 32, "query": 112, "schedule": "late"},
}

TEMPORAL_HIGH_TOKEN_METHODS = {
    "v2_11_high_late",
    "v2_11_high_last",
    "v2_11_high_early",
    "v2_11_high_mid",
}

HIGH_TOKEN_SEED_METHOD_SPECS = {
    f"v2_14_high_token_seed{idx}": {"seed_offset": idx * 100003}
    for idx in range(5)
}

V2_14_PHASE2_METHOD_SPECS = {
    "v2_14_b1_anchor_nearest_grid": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "nearest_grid",
        "detail_policy": "local_residual",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_b1_anchor_uniform_stride": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "uniform_stride",
        "detail_policy": "local_residual",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_b1_anchor_random": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "random",
        "detail_policy": "local_residual",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_b1_anchor_top_detail": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "top_detail",
        "detail_policy": "local_residual",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_c1_detail_local_residual": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "nearest_grid",
        "detail_policy": "local_residual",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_c1_detail_token_l2": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "nearest_grid",
        "detail_policy": "token_l2",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_c1_detail_random": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "nearest_grid",
        "detail_policy": "random",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_c1_detail_lowest_residual": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "nearest_grid",
        "detail_policy": "lowest_residual",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_c2_mask_on": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "nearest_grid",
        "detail_policy": "local_residual",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_c2_mask_off": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "nearest_grid",
        "detail_policy": "local_residual",
        "mask_anchor_tokens": False,
        "quota_policy": "per_ref",
    },
    "v2_14_d1_per_ref_quota": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "nearest_grid",
        "detail_policy": "local_residual",
        "mask_anchor_tokens": True,
        "quota_policy": "per_ref",
    },
    "v2_14_d1_global_quota": {
        "anchor": 64,
        "detail": 80,
        "anchor_policy": "nearest_grid",
        "detail_policy": "local_residual",
        "mask_anchor_tokens": True,
        "quota_policy": "global",
    },
}

V2_14_SAME_ROUTE_METHOD_SPECS = {
    **{
        f"v2_14_a5_same_a064_seed{idx}": {
            "anchor": 64,
            "detail": 80,
            "seed_offset": idx * 100003,
        }
        for idx in range(5)
    },
    **{
        f"v2_14_a5_same_a048_seed{idx}": {
            "anchor": 48,
            "detail": 96,
            "seed_offset": idx * 100003,
        }
        for idx in range(5)
    },
    **{
        f"v2_14_g1_independent_noise_a064_seed{idx}": {
            "anchor": 64,
            "detail": 80,
            "seed_offset": 500000 + idx * 100003,
        }
        for idx in range(3)
    },
}

DEFAULT_METHOD_NAMES = [
    *V2_8A_STATIC_SWEEP_METHODS,
    "v2_8_high_token_direct",
    "v2_8_duplicate512_cluster",
]

LEGACY_V2_8_METHOD_NAMES = [
    "v2_8_static_coreset",
    "v2_8_static_coverage",
    "v2_8_static_detail_heavy",
    "v2_8_static_novelty",
    "v2_8_static_ref1_heavy",
    "v2_8_static_ref2_heavy",
    "v2_8_static_random",
    *schedule_names(),
]


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve(root: str | Path, path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    return str(path)


def _eval_rows(eval_set: str | Path, root: str | Path, manifest_schema: str = FIXED_REFERENCE_SCHEMA) -> list[dict]:
    if manifest_schema == FIXED_REFERENCE_SCHEMA:
        validate_fixed_reference_eval(eval_set, root=root)
        rows = _load_jsonl(eval_set)
        for row in rows:
            row["target_image"] = _resolve(root, row["target_image"])
            row["multi_reference_images"] = [_resolve(root, path) for path in row["multi_reference_images"]]
        return rows

    if manifest_schema != SINGLE_REFERENCE_NO_TARGET_SCHEMA:
        raise ValueError(f"Unsupported manifest_schema: {manifest_schema}")

    rows = _load_jsonl(eval_set)
    for row_idx, row in enumerate(rows, 1):
        if not row.get("case_id"):
            raise ValueError(f"Missing case_id in {eval_set}:{row_idx}")
        if not row.get("prompt"):
            raise ValueError(f"Missing prompt in {eval_set}:{row_idx}")
        ref_paths = row.get("ref_paths") or row.get("single_reference_images")
        if not isinstance(ref_paths, list) or len(ref_paths) != 1:
            raise ValueError(
                f"{SINGLE_REFERENCE_NO_TARGET_SCHEMA} requires exactly one ref_paths/single_reference_images entry "
                f"in {eval_set}:{row_idx}"
            )
        resolved_refs = [_resolve(root, path) for path in ref_paths]
        missing_refs = [path for path in resolved_refs if not Path(path).is_file()]
        if missing_refs:
            raise FileNotFoundError(f"Missing reference image(s) in {eval_set}:{row_idx}: {missing_refs}")
        row["multi_reference_images"] = resolved_refs
        row["num_refs"] = 1
    return rows


def _expected_ref_tokens_for_bank(bank) -> int:
    return int(bank.num_ref_images) * int(bank.tokens_per_image)


def _load_ref_latent_grids(pipe, ref_image_paths: list[str], *, image_size: int | None, batch_size: int, device, dtype):
    return ref_infer.load_native_vae_reference_latent_grids(
        pipe,
        ref_image_paths,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )


def _blend_initial_latents_with_first_reference(pipe, latent_inputs: dict, ref_grids: torch.Tensor, blend: float) -> dict:
    blend = float(blend)
    if blend <= 0.0:
        return latent_inputs
    if blend > 1.0:
        raise ValueError("--init_ref_latent_blend must be in [0, 1].")
    if ref_grids.ndim != 5 or ref_grids.shape[0] != 1 or ref_grids.shape[1] < 1:
        raise ValueError(f"Expected ref_grids [1,N,C,H,W] with N>=1, got {tuple(ref_grids.shape)}")

    ref_packed, _ref_ids = pack_transformer_latents_compat(pipe, ref_grids[:, 0])
    base_packed = latent_inputs["latents_packed"]
    if ref_packed.shape != base_packed.shape:
        raise ValueError(
            "Reference-init latent shape mismatch: "
            f"ref_packed={tuple(ref_packed.shape)} base={tuple(base_packed.shape)}. "
            "Use matching output size and ref_latent_image_size."
        )
    out = dict(latent_inputs)
    out["latents_packed"] = (1.0 - blend) * base_packed + blend * ref_packed.to(
        device=base_packed.device, dtype=base_packed.dtype
    )
    meta = dict(out.get("latent_meta") or {})
    meta["init_ref_latent_blend"] = blend
    out["latent_meta"] = meta
    return out


def _prepare_flowmatch_img2img_latents(
    pipe,
    latent_inputs: dict,
    ref_grids: torch.Tensor,
    *,
    edit_strength: float,
) -> dict:
    if float(edit_strength) == 0.0:
        return latent_inputs
    timesteps = torch.as_tensor(latent_inputs["timesteps"])
    num_steps = int(timesteps.numel())
    init_steps = max(1, min(int(num_steps * float(edit_strength)), num_steps))
    t_start = num_steps - init_steps
    edit_timesteps = timesteps[t_start:]

    clean_packed, _ref_ids = pack_transformer_latents_compat(pipe, ref_grids[:, 0])
    pure_noise = latent_inputs["latents_packed"]
    if clean_packed.shape != pure_noise.shape:
        raise ValueError(
            "FlowMatch img2img latent shape mismatch: "
            f"clean_reference={tuple(clean_packed.shape)} pure_noise={tuple(pure_noise.shape)}. "
            "Use matching --height/--width and reference latent image size."
        )
    scheduler = getattr(pipe, "scheduler", None)
    scheduler.set_begin_index(t_start)
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    edited_latents = scheduler.scale_noise(
        clean_packed.to(device=pure_noise.device, dtype=pure_noise.dtype),
        edit_timesteps[:1].to(device=pure_noise.device),
        pure_noise,
    )

    out = dict(latent_inputs)
    out["latents_packed"] = edited_latents
    out["timesteps"] = edit_timesteps
    meta = dict(out.get("latent_meta") or {})
    meta.update(
        edit_strength=float(edit_strength),
        edit_active=True,
        t_start=t_start,
        actual_edit_steps=int(edit_timesteps.numel()),
    )
    out["latent_meta"] = meta
    return out


def _validate_edit_request(
    *,
    edit_strength: float,
    manifest_schema: str,
    init_ref_latent_blend: float,
) -> float:
    edit_strength = float(edit_strength)
    if not (0.0 <= edit_strength <= 1.0):
        raise ValueError("--edit_strength must be in [0, 1].")
    if edit_strength > 0.0 and manifest_schema != SINGLE_REFERENCE_NO_TARGET_SCHEMA:
        raise ValueError(
            "--edit_strength > 0 requires --manifest_schema single_reference_no_target."
        )
    if edit_strength > 0.0 and float(init_ref_latent_blend) > 0.0:
        raise ValueError(
            "--edit_strength and --init_ref_latent_blend cannot both be greater than zero."
        )
    return edit_strength


def _packed_latent_sha256(latents_packed: torch.Tensor) -> str:
    if not torch.is_tensor(latents_packed):
        raise TypeError("latents_packed must be a torch.Tensor")
    frozen = latents_packed.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(frozen.dtype).encode("ascii"))
    digest.update(json.dumps(list(frozen.shape), separators=(",", ":")).encode("ascii"))
    digest.update(frozen.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _ensure_initial_noise_sha256(latent_inputs: dict) -> dict:
    existing = latent_inputs.get("initial_noise_sha256")
    if existing is not None:
        if not isinstance(existing, str) or len(existing) != 64:
            raise ValueError("initial_noise_sha256 must be a 64-character SHA256 hex digest")
        return latent_inputs
    latent_inputs["initial_noise_sha256"] = _packed_latent_sha256(
        latent_inputs["latents_packed"]
    )
    return latent_inputs


def planned_method_names() -> list[str]:
    return list(DEFAULT_METHOD_NAMES)


def supported_method_names() -> list[str]:
    return list(
        dict.fromkeys(
            [
                *DEFAULT_METHOD_NAMES,
                *LEGACY_V2_8_METHOD_NAMES,
                *QUERY_CONDITIONED_METHODS,
                *sorted(TEMPORAL_HIGH_TOKEN_METHODS),
                *HIGH_TOKEN_SEED_METHOD_SPECS,
                *V2_14_PHASE2_METHOD_SPECS,
                *V2_14_SAME_ROUTE_METHOD_SPECS,
            ]
        )
    )


def generation_seed_for_method(base_seed: int, method_name: str) -> int:
    spec = V2_14_SAME_ROUTE_METHOD_SPECS.get(str(method_name))
    if spec is None:
        spec = HIGH_TOKEN_SEED_METHOD_SPECS.get(str(method_name))
    if spec is None:
        return int(base_seed)
    return int(base_seed) + int(spec.get("seed_offset", 0))


def temporal_high_token_uses_full_bank(method_name: str, *, step: int, num_steps: int) -> bool:
    method_name = str(method_name)
    if method_name not in TEMPORAL_HIGH_TOKEN_METHODS:
        raise ValueError(f"Unsupported temporal high-token method: {method_name}")
    step = max(0, min(int(step), max(int(num_steps), 1) - 1))
    num_steps = max(int(num_steps), 1)
    if method_name == "v2_11_high_last":
        return step == num_steps - 1
    if method_name == "v2_11_high_late":
        return step >= max(num_steps // 2, 1)
    if method_name == "v2_11_high_early":
        return step < max(num_steps // 2, 1)
    if method_name == "v2_11_high_mid":
        if num_steps <= 2:
            return True
        return 0 < step < num_steps - 1
    raise ValueError(f"Unsupported temporal high-token method: {method_name}")


def _route_for_method(
    bank,
    method_name: str,
    *,
    step: int,
    num_steps: int,
    random_seed: int,
    query_tokens: torch.Tensor | None = None,
) -> RoutedReferenceTokens:
    method_name = str(method_name)
    if method_name == "v2_8_high_token_direct":
        return full_reference_tokens(bank)
    if method_name in HIGH_TOKEN_SEED_METHOD_SPECS:
        route = full_reference_tokens(bank)
        route.policy = method_name
        return route
    if method_name == "v2_8_duplicate512_cluster":
        return duplicate_coreset_to_full_tokens(bank)
    if method_name in TEMPORAL_HIGH_TOKEN_METHODS:
        if temporal_high_token_uses_full_bank(method_name, step=step, num_steps=num_steps):
            route = full_reference_tokens(bank)
            route.policy = method_name
            return route
        return select_anchor_detail_reference_tokens(
            bank,
            anchor_tokens_per_ref=64,
            tokens_per_image=bank.tokens_per_image,
            policy_name=f"{method_name}_coreset",
        )
    if method_name in ANCHOR_DETAIL_METHODS:
        anchors, details = ANCHOR_DETAIL_METHODS[method_name]
        tokens_per_image = int(anchors) + int(details)
        if bank.tokens_per_image != tokens_per_image:
            raise ValueError(
                f"{method_name} expects {tokens_per_image} tokens per reference image, "
                f"but config uses {bank.tokens_per_image}."
            )
        return select_anchor_detail_reference_tokens(
            bank,
            anchor_tokens_per_ref=int(anchors),
            tokens_per_image=tokens_per_image,
            policy_name=method_name,
        )
    if method_name in V2_14_SAME_ROUTE_METHOD_SPECS:
        spec = V2_14_SAME_ROUTE_METHOD_SPECS[method_name]
        anchors = int(spec["anchor"])
        details = int(spec["detail"])
        tokens_per_image = anchors + details
        if bank.tokens_per_image != tokens_per_image:
            raise ValueError(
                f"{method_name} expects {tokens_per_image} tokens per reference image, "
                f"but config uses {bank.tokens_per_image}."
            )
        return select_anchor_detail_reference_tokens(
            bank,
            anchor_tokens_per_ref=anchors,
            tokens_per_image=tokens_per_image,
            policy_name=method_name,
        )
    if method_name in V2_14_PHASE2_METHOD_SPECS:
        spec = V2_14_PHASE2_METHOD_SPECS[method_name]
        anchors = int(spec["anchor"])
        details = int(spec["detail"])
        tokens_per_image = anchors + details
        if bank.tokens_per_image != tokens_per_image:
            raise ValueError(
                f"{method_name} expects {tokens_per_image} tokens per reference image, "
                f"but config uses {bank.tokens_per_image}."
            )
        return select_anchor_detail_ablation_reference_tokens(
            bank,
            anchor_tokens_per_ref=anchors,
            tokens_per_image=tokens_per_image,
            anchor_policy=str(spec["anchor_policy"]),
            detail_policy=str(spec["detail_policy"]),
            mask_anchor_tokens=bool(spec["mask_anchor_tokens"]),
            quota_policy=str(spec["quota_policy"]),
            random_seed=random_seed,
            policy_name=method_name,
        )
    if method_name in QUERY_CONDITIONED_METHODS:
        spec = QUERY_CONDITIONED_METHODS[method_name]
        anchors = int(spec["anchor"])
        query_count = int(spec["query"])
        tokens_per_image = anchors + query_count
        if bank.tokens_per_image != tokens_per_image:
            raise ValueError(
                f"{method_name} expects {tokens_per_image} tokens per reference image, "
                f"but config uses {bank.tokens_per_image}."
            )
        use_query = True
        if spec["schedule"] == "late":
            use_query = int(step) >= max(int(num_steps) // 2, 1)
        if not use_query:
            return select_anchor_detail_reference_tokens(
                bank,
                anchor_tokens_per_ref=64,
                tokens_per_image=tokens_per_image,
                policy_name=f"{method_name}_early_coreset",
            )
        if query_tokens is None:
            raise ValueError(f"{method_name} requires query_tokens.")
        return select_query_conditioned_reference_tokens(
            bank,
            query_tokens=query_tokens,
            anchor_tokens_per_ref=anchors,
            tokens_per_image=tokens_per_image,
            policy_name=method_name,
        )
    if method_name in STATIC_POLICY_METHODS:
        return select_ctnr_reference_tokens(
            bank,
            policy=STATIC_POLICY_METHODS[method_name],
            random_seed=random_seed,
        )
    if method_name in set(schedule_names()):
        policy = policy_for_schedule_step(method_name, step=step, num_steps=num_steps)
        return select_ctnr_reference_tokens(bank, policy=policy, random_seed=random_seed)
    raise ValueError(f"Unsupported v2.8 CTNR method: {method_name}")


def _overlap_fraction(current: torch.Tensor, previous: torch.Tensor | None) -> float | None:
    if previous is None:
        return None
    current_set = set(int(value) for value in current.detach().cpu().tolist())
    previous_set = set(int(value) for value in previous.detach().cpu().tolist())
    if not current_set:
        return 0.0
    return len(current_set & previous_set) / float(len(current_set))


def reset_scheduler_for_new_trajectory(scheduler, *, begin_index: int = 0) -> None:
    if scheduler is None:
        return
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(int(begin_index))
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None


def _trace_entry(
    *,
    method_name: str,
    route: RoutedReferenceTokens,
    step: int,
    timestep,
    overlap: float | None,
    reference_conditioning_mode: str = DEFAULT_REFERENCE_CONDITIONING_MODE,
    condition_stats: dict | None = None,
    positive_interaction_stats: dict | None = None,
    negative_interaction_stats: dict | None = None,
) -> dict:
    raw_timestep = timestep.detach().float().mean() if torch.is_tensor(timestep) else torch.tensor(float(timestep))
    entry = {
        "step": int(step),
        "raw_timestep": float(raw_timestep.cpu()),
        "method": method_name,
        "policy": route.policy,
        "num_ref_tokens": int(route.num_ref_tokens),
        "per_ref_counts": list(route.per_ref_counts),
        "unique_source_count": int(route.unique_source_count),
        "overlap_with_previous": overlap,
        "churn_from_previous": None if overlap is None else float(1.0 - overlap),
        "reference_conditioning_mode": reference_conditioning_mode,
    }
    if method_name in ANCHOR_DETAIL_METHODS:
        anchor_tokens_per_ref, detail_tokens_per_ref = ANCHOR_DETAIL_METHODS[method_name]
        entry["route_split"] = {
            "anchor_tokens_per_ref": int(anchor_tokens_per_ref),
            "detail_tokens_per_ref": int(detail_tokens_per_ref),
        }
    if condition_stats:
        entry["condition_stats"] = condition_stats
    if positive_interaction_stats:
        entry["positive_interaction_stats"] = positive_interaction_stats
    if negative_interaction_stats:
        entry["negative_interaction_stats"] = negative_interaction_stats
    if method_name in ANCHOR_DETAIL_METHODS:
        # Formal traces must remain target-free even when diagnostic stats use legacy names.
        for stats_name in ("positive_interaction_stats", "negative_interaction_stats"):
            stats = entry.get(stats_name)
            if isinstance(stats, dict):
                entry[stats_name] = {
                    key.replace("target", "conditioned"): value for key, value in stats.items()
                }
    return entry


def _tensor_l2(value: torch.Tensor | None) -> float:
    if not torch.is_tensor(value) or value.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(value.detach().float()).cpu())


def _condition_stats(cond) -> dict:
    combined = cond.combined_prompt_embeds.detach().float()
    num_semantic = int(getattr(cond, "num_semantic_ref_tokens", 0) or 0)
    text = combined[:, num_semantic:] if num_semantic > 0 else combined
    semantic = combined[:, :num_semantic] if num_semantic > 0 else combined[:, :0]
    text_l2 = _tensor_l2(text)
    semantic_l2 = _tensor_l2(semantic)
    native_l2 = _tensor_l2(getattr(cond, "ref_latents", None))
    return {
        "num_semantic_ref_tokens": num_semantic,
        "num_native_ref_tokens": int(getattr(cond, "num_ref_tokens", 0) or 0),
        "semantic_ref_l2": semantic_l2,
        "native_ref_l2": native_l2,
        "text_l2": text_l2,
        "semantic_ref_l2_to_text_l2": semantic_l2 / max(text_l2, 1.0e-12),
        "native_ref_l2_to_text_l2": native_l2 / max(text_l2, 1.0e-12),
    }


def _run_case_method(
    *,
    pipe,
    cfg: dict,
    row: dict,
    method_name: str,
    bank,
    prompt_bundle,
    negative_prompt_bundle,
    latent_inputs: dict,
    latent_seed: int,
    output_root: Path,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_ref_mode: str,
    ref_latent_scale: float,
    config_path: str | Path | None = None,
    reference_conditioning_mode: str = DEFAULT_REFERENCE_CONDITIONING_MODE,
    checkpoint_metadata: dict | None = None,
    device,
    dtype,
) -> dict:
    latents_packed = latent_inputs["latents_packed"].clone()
    img_ids = latent_inputs["img_ids"]
    timesteps = torch.as_tensor(latent_inputs["timesteps"], device=device, dtype=torch.float32)
    adapter_cfg = cfg["reference_adapter"]
    previous_indices = None
    trace = []
    random_seed = int(latent_seed)
    reset_scheduler_for_new_trajectory(
        getattr(pipe, "scheduler", None),
        begin_index=int((latent_inputs.get("latent_meta") or {}).get("t_start", 0)),
    )

    for step, timestep in enumerate(timesteps):
        route = _route_for_method(
            bank,
            method_name,
            step=step,
            num_steps=int(timesteps.numel()),
            random_seed=random_seed,
            query_tokens=latents_packed,
        )
        overlap = _overlap_fraction(route.flat_indices[0], previous_indices)
        previous_indices = route.flat_indices[0].detach().clone()
        ref_latents = route.tokens.to(device=device, dtype=dtype)
        ref_latents = ref_latents * float(ref_latent_scale)
        route_ref_img_ids = route.ids.to(device=device)
        cond = build_native_vae_reference_condition_bundle(
            prompt_bundle=prompt_bundle,
            img_ids=img_ids,
            ref_latents=ref_latents,
            ref_img_ids=route_ref_img_ids,
            txt_id_strategy=adapter_cfg.get("txt_id_strategy", "zero_prefix"),
        )
        if route.num_ref_tokens != int(cond.num_ref_tokens):
            raise AssertionError("Route token count and ConditionBundle token count diverged.")
        variable_length_methods = {
            "v2_8_high_token_direct",
            "v2_8_duplicate512_cluster",
            *TEMPORAL_HIGH_TOKEN_METHODS,
            *HIGH_TOKEN_SEED_METHOD_SPECS,
        }
        expected_ref_tokens = _expected_ref_tokens_for_bank(bank)
        if method_name not in variable_length_methods and route.num_ref_tokens != expected_ref_tokens:
            raise AssertionError(
                f"{method_name} must provide exactly {expected_ref_tokens} reference tokens per step."
            )
        model_timestep = ref_infer.transformer_timestep_from_scheduler(pipe, timestep).view(1)
        with torch.no_grad():
            velocity = ref_infer.predict_velocity_compat(
                pipe,
                latents_packed,
                model_timestep,
                cond,
                reference_conditioning_mode=reference_conditioning_mode,
            )
            negative_interaction_stats = {}
            if float(guidance_scale) > 1.0:
                if negative_ref_mode == "same":
                    neg_ref_latents = ref_latents
                    neg_ref_img_ids = route.ids.to(device=device)
                elif negative_ref_mode == "drop":
                    neg_ref_latents = ref_latents[:, :0, :].to(device=device, dtype=dtype)
                    neg_ref_img_ids = route.ids[:, :0, :].to(device=device)
                else:
                    raise ValueError(f"Unsupported negative_ref_mode: {negative_ref_mode}")
                neg_cond = build_native_vae_reference_condition_bundle(
                    prompt_bundle=negative_prompt_bundle,
                    img_ids=img_ids,
                    ref_latents=neg_ref_latents,
                    ref_img_ids=neg_ref_img_ids,
                    txt_id_strategy=adapter_cfg.get("txt_id_strategy", "zero_prefix"),
                )
                neg_velocity = ref_infer.predict_velocity_compat(
                    pipe,
                    latents_packed,
                    model_timestep,
                    neg_cond,
                    reference_conditioning_mode=reference_conditioning_mode,
                )
                velocity = neg_velocity + float(guidance_scale) * (velocity - neg_velocity)
        latents_packed = ref_infer.scheduler_step_packed_compat(
            getattr(pipe, "scheduler", None),
            velocity,
            timestep,
            latents_packed,
            latent_meta=latent_inputs.get("latent_meta"),
        )
        trace.append(
            _trace_entry(
                method_name=method_name,
                route=route,
                step=step,
                timestep=timestep,
                overlap=overlap,
                reference_conditioning_mode=reference_conditioning_mode,
                condition_stats=_condition_stats(cond),
                positive_interaction_stats={},
                negative_interaction_stats=negative_interaction_stats,
            )
        )

    output = {
        "latents_packed": latents_packed,
        "gate_trace": trace,
        "images": ref_infer.decode_packed_latents_to_images(
            pipe,
            latents_packed,
            img_ids,
            latent_inputs["latent_meta"],
        ),
    }
    out_args = SimpleNamespace(
        config=str(
            Path(config_path)
            if config_path is not None
            else Path("configs/qm_ref_sop_small_valid_native_vae_native_coreset_k144_eval.yaml")
        ),
        checkpoint=None,
        mode="stepwise_gate",
        prompt=row["prompt"],
        ref_images=row["multi_reference_images"],
        out_dir=str(output_root / row["case_id"] / method_name),
        save_gate_trace=True,
        num_images=1,
        num_inference_steps=int(num_inference_steps),
        height=int(height),
        width=int(width),
        seed=int(latent_seed),
        guidance_scale=float(guidance_scale),
        negative_ref_mode=str(negative_ref_mode),
        ref_latent_scale=float(ref_latent_scale),
        reference_conditioning_mode=reference_conditioning_mode,
    )
    metadata_latent_inputs = dict(latent_inputs)
    latent_meta = dict(metadata_latent_inputs.get("latent_meta") or {})
    latent_meta.setdefault("edit_strength", 0.0)
    latent_meta.setdefault("edit_active", False)
    latent_meta.setdefault("t_start", 0)
    latent_meta.setdefault("actual_edit_steps", 0)
    metadata_latent_inputs["latent_meta"] = latent_meta
    return ref_infer.save_inference_outputs(
        output,
        out_args,
        cfg,
        checkpoint_metadata or {},
        metadata_latent_inputs,
    )


def run(args) -> dict:
    reference_conditioning_mode = validate_reference_conditioning_mode(
        getattr(args, "reference_conditioning_mode", DEFAULT_REFERENCE_CONDITIONING_MODE)
    )
    cfg = validate_config(ref_infer.load_config(args.config))
    if ref_infer.reference_source(cfg) != "native_vae":
        raise ValueError("v2.8 CTNR oracle requires reference_adapter.source='native_vae'.")
    checkpoint = getattr(args, "checkpoint", None)
    semantic_reference_enabled = bool(cfg.get("semantic_reference", {}).get("enabled", False))
    if checkpoint:
        raise ValueError("v2.8 CTNR oracle paper protocol does not support --checkpoint.")
    if semantic_reference_enabled:
        raise ValueError("v2.8 CTNR oracle paper protocol requires semantic_reference.enabled=false.")
    if bool(cfg.get("reference_adapter", {}).get("apply_native_adapter", False)):
        raise ValueError("v2.8 CTNR oracle paper protocol requires reference_adapter.apply_native_adapter=false.")

    init_ref_latent_blend = float(getattr(args, "init_ref_latent_blend", 0.0) or 0.0)
    if not (0.0 <= init_ref_latent_blend <= 1.0):
        raise ValueError("--init_ref_latent_blend must be in [0, 1].")
    edit_strength = _validate_edit_request(
        edit_strength=float(getattr(args, "edit_strength", 0.0) or 0.0),
        manifest_schema=str(args.manifest_schema),
        init_ref_latent_blend=init_ref_latent_blend,
    )

    methods = list(args.methods or planned_method_names())
    unsupported = [method for method in methods if method not in set(supported_method_names())]
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = ref_infer._dtype_from_cfg(cfg)
    root = Path(args.root)
    output_root = Path(args.output_root)
    rows = _eval_rows(args.eval_set, root, manifest_schema=args.manifest_schema)
    pipe = load_flux2_pipeline(cfg["model"]["inference_model_id"], dtype=dtype, device=device)
    freeze_base_model(pipe)
    checkpoint_metadata: dict = {}
    adapter_cfg = cfg["reference_adapter"]
    tokens_per_image = int(adapter_cfg.get("tokens_per_image", 144))
    anchor_tokens_per_ref = int(adapter_cfg.get("coreset_anchor_tokens", 64))
    guidance_scale = float(getattr(args, "guidance_scale", 1.0))
    if guidance_scale < 1.0:
        raise ValueError("--guidance_scale must be >= 1.0.")
    negative_ref_mode = str(getattr(args, "negative_ref_mode", "same"))
    if negative_ref_mode not in {"same", "drop"}:
        raise ValueError("--negative_ref_mode must be one of: same, drop")
    ref_latent_scale = float(getattr(args, "ref_latent_scale", 1.0))
    if ref_latent_scale <= 0.0:
        raise ValueError("--ref_latent_scale must be > 0.")
    outputs = []
    for case_idx, row in enumerate(rows):
        prompt_bundle = ref_infer.encode_prompt_compat(pipe, row["prompt"], device=device)
        negative_prompt_bundle = (
            ref_infer.encode_prompt_compat(pipe, "", device=device) if guidance_scale > 1.0 else None
        )
        base_seed = int(row.get("seed", 0))
        latent_args = SimpleNamespace(
            mode="stepwise_gate",
            seed=base_seed,
            num_images=1,
            height=int(args.height),
            width=int(args.width),
            num_inference_steps=int(args.num_inference_steps),
        )
        latent_inputs = ref_infer.prepare_inference_latents(pipe, cfg, latent_args, device=device, dtype=dtype)
        ref_grids = _load_ref_latent_grids(
            pipe,
            row["multi_reference_images"],
            image_size=adapter_cfg.get("ref_latent_image_size"),
            batch_size=1,
            device=device,
            dtype=dtype,
        )
        latent_inputs = _ensure_initial_noise_sha256(latent_inputs)
        if edit_strength > 0.0:
            latent_inputs = _prepare_flowmatch_img2img_latents(
                pipe,
                latent_inputs,
                ref_grids,
                edit_strength=edit_strength,
            )
        else:
            latent_inputs = _blend_initial_latents_with_first_reference(
                pipe, latent_inputs, ref_grids, init_ref_latent_blend
            )
        latent_inputs_by_seed = {base_seed: latent_inputs}
        bank = build_reference_token_bank(
            pipe,
            ref_grids,
            tokens_per_image=tokens_per_image,
            pool_grid=adapter_cfg.get("pool_grid"),
            anchor_tokens_per_ref=anchor_tokens_per_ref,
        )
        for method_name in methods:
            latent_seed = generation_seed_for_method(base_seed, method_name)
            if latent_seed not in latent_inputs_by_seed:
                method_latent_args = SimpleNamespace(
                    mode="stepwise_gate",
                    seed=latent_seed,
                    num_images=1,
                    height=int(args.height),
                    width=int(args.width),
                    num_inference_steps=int(args.num_inference_steps),
                )
                latent_inputs_by_seed[latent_seed] = ref_infer.prepare_inference_latents(
                    pipe,
                    cfg,
                    method_latent_args,
                    device=device,
                    dtype=dtype,
                )
                latent_inputs_by_seed[latent_seed] = _ensure_initial_noise_sha256(
                    latent_inputs_by_seed[latent_seed]
                )
                if edit_strength > 0.0:
                    latent_inputs_by_seed[latent_seed] = _prepare_flowmatch_img2img_latents(
                        pipe,
                        latent_inputs_by_seed[latent_seed],
                        ref_grids,
                        edit_strength=edit_strength,
                    )
                else:
                    latent_inputs_by_seed[latent_seed] = _blend_initial_latents_with_first_reference(
                        pipe, latent_inputs_by_seed[latent_seed], ref_grids, init_ref_latent_blend
                    )
            result = _run_case_method(
                pipe=pipe,
                cfg=cfg,
                row=row,
                method_name=method_name,
                bank=bank,
                prompt_bundle=prompt_bundle,
                negative_prompt_bundle=negative_prompt_bundle,
                latent_inputs=latent_inputs_by_seed[latent_seed],
                latent_seed=latent_seed,
                output_root=output_root,
                height=int(args.height),
                width=int(args.width),
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=guidance_scale,
                negative_ref_mode=negative_ref_mode,
                ref_latent_scale=ref_latent_scale,
                config_path=args.config,
                reference_conditioning_mode=reference_conditioning_mode,
                checkpoint_metadata=checkpoint_metadata,
                device=device,
                dtype=dtype,
            )
            outputs.append(
                {
                    "case_id": row["case_id"],
                    "method": method_name,
                    "metadata_path": result["metadata_path"],
                    "image_paths": result.get("image_paths", []),
                    "gate_trace_path": result.get("gate_trace_path"),
                }
            )
            print(f"[CTNR] processed case {case_idx + 1}/{len(rows)} {row['case_id']} method={method_name}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    edit_active = edit_strength > 0.0
    if edit_active:
        actual_edit_steps = max(
            1,
            min(int(int(args.num_inference_steps) * edit_strength), int(args.num_inference_steps)),
        )
        t_start = int(args.num_inference_steps) - actual_edit_steps
    else:
        actual_edit_steps = 0
        t_start = 0
    summary = {
        "backend": "v2_8_ctnr_oracle",
        "eval_set": str(Path(args.eval_set)),
        "manifest_schema": str(args.manifest_schema),
        "output_root": str(output_root),
        "config": str(Path(args.config)),
        "checkpoint": None,
        "checkpoint_loaded": False,
        "checkpoint_global_step": None,
        "checkpoint_hook_count": 0,
        "case_count": len(rows),
        "methods": methods,
        "height": int(args.height),
        "width": int(args.width),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": guidance_scale,
        "negative_ref_mode": negative_ref_mode,
        "ref_latent_scale": ref_latent_scale,
        "init_ref_latent_blend": init_ref_latent_blend,
        "edit_strength": edit_strength,
        "edit_active": edit_active,
        "t_start": t_start,
        "actual_edit_steps": actual_edit_steps,
        "reference_conditioning_mode": reference_conditioning_mode,
        "outputs": outputs,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run v2.8 CTNR raw-token static and dynamic oracle fixed eval.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--negative_ref_mode", choices=["same", "drop"], default="same")
    parser.add_argument(
        "--reference_conditioning_mode",
        "--reference-conditioning-mode",
        dest="reference_conditioning_mode",
        choices=REFERENCE_CONDITIONING_MODES,
        default=DEFAULT_REFERENCE_CONDITIONING_MODE,
    )
    parser.add_argument("--ref_latent_scale", type=float, default=1.0)
    parser.add_argument("--init_ref_latent_blend", type=float, default=0.0)
    parser.add_argument("--edit_strength", type=float, default=0.0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--allow_base_mismatch", action="store_true")
    parser.add_argument("--allow_gate_override", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="*", default=None, choices=supported_method_names())
    parser.add_argument("--manifest_schema", default=FIXED_REFERENCE_SCHEMA, choices=MANIFEST_SCHEMAS)
    return parser.parse_args(argv)


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
