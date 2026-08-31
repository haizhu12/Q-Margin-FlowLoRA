from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

REFERENCE_CONDITIONING_MODES = (
    "joint_append_current_timestep",
    "kv_extract_prepend_fixed_timestep",
)
DEFAULT_REFERENCE_CONDITIONING_MODE = "joint_append_current_timestep"


def validate_reference_conditioning_mode(mode: str | None) -> str:
    normalized = DEFAULT_REFERENCE_CONDITIONING_MODE if mode is None else str(mode)
    if normalized not in REFERENCE_CONDITIONING_MODES:
        choices = ", ".join(REFERENCE_CONDITIONING_MODES)
        raise ValueError(f"reference_conditioning_mode must be one of: {choices}; got {normalized!r}")
    return normalized


@dataclass
class PromptBundle:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None = None
    internal_text_ids: torch.Tensor | None = None
    raw_output: Any | None = None

    @property
    def extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.internal_text_ids is not None:
            extra["txt_ids"] = self.internal_text_ids
        if self.pooled_prompt_embeds is not None:
            extra["pooled_prompt_embeds"] = self.pooled_prompt_embeds
        if self.raw_output is not None:
            extra["raw"] = self.raw_output
        return extra


@dataclass
class ConditionBundle:
    prompt_embeds: torch.Tensor
    combined_prompt_embeds: torch.Tensor
    txt_ids: torch.Tensor | None
    img_ids: torch.Tensor | None
    guidance: torch.Tensor | None = None
    ref_tokens: torch.Tensor | None = None
    ref_latents: torch.Tensor | None = None
    ref_img_ids: torch.Tensor | None = None
    pooled_prompt_embeds: torch.Tensor | None = None
    num_ref_tokens: int = 0
    num_semantic_ref_tokens: int = 0
    ref_fixed_timestep: float = 0.0


def get_pipeline_class(model_id: str):
    lower = model_id.lower()
    if "klein" in lower and "kv" in lower:
        from diffusers import Flux2KleinKVPipeline

        return Flux2KleinKVPipeline
    if "klein" in lower:
        from diffusers import Flux2KleinPipeline

        return Flux2KleinPipeline
    from diffusers import Flux2Pipeline

    return Flux2Pipeline


def load_flux2_pipeline(model_id: str, dtype=torch.bfloat16, device="cuda", device_map=None, **kwargs):
    Pipe = get_pipeline_class(model_id)
    load_kwargs = dict(torch_dtype=dtype, **kwargs)
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    pipe = Pipe.from_pretrained(model_id, **load_kwargs)
    if device_map is None and device is not None:
        pipe.to(device)
    return pipe


def freeze_base_model(pipe) -> None:
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.requires_grad_(False)
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        pipe.transformer.requires_grad_(False)
    for name in ("text_encoder", "text_encoder_2"):
        module = getattr(pipe, name, None)
        if hasattr(module, "requires_grad_"):
            module.requires_grad_(False)


def normalize_timestep_from_schedule(timestep: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    ts = timesteps.detach().float()
    if ts.numel() < 2:
        raise RuntimeError("Need full scheduler timesteps to normalize timestep; scalar fallback is forbidden.")
    t_max = ts.max()
    t_min = ts.min()
    t = timestep.detach().float()
    return ((t - t_min) / (t_max - t_min).clamp_min(1e-8)).clamp(0.0, 1.0)


def _looks_like_text_ids(x: torch.Tensor | None) -> bool:
    if x is None or not torch.is_tensor(x):
        return False
    return x.ndim in {2, 3} and x.shape[-1] in {3, 4}


def encode_prompt_compat(pipe, prompt, device="cuda", max_sequence_length=None) -> PromptBundle:
    if not hasattr(pipe, "encode_prompt"):
        raise AttributeError("Current pipeline has no encode_prompt; update diffusers or adapt flux2_utils.py.")

    sig = inspect.signature(pipe.encode_prompt)
    kwargs: dict[str, Any] = {}
    if "prompt" in sig.parameters:
        kwargs["prompt"] = prompt
    elif len(sig.parameters) > 0:
        first = next(iter(sig.parameters))
        kwargs[first] = prompt
    else:
        raise RuntimeError("encode_prompt_compat cannot find a prompt parameter.")
    if "device" in sig.parameters:
        kwargs["device"] = device
    if "num_images_per_prompt" in sig.parameters:
        kwargs["num_images_per_prompt"] = 1
    if "max_sequence_length" in sig.parameters and max_sequence_length is not None:
        kwargs["max_sequence_length"] = max_sequence_length

    out = pipe.encode_prompt(**kwargs)

    if isinstance(out, dict):
        prompt_embeds = out.get("prompt_embeds")
        if prompt_embeds is None:
            raise RuntimeError("encode_prompt returned dict without prompt_embeds.")
        text_ids = out.get("txt_ids", out.get("text_ids"))
        pooled = out.get("pooled_prompt_embeds", out.get("pooled_projections"))
        return PromptBundle(
            prompt_embeds=prompt_embeds.to(device),
            pooled_prompt_embeds=None if pooled is None else pooled.to(device),
            internal_text_ids=None if text_ids is None else text_ids.to(device),
            raw_output=out,
        )

    if isinstance(out, tuple):
        if not out:
            raise RuntimeError("encode_prompt returned an empty tuple.")
        prompt_embeds = out[0]
        second = out[1] if len(out) > 1 else None
        third = out[2] if len(out) > 2 else None
        pooled = None
        text_ids = None
        if _looks_like_text_ids(second):
            text_ids = second
        else:
            pooled = second if torch.is_tensor(second) else None
            if _looks_like_text_ids(third):
                text_ids = third
        return PromptBundle(
            prompt_embeds=prompt_embeds.to(device),
            pooled_prompt_embeds=None if pooled is None else pooled.to(device),
            internal_text_ids=None if text_ids is None else text_ids.to(device),
            raw_output=out,
        )

    if torch.is_tensor(out):
        return PromptBundle(prompt_embeds=out.to(device), raw_output=out)

    raise RuntimeError(f"Unsupported encode_prompt output type: {type(out)}")


def extend_txt_ids_for_ref_tokens(
    text_ids: torch.Tensor | None,
    num_ref_tokens: int,
    device,
    dtype=None,
    strategy: str = "zero_prefix",
) -> torch.Tensor:
    if text_ids is None:
        raise RuntimeError(
            "Prompt text_ids are required after prepending reference tokens. "
            "Current FLUX.2 transformer applies RoPE to txt_ids; run shape probe first."
        )
    if strategy != "zero_prefix":
        raise ValueError(f"Unsupported txt_id extension strategy: {strategy}")

    ids = text_ids.to(device=device)
    if dtype is not None and torch.is_floating_point(ids):
        ids = ids.to(dtype=dtype)

    if num_ref_tokens <= 0:
        return ids

    if ids.ndim == 2:
        ref_ids = torch.zeros((num_ref_tokens, ids.shape[-1]), device=ids.device, dtype=ids.dtype)
        return torch.cat([ref_ids, ids], dim=0)

    if ids.ndim == 3:
        ref_ids = torch.zeros((ids.shape[0], num_ref_tokens, ids.shape[-1]), device=ids.device, dtype=ids.dtype)
        return torch.cat([ref_ids, ids], dim=1)

    raise ValueError(f"Unsupported text_ids shape: {tuple(ids.shape)}")


def assert_text_ids_match_condition(combined_prompt_embeds: torch.Tensor, txt_ids: torch.Tensor) -> None:
    cond_len = combined_prompt_embeds.shape[1]
    if txt_ids.ndim == 2:
        ids_len = txt_ids.shape[0]
    elif txt_ids.ndim == 3:
        ids_len = txt_ids.shape[1]
    else:
        raise ValueError(f"Unsupported txt_ids shape: {tuple(txt_ids.shape)}")
    if cond_len != ids_len:
        raise AssertionError(f"combined_prompt_embeds length {cond_len} != txt_ids length {ids_len}")


def _resolve_pool_grid(tokens_per_image: int, pool_grid: tuple[int, int] | list[int] | None = None) -> tuple[int, int]:
    tokens_per_image = int(tokens_per_image)
    if tokens_per_image < 1:
        raise ValueError("tokens_per_image must be >= 1")
    if pool_grid is None:
        width = int(tokens_per_image**0.5)
        if width * width < tokens_per_image:
            width += 1
        height = (tokens_per_image + width - 1) // width
        return height, width
    if len(pool_grid) != 2:
        raise ValueError("pool_grid must contain exactly two integers: [height, width].")
    grid_h, grid_w = int(pool_grid[0]), int(pool_grid[1])
    if grid_h < 1 or grid_w < 1:
        raise ValueError("pool_grid height and width must be >= 1.")
    if grid_h * grid_w != tokens_per_image:
        raise ValueError(
            f"tokens_per_image={tokens_per_image} does not match pool_grid area {grid_h}x{grid_w}={grid_h * grid_w}."
        )
    return grid_h, grid_w


def make_reference_latent_ids(
    batch_size: int,
    num_ref_images: int,
    tokens_per_image: int,
    device,
    dtype=None,
    t_scale: int = 10,
    pool_grid: tuple[int, int] | list[int] | None = None,
) -> torch.Tensor:
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if int(num_ref_images) < 1:
        raise ValueError("num_ref_images must be >= 1")
    if int(tokens_per_image) < 1:
        raise ValueError("tokens_per_image must be >= 1")

    _height, width = _resolve_pool_grid(int(tokens_per_image), pool_grid=pool_grid)
    rows = []
    for ref_idx in range(int(num_ref_images)):
        t_coord = int(t_scale) * (ref_idx + 1)
        for token_idx in range(int(tokens_per_image)):
            rows.append((t_coord, token_idx // width, token_idx % width, 0))
    ids = torch.tensor(rows, device=device, dtype=torch.float32 if dtype is None else dtype)
    return ids.unsqueeze(0).expand(int(batch_size), -1, -1).contiguous()


def build_native_vae_reference_condition_bundle(
    prompt_bundle: PromptBundle,
    img_ids: torch.Tensor,
    ref_latents: torch.Tensor,
    ref_img_ids: torch.Tensor,
    txt_id_strategy: str = "zero_prefix",
    guidance: torch.Tensor | None = None,
) -> ConditionBundle:
    if ref_latents.ndim != 3:
        raise ValueError(f"ref_latents must be [B,Lr,D], got {tuple(ref_latents.shape)}")
    if ref_img_ids.ndim != 3:
        raise ValueError(f"ref_img_ids must be [B,Lr,4], got {tuple(ref_img_ids.shape)}")
    if ref_latents.shape[:2] != ref_img_ids.shape[:2]:
        raise ValueError("ref_latents and ref_img_ids must share batch and sequence length.")

    combined = prompt_bundle.prompt_embeds
    txt_ids = extend_txt_ids_for_ref_tokens(
        prompt_bundle.internal_text_ids,
        num_ref_tokens=0,
        device=combined.device,
        dtype=None,
        strategy=txt_id_strategy,
    )
    assert_text_ids_match_condition(combined, txt_ids)
    return ConditionBundle(
        prompt_embeds=prompt_bundle.prompt_embeds,
        pooled_prompt_embeds=prompt_bundle.pooled_prompt_embeds,
        ref_tokens=ref_latents,
        ref_latents=ref_latents,
        ref_img_ids=ref_img_ids,
        combined_prompt_embeds=combined,
        txt_ids=txt_ids,
        img_ids=img_ids,
        guidance=guidance,
        num_ref_tokens=ref_latents.shape[1],
        ref_fixed_timestep=0.0,
    )


def encode_image_latents_compat(pipe, images: torch.Tensor, generator=None) -> torch.Tensor:
    """Return verified FLUX.2 Klein transformer latents in BCHW layout."""
    if not hasattr(pipe, "_encode_vae_image"):
        raise RuntimeError(
            "Current pipeline has no _encode_vae_image. "
            "Implement a verified FLUX.2 latent encoder instead of using generic VAE scaling."
        )
    with torch.no_grad():
        return pipe._encode_vae_image(images, generator=generator)


def pool_reference_latent_grids_2d(
    pipe,
    ref_latent_grids: torch.Tensor,
    tokens_per_image: int,
    pool_grid: tuple[int, int] | list[int] | None = None,
    img_id_device=None,
    img_id_dtype=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool per-reference native latent grids in 2D and return aligned tokens and image ids.

    Accepts native VAE latent grids shaped [B, N, C, H, W], where N is the
    reference image count. Each reference image is pooled independently to
    `pool_grid`, packed with the pipeline helper, and returned as
    [B, N * tokens_per_image, D] with matching [B, N * tokens_per_image, 4]
    reference image ids.
    """
    if ref_latent_grids.ndim != 5:
        raise ValueError(f"ref_latent_grids must be [B,N,C,H,W], got {tuple(ref_latent_grids.shape)}")
    grid_h, grid_w = _resolve_pool_grid(int(tokens_per_image), pool_grid=pool_grid)
    batch_size, num_ref_images = ref_latent_grids.shape[:2]
    if batch_size < 1 or num_ref_images < 1:
        raise ValueError("ref_latent_grids must contain at least one batch item and one reference image.")
    flat = ref_latent_grids.reshape(batch_size * num_ref_images, *ref_latent_grids.shape[2:])
    pooled_grid = F.adaptive_avg_pool2d(flat.float(), (grid_h, grid_w)).to(dtype=ref_latent_grids.dtype)
    packed, _packed_img_ids = pack_transformer_latents_compat(pipe, pooled_grid)
    if packed.shape[1] != int(tokens_per_image):
        raise AssertionError(
            f"Packed pooled reference token count {packed.shape[1]} != tokens_per_image {int(tokens_per_image)}"
        )
    pooled_tokens = packed.reshape(batch_size, num_ref_images * int(tokens_per_image), packed.shape[-1])
    ids_device = img_id_device if img_id_device is not None else ref_latent_grids.device
    pooled_img_ids = make_reference_latent_ids(
        batch_size=batch_size,
        num_ref_images=num_ref_images,
        tokens_per_image=int(tokens_per_image),
        device=ids_device,
        dtype=img_id_dtype,
        pool_grid=(grid_h, grid_w),
    )
    return pooled_tokens, pooled_img_ids


def _sort_indices_by_native_ids(native_ids: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    selected_ids = native_ids[indices]
    keys = (
        selected_ids[:, 1].float() * 1.0e6
        + selected_ids[:, 2].float() * 1.0e3
        + selected_ids[:, 3].float()
    )
    return indices[torch.argsort(keys)]


def _native_nearest_grid_indices(
    native_ids: torch.Tensor,
    tokens_per_image: int,
    pool_grid: tuple[int, int] | list[int] | None,
) -> torch.Tensor:
    tokens_per_image = int(tokens_per_image)
    if native_ids.ndim != 2 or native_ids.shape[-1] != 4:
        raise ValueError(f"native_ids must be [L,4], got {tuple(native_ids.shape)}")
    if native_ids.shape[0] < tokens_per_image:
        raise ValueError(
            f"Cannot select {tokens_per_image} native tokens from only {native_ids.shape[0]} available tokens."
        )
    grid_h, grid_w = _resolve_pool_grid(tokens_per_image, pool_grid=pool_grid)
    coords = native_ids[:, 1:3].float()
    target_y = torch.linspace(float(coords[:, 0].min()), float(coords[:, 0].max()), grid_h, device=coords.device)
    target_x = torch.linspace(float(coords[:, 1].min()), float(coords[:, 1].max()), grid_w, device=coords.device)
    selected: list[int] = []
    used: set[int] = set()
    for y in target_y:
        for x in target_x:
            distances = (coords[:, 0] - y).square() + (coords[:, 1] - x).square()
            for candidate in torch.argsort(distances).detach().cpu().tolist():
                candidate = int(candidate)
                if candidate not in used:
                    selected.append(candidate)
                    used.add(candidate)
                    break
            if len(selected) >= tokens_per_image:
                break
        if len(selected) >= tokens_per_image:
            break
    if len(selected) < tokens_per_image:
        keys = coords[:, 0] * 1.0e6 + coords[:, 1] * 1.0e3 + native_ids[:, 3].float()
        for candidate in torch.argsort(keys).detach().cpu().tolist():
            candidate = int(candidate)
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
                if len(selected) >= tokens_per_image:
                    break
    indices = torch.tensor(selected[:tokens_per_image], device=native_ids.device, dtype=torch.long)
    return _sort_indices_by_native_ids(native_ids, indices)


def _native_detail_scores(tokens: torch.Tensor, native_ids: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [L,D], got {tuple(tokens.shape)}")
    coords = native_ids[:, 1:3]
    unique_y = torch.unique(coords[:, 0]).sort().values
    unique_x = torch.unique(coords[:, 1]).sort().values
    if unique_y.numel() * unique_x.numel() != tokens.shape[0]:
        return torch.linalg.vector_norm(tokens.float() - tokens.float().mean(dim=0, keepdim=True), dim=-1)
    order = torch.argsort(coords[:, 0].float() * 1.0e6 + coords[:, 1].float())
    token_grid = tokens[order].float().transpose(0, 1).reshape(1, tokens.shape[1], unique_y.numel(), unique_x.numel())
    local_mean = F.avg_pool2d(token_grid, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    residual = torch.linalg.vector_norm((token_grid - local_mean).reshape(tokens.shape[1], -1).transpose(0, 1), dim=-1)
    scores = torch.empty_like(residual)
    scores[order] = residual
    return scores


def _native_coreset_indices(
    tokens: torch.Tensor,
    native_ids: torch.Tensor,
    tokens_per_image: int,
    pool_grid: tuple[int, int] | list[int] | None,
    coreset_anchor_tokens: int,
) -> torch.Tensor:
    tokens_per_image = int(tokens_per_image)
    if tokens.shape[0] < tokens_per_image:
        raise ValueError(
            f"Cannot select {tokens_per_image} native tokens from only {tokens.shape[0]} available tokens."
        )
    anchor_count = max(0, min(int(coreset_anchor_tokens), tokens_per_image))
    if anchor_count > 0:
        anchor_indices = _native_nearest_grid_indices(tokens.new_zeros(native_ids.shape).copy_(native_ids), anchor_count, None)
    else:
        anchor_indices = torch.empty(0, device=tokens.device, dtype=torch.long)
    detail_count = tokens_per_image - int(anchor_indices.numel())
    if detail_count <= 0:
        return _sort_indices_by_native_ids(native_ids, anchor_indices[:tokens_per_image])

    scores = _native_detail_scores(tokens, native_ids)
    if anchor_indices.numel() > 0:
        scores = scores.clone()
        scores[anchor_indices] = -torch.inf
    detail_indices = torch.topk(scores, k=detail_count, largest=True).indices
    combined = torch.cat([anchor_indices, detail_indices], dim=0)
    return _sort_indices_by_native_ids(native_ids, combined[:tokens_per_image])


def _native_random_indices(native_ids: torch.Tensor, tokens_per_image: int, seed: int) -> torch.Tensor:
    tokens_per_image = int(tokens_per_image)
    if native_ids.shape[0] < tokens_per_image:
        raise ValueError(
            f"Cannot select {tokens_per_image} native tokens from only {native_ids.shape[0]} available tokens."
        )
    generator = torch.Generator(device=native_ids.device)
    generator.manual_seed(int(seed))
    indices = torch.randperm(native_ids.shape[0], device=native_ids.device, generator=generator)[:tokens_per_image]
    return _sort_indices_by_native_ids(native_ids, indices)


def select_reference_native_latent_tokens_2d(
    pipe,
    ref_latent_grids: torch.Tensor,
    tokens_per_image: int,
    mode: str = "native_nearest_grid_2d",
    pool_grid: tuple[int, int] | list[int] | None = None,
    img_id_device=None,
    img_id_dtype=None,
    coreset_anchor_tokens: int = 64,
    random_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select original packed native reference tokens without averaging.

    Accepts native VAE latent grids shaped [B,N,C,H,W]. Each reference image is
    packed with the pipeline helper, then a fixed number of original packed
    tokens is selected per reference image. Token values and native y/x ids are
    preserved; the first id coordinate is reassigned to the existing reference
    convention 10, 20, ... so multi-reference ordering stays explicit.
    """
    if ref_latent_grids.ndim != 5:
        raise ValueError(f"ref_latent_grids must be [B,N,C,H,W], got {tuple(ref_latent_grids.shape)}")
    tokens_per_image = int(tokens_per_image)
    if tokens_per_image < 1:
        raise ValueError("tokens_per_image must be >= 1")
    mode = str(mode).lower()
    aliases = {
        "native_nearest": "native_nearest_grid_2d",
        "native_nearest_grid": "native_nearest_grid_2d",
        "native_nearest_grid_2d": "native_nearest_grid_2d",
        "native_coreset": "native_coreset",
        "native_random": "native_random_coverage",
        "native_random_coverage": "native_random_coverage",
    }
    if mode not in aliases:
        raise ValueError(
            "native token selection mode must be one of: "
            "native_nearest_grid_2d, native_coreset, native_random_coverage."
        )
    mode = aliases[mode]
    batch_size, num_ref_images = ref_latent_grids.shape[:2]
    if batch_size < 1 or num_ref_images < 1:
        raise ValueError("ref_latent_grids must contain at least one batch item and one reference image.")

    flat = ref_latent_grids.reshape(batch_size * num_ref_images, *ref_latent_grids.shape[2:])
    packed, packed_ids = pack_transformer_latents_compat(pipe, flat)
    packed = packed.reshape(batch_size, num_ref_images, packed.shape[1], packed.shape[2])
    packed_ids = packed_ids.reshape(batch_size, num_ref_images, packed_ids.shape[1], packed_ids.shape[2]).to(
        device=packed.device
    )
    if packed.shape[2] < tokens_per_image:
        raise ValueError(
            f"Cannot select {tokens_per_image} native tokens from only {packed.shape[2]} packed tokens per reference."
        )

    out_tokens = []
    out_ids = []
    for batch_idx in range(batch_size):
        batch_tokens = []
        batch_ids = []
        for ref_idx in range(num_ref_images):
            tokens = packed[batch_idx, ref_idx]
            native_ids = packed_ids[batch_idx, ref_idx].clone()
            native_ids[:, 0] = float(10 * (ref_idx + 1))
            if mode == "native_nearest_grid_2d":
                indices = _native_nearest_grid_indices(native_ids, tokens_per_image, pool_grid)
            elif mode == "native_coreset":
                indices = _native_coreset_indices(
                    tokens,
                    native_ids,
                    tokens_per_image,
                    pool_grid,
                    coreset_anchor_tokens=int(coreset_anchor_tokens),
                )
            else:
                indices = _native_random_indices(
                    native_ids,
                    tokens_per_image,
                    seed=int(random_seed) + batch_idx * 1009 + ref_idx * 9176,
                )
            batch_tokens.append(tokens[indices])
            batch_ids.append(native_ids[indices])
        out_tokens.append(torch.cat(batch_tokens, dim=0))
        out_ids.append(torch.cat(batch_ids, dim=0))
    selected_tokens = torch.stack(out_tokens, dim=0)
    selected_ids = torch.stack(out_ids, dim=0)
    ids_device = img_id_device if img_id_device is not None else ref_latent_grids.device
    ids_dtype = torch.float32 if img_id_dtype is None else img_id_dtype
    return selected_tokens, selected_ids.to(device=ids_device, dtype=ids_dtype)


def pack_transformer_latents_compat(pipe, transformer_latent_bchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack already prepared FLUX.2 transformer latents and create 4D latent ids."""
    if transformer_latent_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW transformer latent, got {tuple(transformer_latent_bchw.shape)}")
    if not hasattr(pipe, "_pack_latents") or not hasattr(pipe, "_prepare_latent_ids"):
        raise RuntimeError("Current pipeline is missing FLUX.2 latent packing helpers.")

    packed = pipe._pack_latents(transformer_latent_bchw)
    img_ids = pipe._prepare_latent_ids(transformer_latent_bchw).to(device=transformer_latent_bchw.device)
    if img_ids.shape[-1] != 4:
        raise AssertionError(f"FLUX.2 latent ids must have 4 coordinates (T,H,W,L), got {tuple(img_ids.shape)}")
    if img_ids.shape[-2] != packed.shape[1]:
        raise AssertionError(f"img_ids length {img_ids.shape[-2]} != packed latent tokens {packed.shape[1]}")
    return packed, img_ids


def _filter_kwargs_by_signature(fn, kwargs: dict) -> dict:
    params = inspect.signature(fn).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    internal_kwargs = set(getattr(fn, "__qmargin_internal_kwargs__", set()) or set())
    return {key: value for key, value in kwargs.items() if key in params or key in internal_kwargs}


def _require_if_supported(sig_params, name: str, value) -> None:
    if name in sig_params and value is None:
        raise RuntimeError(f"transformer.forward supports/requires `{name}`, but ConditionBundle has None.")


def _forward_supports_keyword(forward, sig_params, name: str) -> bool:
    if name in sig_params:
        return True
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig_params.values()):
        return True
    internal_kwargs = set(getattr(forward, "__qmargin_internal_kwargs__", set()) or set())
    return name in internal_kwargs


def _validate_img_ids_for_reference_concat(name: str, ids, *, batch_size: int, token_count: int) -> None:
    if ids is None:
        raise RuntimeError(f"{name} are required when native reference latents are concatenated.")
    if not torch.is_tensor(ids):
        raise ValueError(f"{name} must be a tensor when native reference latents are concatenated.")
    if ids.ndim != 3:
        raise ValueError(f"{name} must be [batch, tokens, id_dim], got {tuple(ids.shape)}.")
    if ids.shape[0] != batch_size:
        raise ValueError(f"{name} batch {ids.shape[0]} must match packed_latents batch {batch_size}.")
    if ids.shape[1] != token_count:
        raise ValueError(f"{name} sequence {ids.shape[1]} must match token count {token_count}.")


def _validate_native_reference_concat_contract(
    condition: ConditionBundle,
    packed_latents: torch.Tensor,
    reference_conditioning_mode: str,
) -> bool:
    num_ref_tokens = int(condition.num_ref_tokens)
    if num_ref_tokens < 0:
        raise ValueError("ConditionBundle.num_ref_tokens must be >= 0.")
    ref_latents = condition.ref_latents
    if ref_latents is None:
        if reference_conditioning_mode == "kv_extract_prepend_fixed_timestep" and num_ref_tokens > 0:
            raise RuntimeError(
                "reference_conditioning_mode='kv_extract_prepend_fixed_timestep' with num_ref_tokens > 0 "
                "requires native ref_latents; semantic-only reference tokens cannot use KV extraction."
            )
        return False
    if not torch.is_tensor(ref_latents):
        raise ValueError("ConditionBundle.ref_latents must be a tensor when provided.")
    if ref_latents.ndim != 3:
        raise ValueError(f"ref_latents must be [batch, tokens, hidden], got {tuple(ref_latents.shape)}.")
    if ref_latents.shape[1] != num_ref_tokens:
        raise ValueError(
            f"ConditionBundle.num_ref_tokens {num_ref_tokens} must match ref_latents sequence {ref_latents.shape[1]}."
        )
    if num_ref_tokens == 0:
        return False
    if ref_latents.shape[0] != packed_latents.shape[0]:
        raise ValueError("ref_latents batch must match packed_latents batch.")
    if ref_latents.shape[-1] != packed_latents.shape[-1]:
        raise ValueError("ref_latents hidden dimension must match packed_latents hidden dimension.")

    _validate_img_ids_for_reference_concat(
        "generated img_ids",
        condition.img_ids,
        batch_size=packed_latents.shape[0],
        token_count=packed_latents.shape[1],
    )
    _validate_img_ids_for_reference_concat(
        "ref_img_ids",
        condition.ref_img_ids,
        batch_size=packed_latents.shape[0],
        token_count=num_ref_tokens,
    )
    if condition.img_ids.shape[-1] != condition.ref_img_ids.shape[-1]:
        raise ValueError(
            f"generated img_ids id_dim {condition.img_ids.shape[-1]} must match ref_img_ids id_dim "
            f"{condition.ref_img_ids.shape[-1]}."
        )
    return True


def predict_velocity_compat(
    pipe,
    packed_latents,
    timestep,
    condition: ConditionBundle,
    reference_conditioning_mode: str = DEFAULT_REFERENCE_CONDITIONING_MODE,
    **extra_kwargs,
):
    """Call FLUX.2 transformer.forward with an explicit condition bundle."""
    reference_conditioning_mode = validate_reference_conditioning_mode(reference_conditioning_mode)

    transformer = pipe.transformer
    forward = transformer.forward
    sig_params = inspect.signature(forward).parameters
    base_latent_len = packed_latents.shape[1]

    if torch.is_tensor(condition.combined_prompt_embeds):
        model_dtype = packed_latents.dtype
        try:
            first_param = next(transformer.parameters())
            model_dtype = first_param.dtype
        except Exception:
            pass
        condition = ConditionBundle(
            prompt_embeds=condition.prompt_embeds,
            combined_prompt_embeds=condition.combined_prompt_embeds.to(device=packed_latents.device, dtype=model_dtype),
            txt_ids=None if condition.txt_ids is None else condition.txt_ids.to(device=packed_latents.device),
            img_ids=None if condition.img_ids is None else condition.img_ids.to(device=packed_latents.device),
            guidance=None if condition.guidance is None else condition.guidance.to(device=packed_latents.device),
            ref_tokens=condition.ref_tokens,
            ref_latents=None
            if condition.ref_latents is None
            else condition.ref_latents.to(device=packed_latents.device, dtype=model_dtype),
            ref_img_ids=None
            if condition.ref_img_ids is None
            else condition.ref_img_ids.to(device=packed_latents.device),
            pooled_prompt_embeds=None
            if condition.pooled_prompt_embeds is None
            else condition.pooled_prompt_embeds.to(device=packed_latents.device, dtype=model_dtype),
            num_ref_tokens=condition.num_ref_tokens,
            num_semantic_ref_tokens=condition.num_semantic_ref_tokens,
            ref_fixed_timestep=condition.ref_fixed_timestep,
        )
        packed_latents = packed_latents.to(dtype=model_dtype)

    transformer_hidden_states = packed_latents
    transformer_img_ids = condition.img_ids
    has_reference_tokens = _validate_native_reference_concat_contract(
        condition,
        packed_latents,
        reference_conditioning_mode,
    )
    if has_reference_tokens:
        if reference_conditioning_mode == "joint_append_current_timestep":
            transformer_hidden_states = torch.cat([packed_latents, condition.ref_latents], dim=1)
            transformer_img_ids = torch.cat([condition.img_ids, condition.ref_img_ids], dim=1)
        else:
            transformer_hidden_states = torch.cat([condition.ref_latents, packed_latents], dim=1)
            transformer_img_ids = torch.cat([condition.ref_img_ids, condition.img_ids], dim=1)

    kwargs = {
        "hidden_states": transformer_hidden_states,
        "encoder_hidden_states": condition.combined_prompt_embeds,
        "timestep": timestep.to(device=packed_latents.device) if torch.is_tensor(timestep) else timestep,
        "img_ids": transformer_img_ids,
        "txt_ids": condition.txt_ids,
        "guidance": condition.guidance,
        "return_dict": False,
        "num_ref_tokens": condition.num_ref_tokens,
        "num_semantic_ref_tokens": condition.num_semantic_ref_tokens,
        "ref_fixed_timestep": condition.ref_fixed_timestep,
    }
    # This contract owns cache mode; callers cannot smuggle a conflicting mode through extra kwargs.
    extra_kwargs = dict(extra_kwargs)
    extra_kwargs.pop("kv_cache_mode", None)
    kwargs.update(extra_kwargs)

    if has_reference_tokens and reference_conditioning_mode == "kv_extract_prepend_fixed_timestep":
        if not _forward_supports_keyword(forward, sig_params, "kv_cache_mode"):
            raise RuntimeError(
                "reference_conditioning_mode='kv_extract_prepend_fixed_timestep' requires "
                "transformer.forward to support `kv_cache_mode`; refusing to silently filter it."
            )
        kwargs["kv_cache_mode"] = "extract"

    if "pooled_projections" in sig_params and condition.pooled_prompt_embeds is not None:
        kwargs["pooled_projections"] = condition.pooled_prompt_embeds

    _require_if_supported(sig_params, "img_ids", condition.img_ids)
    _require_if_supported(sig_params, "txt_ids", condition.txt_ids)

    kwargs = _filter_kwargs_by_signature(forward, kwargs)
    if callable(transformer):
        out = transformer(**kwargs)
    else:
        out = forward(**kwargs)

    if isinstance(out, tuple):
        pred = out[0]
    elif hasattr(out, "sample"):
        pred = out.sample
    else:
        pred = out

    if has_reference_tokens and pred.shape[1] == transformer_hidden_states.shape[1]:
        if reference_conditioning_mode == "joint_append_current_timestep":
            pred = pred[:, :base_latent_len]
        else:
            pred = pred[:, -base_latent_len:]

    if pred.shape != packed_latents.shape:
        raise AssertionError(f"pred velocity shape {tuple(pred.shape)} != packed latent shape {tuple(packed_latents.shape)}")
    return pred
