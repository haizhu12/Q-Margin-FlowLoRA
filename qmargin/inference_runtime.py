"""Dependency-light helpers for frozen FLUX.2 inference runners.

This module intentionally contains no reference training, adapter, teacher, or
checkpoint imports.  It is shared by paper-protocol runners that use native VAE
reference latents directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from PIL import Image

from qmargin.flux2_utils import (
    DEFAULT_REFERENCE_CONDITIONING_MODE,
    encode_image_latents_compat,
    encode_prompt_compat,
    normalize_timestep_from_schedule,
    predict_velocity_compat,
    validate_reference_conditioning_mode,
)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _dtype_from_cfg(cfg: dict) -> torch.dtype:
    name = str(cfg.get("model", {}).get("torch_dtype", "bf16")).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("model.torch_dtype must be one of: bf16, fp16, fp32")


def reference_source(cfg: dict) -> str:
    return str(cfg.get("reference_adapter", {}).get("source", "learned_projector"))


def load_native_vae_reference_latent_grids(
    pipe,
    ref_image_paths: list[str | Path],
    *,
    image_size: int | None,
    batch_size: int,
    device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode reference images through the pipeline's native VAE pathway."""
    condition_images = []
    multiple_of = int(getattr(pipe, "vae_scale_factor", 8)) * 2
    target_image_size = None if image_size is None else int(image_size)
    if target_image_size is not None and target_image_size <= 0:
        raise ValueError("reference image_size must be > 0 when provided.")
    for path in ref_image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            pipe.image_processor.check_image_input(image)
            if target_image_size is not None:
                image_width = target_image_size
                image_height = target_image_size
            else:
                image_width, image_height = image.size
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            if image_width <= 0 or image_height <= 0:
                raise ValueError(
                    f"Reference image size is too small after rounding to VAE multiple: {image_width}x{image_height}"
                )
            condition_images.append(
                pipe.image_processor.preprocess(
                    image,
                    height=image_height,
                    width=image_width,
                    resize_mode="crop",
                )
            )
    if not condition_images:
        raise ValueError("At least one reference image is required.")
    vae_dtype = getattr(getattr(pipe, "vae", None), "dtype", dtype)
    image_tensor = torch.cat(condition_images, dim=0).to(device=device, dtype=vae_dtype)
    ref_latent_bchw = encode_image_latents_compat(pipe, image_tensor)
    return ref_latent_bchw.unsqueeze(0).expand(int(batch_size), -1, -1, -1, -1).contiguous()


def scheduler_step_packed_compat(scheduler, velocity, timestep, latents_packed, latent_meta=None):
    if scheduler is None:
        return latents_packed - velocity
    if hasattr(scheduler, "step_packed"):
        out = scheduler.step_packed(velocity, timestep, latents_packed, latent_meta=latent_meta)
    elif hasattr(scheduler, "step"):
        try:
            out = scheduler.step(velocity, timestep, latents_packed, return_dict=False)
        except TypeError:
            out = scheduler.step(velocity, timestep, latents_packed)
    else:
        raise RuntimeError("Scheduler has neither step_packed nor step.")
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "prev_sample"):
        return out.prev_sample
    if torch.is_tensor(out):
        return out
    raise RuntimeError(f"Unsupported scheduler step output type: {type(out)!r}")


def transformer_timestep_from_scheduler(pipe, scheduler_timestep: torch.Tensor) -> torch.Tensor:
    scheduler_config = getattr(getattr(pipe, "scheduler", None), "config", None)
    num_train_timesteps = float(getattr(scheduler_config, "num_train_timesteps", 1000))
    return scheduler_timestep / num_train_timesteps


def decode_packed_latents_to_images(pipe, latents_packed, img_ids, latent_meta: dict, output_type: str = "pil"):
    if not hasattr(pipe, "_unpack_latents_with_ids") or not hasattr(pipe, "_unpatchify_latents"):
        raise RuntimeError("Current FLUX.2 pipeline has no packed-latent decode helpers.")
    if not hasattr(pipe, "vae") or pipe.vae is None or not hasattr(pipe, "image_processor"):
        raise RuntimeError("Current FLUX.2 pipeline has no VAE/image processor for decoding latents.")
    height = int(latent_meta.get("height") or getattr(pipe, "default_sample_size", 64) * getattr(pipe, "vae_scale_factor", 8))
    width = int(latent_meta.get("width") or getattr(pipe, "default_sample_size", 64) * getattr(pipe, "vae_scale_factor", 8))
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8))
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))
    latents = pipe._unpack_latents_with_ids(latents_packed, img_ids, latent_height // 2, latent_width // 2)
    if isinstance(latents, (list, tuple)):
        latents = torch.stack(list(latents), dim=0)
    bn = getattr(pipe.vae, "bn", None)
    if bn is not None:
        eps = float(getattr(getattr(pipe.vae, "config", None), "batch_norm_eps", 1.0e-5))
        mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + eps).to(latents.device, latents.dtype)
        latents = latents * std + mean
    decoded = pipe.vae.decode(pipe._unpatchify_latents(latents), return_dict=False)[0].detach()
    return pipe.image_processor.postprocess(decoded, output_type=output_type)


def prepare_inference_timesteps(pipe, latents_packed: torch.Tensor, num_inference_steps: int, device) -> torch.Tensor:
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("Stepwise inference requires a scheduler on the FLUX.2 pipeline.")
    try:
        import numpy as np
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

        sigmas = np.linspace(1.0, 1 / int(num_inference_steps), int(num_inference_steps))
        if bool(getattr(getattr(scheduler, "config", None), "use_flow_sigmas", False)):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=latents_packed.shape[1], num_steps=int(num_inference_steps))
        timesteps, _ = retrieve_timesteps(scheduler, int(num_inference_steps), device, sigmas=sigmas, mu=mu)
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(0)
        return timesteps
    except Exception as exc:
        try:
            scheduler.set_timesteps(int(num_inference_steps), device=device)
            return scheduler.timesteps
        except Exception as fallback_exc:
            raise RuntimeError("Could not prepare scheduler timesteps for stepwise inference.") from fallback_exc or exc


def prepare_inference_latents(pipe, cfg: dict, args, device, dtype: torch.dtype) -> dict:
    if not hasattr(pipe, "prepare_latents"):
        raise RuntimeError("Current FLUX.2 pipeline has no prepare_latents helper.")
    height = int(args.height or getattr(pipe, "default_sample_size", 64) * getattr(pipe, "vae_scale_factor", 8))
    width = int(args.width or getattr(pipe, "default_sample_size", 64) * getattr(pipe, "vae_scale_factor", 8))
    generator = None if args.seed is None else torch.Generator(device="cpu").manual_seed(int(args.seed))
    in_channels = getattr(getattr(getattr(pipe, "transformer", None), "config", None), "in_channels", None)
    if in_channels is None:
        raise RuntimeError("Could not infer transformer.config.in_channels for latent preparation.")
    latents_packed, img_ids = pipe.prepare_latents(
        batch_size=int(args.num_images),
        num_latents_channels=int(in_channels) // 4,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    timesteps = prepare_inference_timesteps(pipe, latents_packed, int(args.num_inference_steps), device)
    return {"latents_packed": latents_packed, "img_ids": img_ids, "timesteps": timesteps, "latent_meta": {"height": height, "width": width}}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_inference_outputs(output: dict, args, cfg: dict, checkpoint_metadata: dict, latent_inputs: dict) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    if "latents_packed" in output:
        path = out_dir / "latents_packed.pt"
        torch.save(output["latents_packed"].detach().cpu(), path)
        result["latents_packed_path"] = str(path)
    if "velocity" in output:
        path = out_dir / "velocity.pt"
        torch.save(output["velocity"].detach().cpu(), path)
        result["velocity_path"] = str(path)
    if "images" in output:
        image_paths = []
        for idx, image in enumerate(output["images"]):
            image_path = out_dir / f"{idx:06d}.png"
            image.save(image_path)
            image_paths.append(str(image_path))
        result["image_paths"] = image_paths
    mode = validate_reference_conditioning_mode(getattr(args, "reference_conditioning_mode", DEFAULT_REFERENCE_CONDITIONING_MODE))
    raw_trace = output.get("gate_trace", [])
    executed_modes = {
        entry["reference_conditioning_mode"]
        for entry in raw_trace
        if isinstance(entry, dict) and entry.get("reference_conditioning_mode") is not None
    }
    if executed_modes and executed_modes != {mode}:
        raise RuntimeError(
            "reference_conditioning_mode does not match executed gate trace: "
            f"requested {mode!r}, trace recorded {sorted(executed_modes)!r}."
        )
    trace = [{**entry, "reference_conditioning_mode": mode} for entry in raw_trace]
    if args.save_gate_trace or cfg.get("adapter_injection", {}).get("log_gate_trace", False):
        trace_path = out_dir / "gate_trace.json"
        trace_path.write_text(json.dumps(_json_safe(trace), indent=2, ensure_ascii=False), encoding="utf-8")
        result["gate_trace_path"] = str(trace_path)
    checkpoint = getattr(args, "checkpoint", None)
    metadata = {
        "mode": args.mode,
        "prompt": args.prompt,
        "checkpoint": str(Path(checkpoint)) if checkpoint else None,
        "config": str(Path(args.config)),
        "reference_images": [str(Path(path)) for path in args.ref_images],
        "num_images": int(args.num_images),
        "num_inference_steps": int(args.num_inference_steps),
        "seed": args.seed,
        "initial_noise_sha256": latent_inputs.get("initial_noise_sha256"),
        "reference_conditioning_mode": mode,
        "latent_meta": latent_inputs.get("latent_meta", {}),
        "checkpoint_metadata": checkpoint_metadata,
        "outputs": result,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    result["metadata_path"] = str(metadata_path)
    result["metadata"] = metadata
    return result
