from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qmargin.background_composite import (
    build_white_background_alpha_mask,
    save_background_replacement_artifacts,
)


def _default_pipeline_loader(model_id: str, dtype_name: str, device: str):
    import torch

    from qmargin.flux2_utils import freeze_base_model, load_flux2_pipeline

    normalized = str(dtype_name).lower()
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in dtype_map:
        raise ValueError("model.torch_dtype must be one of: bf16, fp16, fp32")
    pipe = load_flux2_pipeline(model_id, dtype=dtype_map[normalized], device=device)
    freeze_base_model(pipe)
    return pipe


def _default_generator_factory(seed: int):
    import torch

    return torch.Generator(device="cpu").manual_seed(int(seed))


def _extract_pipeline_image(output) -> Image.Image:
    images = getattr(output, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1 or not isinstance(images[0], Image.Image):
        raise RuntimeError("Background pipeline must return exactly one PIL image.")
    return images[0].convert("RGB")


def run(args, *, pipeline_loader=None, generator_factory=None) -> dict:
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = cfg.get("model") or {}
    model_id = str(model_cfg.get("inference_model_id") or "")
    if not model_id:
        raise ValueError("Config must define model.inference_model_id")
    dtype_name = str(model_cfg.get("torch_dtype", "bf16"))
    source_path = Path(args.source_image)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image does not exist: {source_path}")
    width = int(args.width)
    height = int(args.height)
    if width <= 0 or height <= 0:
        raise ValueError("--width and --height must be positive")

    pipeline_loader = pipeline_loader or _default_pipeline_loader
    generator_factory = generator_factory or _default_generator_factory
    pipe = pipeline_loader(model_id, dtype_name, str(args.device))
    generator = generator_factory(int(args.seed))
    pipeline_output = pipe(
        prompt=str(args.background_prompt),
        height=height,
        width=width,
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        generator=generator,
    )
    background = _extract_pipeline_image(pipeline_output)
    if background.size != (width, height):
        raise RuntimeError(
            f"Background pipeline returned {background.size}, expected {(width, height)}."
        )
    with Image.open(source_path) as source_image:
        source = source_image.convert("RGB").copy()
    mask_parameters = {
        "transparent_distance": int(args.transparent_distance),
        "opaque_distance": int(args.opaque_distance),
        "edge_feather_radius": int(args.edge_feather_radius),
        "core_neighbor_distance": int(args.core_neighbor_distance),
        "bottom_start_y": None if args.bottom_start_y is None else int(args.bottom_start_y),
        "bottom_end_y": None if args.bottom_end_y is None else int(args.bottom_end_y),
        "bottom_transparent_distance": (
            None
            if args.bottom_transparent_distance is None
            else int(args.bottom_transparent_distance)
        ),
        "bottom_opaque_distance": (
            None if args.bottom_opaque_distance is None else int(args.bottom_opaque_distance)
        ),
        "bottom_core_neighbor_distance": (
            None
            if args.bottom_core_neighbor_distance is None
            else int(args.bottom_core_neighbor_distance)
        ),
    }
    mask = build_white_background_alpha_mask(source, **mask_parameters)
    result = save_background_replacement_artifacts(
        output_dir=args.output_dir,
        background=background,
        source=source,
        mask=mask,
        source_path=source_path,
        config_path=config_path,
        model_id=model_id,
        background_prompt=str(args.background_prompt),
        seed=int(args.seed),
        num_inference_steps=int(args.num_inference_steps),
        mask_parameters=mask_parameters,
    )
    print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate an empty background, lock the foreground core, and decontaminate its white-screen edge."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--background_prompt", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transparent_distance", type=int, default=6)
    parser.add_argument("--opaque_distance", type=int, default=96)
    parser.add_argument("--edge_feather_radius", type=int, default=1)
    parser.add_argument("--core_neighbor_distance", type=int, default=64)
    parser.add_argument("--bottom_start_y", type=int, default=None)
    parser.add_argument("--bottom_end_y", type=int, default=None)
    parser.add_argument("--bottom_transparent_distance", type=int, default=None)
    parser.add_argument("--bottom_opaque_distance", type=int, default=None)
    parser.add_argument("--bottom_core_neighbor_distance", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
