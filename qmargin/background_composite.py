"""Pixel-locked foreground compositing for background-only image edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def build_white_background_binary_mask(
    source: Image.Image,
    *,
    foreground_distance: int = 20,
    bottom_start_y: int | None = None,
    bottom_foreground_distance: int | None = None,
) -> Image.Image:
    """Return a binary mask for a dark/colored subject on a near-white backdrop.

    Distance is measured as ``255 - min(R, G, B)``.  A higher threshold can be
    applied to the lower image band to reject warm studio-floor shadows while
    retaining dark tires and other foreground parts.
    """

    foreground_distance = int(foreground_distance)
    if not 0 <= foreground_distance <= 255:
        raise ValueError("foreground_distance must be in [0, 255]")
    rgb = np.asarray(source.convert("RGB"), dtype=np.int16)
    height = int(rgb.shape[0])
    thresholds = np.full((height, 1), foreground_distance, dtype=np.int16)
    if bottom_start_y is not None:
        bottom_start_y = int(bottom_start_y)
        if not 0 <= bottom_start_y <= height:
            raise ValueError("bottom_start_y must be within the source image")
        if bottom_foreground_distance is None:
            raise ValueError("bottom_foreground_distance is required with bottom_start_y")
        bottom_foreground_distance = int(bottom_foreground_distance)
        if not 0 <= bottom_foreground_distance <= 255:
            raise ValueError("bottom_foreground_distance must be in [0, 255]")
        thresholds[bottom_start_y:, 0] = bottom_foreground_distance
    white_distance = 255 - rgb.min(axis=2)
    mask = np.where(white_distance >= thresholds, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def build_white_background_alpha_mask(
    source: Image.Image,
    *,
    transparent_distance: int = 6,
    opaque_distance: int = 96,
    edge_feather_radius: int = 1,
    core_neighbor_distance: int = 64,
    bottom_start_y: int | None = None,
    bottom_end_y: int | None = None,
    bottom_transparent_distance: int | None = None,
    bottom_opaque_distance: int | None = None,
    bottom_core_neighbor_distance: int | None = None,
) -> Image.Image:
    """Build a white-background matte with a spatially feathered boundary.

    A pixel is locked only when both it and its full local neighborhood are
    sufficiently far from white. Boundary pixels use a physical white-screen
    alpha estimate so pale JPEG antialiasing is not copied as a bright outline.
    An optional lower-band transition rejects a studio-floor shadow.
    """

    transparent_distance = int(transparent_distance)
    opaque_distance = int(opaque_distance)
    if not 0 <= transparent_distance < opaque_distance <= 255:
        raise ValueError("Require 0 <= transparent_distance < opaque_distance <= 255")
    edge_feather_radius = int(edge_feather_radius)
    if edge_feather_radius < 0:
        raise ValueError("edge_feather_radius must be non-negative")
    core_neighbor_distance = int(core_neighbor_distance)
    if not 0 <= core_neighbor_distance <= 255:
        raise ValueError("core_neighbor_distance must be in [0, 255]")
    rgb = np.asarray(source.convert("RGB"), dtype=np.int16)
    height = int(rgb.shape[0])
    transparent = np.full((height, 1), transparent_distance, dtype=np.float32)
    opaque = np.full((height, 1), opaque_distance, dtype=np.float32)
    neighbor_threshold = np.full(
        (height, 1), core_neighbor_distance, dtype=np.float32
    )
    active_rows = np.ones((height, 1), dtype=bool)
    if bottom_start_y is not None:
        bottom_start_y = int(bottom_start_y)
        if bottom_end_y is None:
            raise ValueError("bottom_end_y is required with bottom_start_y")
        bottom_end_y = int(bottom_end_y)
        if not 0 <= bottom_start_y < bottom_end_y <= height:
            raise ValueError("Require 0 <= bottom_start_y < bottom_end_y <= image height")
        if (
            bottom_transparent_distance is None
            or bottom_opaque_distance is None
            or bottom_core_neighbor_distance is None
        ):
            raise ValueError(
                "bottom transparent, opaque, and neighbor distances are required "
                "with bottom_start_y"
            )
        bottom_transparent_distance = int(bottom_transparent_distance)
        bottom_opaque_distance = int(bottom_opaque_distance)
        bottom_core_neighbor_distance = int(bottom_core_neighbor_distance)
        if not 0 <= bottom_transparent_distance <= 255:
            raise ValueError("bottom_transparent_distance must be in [0, 255]")
        if not 0 <= bottom_opaque_distance <= 255:
            raise ValueError("bottom_opaque_distance must be in [0, 255]")
        if not 0 <= bottom_core_neighbor_distance <= 255:
            raise ValueError(
                "bottom_core_neighbor_distance must be in [0, 255]"
            )
        row_indices = np.arange(height, dtype=np.float32)[:, None]
        progress = np.clip(
            (row_indices - float(bottom_start_y))
            / float(bottom_end_y - bottom_start_y),
            0.0,
            1.0,
        )
        transitioned = transparent_distance + progress * (
            bottom_transparent_distance - transparent_distance
        )
        transparent = np.where(row_indices >= bottom_start_y, transitioned, transparent)
        opaque[bottom_start_y:, 0] = bottom_opaque_distance
        neighbor_threshold[bottom_start_y:, 0] = bottom_core_neighbor_distance
        active_rows[bottom_end_y:, 0] = False
    elif any(
        value is not None
        for value in (
            bottom_end_y,
            bottom_transparent_distance,
            bottom_opaque_distance,
            bottom_core_neighbor_distance,
        )
    ):
        raise ValueError("bottom_start_y is required with bottom-band parameters")
    white_distance = (255 - rgb.min(axis=2)).astype(np.float32)
    weak_foreground = (white_distance > transparent) & active_rows
    if edge_feather_radius:
        radius = edge_feather_radius
        padded = np.pad(white_distance, radius, mode="constant", constant_values=0.0)
        neighborhood_min = np.full_like(white_distance, 255.0)
        kernel_width = 2 * radius + 1
        for dy in range(kernel_width):
            for dx in range(kernel_width):
                neighborhood_min = np.minimum(
                    neighborhood_min,
                    padded[
                        dy : dy + white_distance.shape[0],
                        dx : dx + white_distance.shape[1],
                    ],
                )
    else:
        neighborhood_min = white_distance
    locked = (
        (white_distance >= opaque)
        & (neighborhood_min >= neighbor_threshold)
        & active_rows
    )
    edge_alpha = np.clip(
        (white_distance - transparent) / (255.0 - transparent),
        0.0,
        1.0,
    )
    alpha = np.rint(edge_alpha * 255.0).astype(np.uint8)
    alpha[~weak_foreground] = 0
    alpha[locked] = 255
    return Image.fromarray(alpha, mode="L")


def composite_hard_foreground(
    background: Image.Image,
    source: Image.Image,
    mask: Image.Image,
    *,
    placement: str = "center_no_resize",
) -> tuple[Image.Image, dict]:
    """Copy masked source RGB pixels exactly onto a generated background."""

    if placement != "center_no_resize":
        raise ValueError("placement must be 'center_no_resize'")
    source_rgb = source.convert("RGB")
    background_rgb = background.convert("RGB")
    mask_l = mask.convert("L")
    if mask_l.size != source_rgb.size:
        raise ValueError("mask size must match source image size")
    mask_values = np.unique(np.asarray(mask_l, dtype=np.uint8))
    if not set(mask_values.tolist()).issubset({0, 255}):
        raise ValueError("binary mask must contain only 0 and 255")
    source_width, source_height = source_rgb.size
    background_width, background_height = background_rgb.size
    if source_width > background_width or source_height > background_height:
        raise ValueError("source image must fit inside background without resizing")

    offset_x = (background_width - source_width) // 2
    offset_y = (background_height - source_height) // 2
    source_array = np.asarray(source_rgb, dtype=np.uint8)
    mask_array = np.asarray(mask_l, dtype=np.uint8) == 255
    result_array = np.asarray(background_rgb, dtype=np.uint8).copy()
    result_region = result_array[
        offset_y : offset_y + source_height,
        offset_x : offset_x + source_width,
    ]
    result_region[mask_array] = source_array[mask_array]

    preserved_count = int(mask_array.sum())
    if preserved_count:
        error = np.abs(
            result_region[mask_array].astype(np.int16)
            - source_array[mask_array].astype(np.int16)
        )
        max_error = int(error.max())
    else:
        max_error = 0
    audit = {
        "placement": placement,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "preserved_pixel_count": preserved_count,
        "preserved_fraction": preserved_count / float(background_width * background_height),
        "foreground_pixel_identity_verified": max_error == 0,
        "max_abs_foreground_error": max_error,
    }
    return Image.fromarray(result_array, mode="RGB"), audit


def composite_pixel_locked_foreground(
    background: Image.Image,
    source: Image.Image,
    alpha_mask: Image.Image,
    *,
    placement: str = "center_no_resize",
) -> tuple[Image.Image, dict]:
    """Composite a decontaminated soft edge while copying alpha=255 pixels exactly."""

    if placement != "center_no_resize":
        raise ValueError("placement must be 'center_no_resize'")
    source_rgb = source.convert("RGB")
    background_rgb = background.convert("RGB")
    alpha_l = alpha_mask.convert("L")
    if alpha_l.size != source_rgb.size:
        raise ValueError("mask size must match source image size")
    source_width, source_height = source_rgb.size
    background_width, background_height = background_rgb.size
    if source_width > background_width or source_height > background_height:
        raise ValueError("source image must fit inside background without resizing")

    offset_x = (background_width - source_width) // 2
    offset_y = (background_height - source_height) // 2
    source_array = np.asarray(source_rgb, dtype=np.uint8)
    alpha_u8 = np.asarray(alpha_l, dtype=np.uint8)
    alpha = alpha_u8.astype(np.float32) / 255.0
    locked = alpha_u8 == 255
    soft = (alpha_u8 > 0) & (alpha_u8 < 255)
    active = alpha_u8 > 0
    result_array = np.asarray(background_rgb, dtype=np.uint8).copy()
    result_region = result_array[
        offset_y : offset_y + source_height,
        offset_x : offset_x + source_width,
    ]
    background_region = result_region.astype(np.float32)
    decontaminated_mix = source_array.astype(np.float32) + (
        1.0 - alpha[..., None]
    ) * (background_region - 255.0)
    mixed_u8 = np.rint(np.clip(decontaminated_mix, 0.0, 255.0)).astype(np.uint8)
    result_region[active] = mixed_u8[active]
    result_region[locked] = source_array[locked]

    locked_count = int(locked.sum())
    if locked_count:
        error = np.abs(
            result_region[locked].astype(np.int16) - source_array[locked].astype(np.int16)
        )
        max_error = int(error.max())
    else:
        max_error = 0
    audit = {
        "placement": placement,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "locked_pixel_count": locked_count,
        "soft_edge_pixel_count": int(soft.sum()),
        "preserved_pixel_count": locked_count,
        "preserved_fraction": locked_count / float(background_width * background_height),
        "foreground_pixel_identity_verified": max_error == 0,
        "max_abs_foreground_error": max_error,
    }
    return Image.fromarray(result_array, mode="RGB"), audit


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_background_replacement_artifacts(
    *,
    output_dir: str | Path,
    background: Image.Image,
    source: Image.Image,
    mask: Image.Image,
    source_path: str | Path,
    config_path: str | Path,
    model_id: str,
    background_prompt: str,
    seed: int,
    num_inference_steps: int,
    mask_parameters: dict,
) -> dict:
    """Save a background-only edit and a pixel-identity audit bundle."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image does not exist: {source_path}")

    background_path = output_dir / "background.png"
    mask_path = output_dir / "foreground_mask.png"
    image_path = output_dir / "000000.png"
    metadata_path = output_dir / "metadata.json"
    background.convert("RGB").save(background_path)
    mask.convert("L").save(mask_path)
    result, audit = composite_pixel_locked_foreground(background, source, mask)
    result.save(image_path)

    with Image.open(image_path) as reopened_image:
        reopened = np.asarray(reopened_image.convert("RGB"), dtype=np.uint8)
    source_array = np.asarray(source.convert("RGB"), dtype=np.uint8)
    locked = np.asarray(mask.convert("L"), dtype=np.uint8) == 255
    offset_x = int(audit["offset_x"])
    offset_y = int(audit["offset_y"])
    source_height, source_width = source_array.shape[:2]
    reopened_region = reopened[
        offset_y : offset_y + source_height,
        offset_x : offset_x + source_width,
    ]
    if locked.any():
        persisted_error = np.abs(
            reopened_region[locked].astype(np.int16) - source_array[locked].astype(np.int16)
        )
        persisted_max_error = int(persisted_error.max())
    else:
        persisted_max_error = 0
    audit["foreground_pixel_identity_verified"] = persisted_max_error == 0
    audit["max_abs_foreground_error"] = persisted_max_error

    metadata = {
        "mode": "pixel_locked_soft_edge_composite_v2",
        "config": str(Path(config_path)),
        "model_id": str(model_id),
        "checkpoint": None,
        "checkpoint_loaded": False,
        "background_prompt": str(background_prompt),
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "width": int(background.width),
        "height": int(background.height),
        "source_image": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "mask_parameters": dict(mask_parameters),
        "pixel_preservation": audit,
        "outputs": {
            "background_path": str(background_path),
            "foreground_mask_path": str(mask_path),
            "image_path": str(image_path),
        },
        "output_sha256": _sha256_file(image_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "background_path": str(background_path),
        "foreground_mask_path": str(mask_path),
        "image_path": str(image_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }
