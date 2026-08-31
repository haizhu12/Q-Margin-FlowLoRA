import importlib
import json

import numpy as np
import pytest
from PIL import Image


def _module():
    return importlib.import_module("qmargin.background_composite")


def test_hard_foreground_composite_preserves_locked_pixels_exactly():
    mod = _module()
    background = Image.new("RGB", (4, 4), (10, 20, 30))
    source = Image.fromarray(
        np.array(
            [
                [[200, 1, 2], [3, 210, 4]],
                [[5, 6, 220], [230, 7, 8]],
            ],
            dtype=np.uint8,
        ),
        mode="RGB",
    )
    mask = Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8), mode="L")

    result, audit = mod.composite_hard_foreground(
        background,
        source,
        mask,
        placement="center_no_resize",
    )

    actual = np.asarray(result)
    assert actual[1, 1].tolist() == [200, 1, 2]
    assert actual[2, 2].tolist() == [230, 7, 8]
    assert actual[1, 2].tolist() == [10, 20, 30]
    assert actual[2, 1].tolist() == [10, 20, 30]
    assert audit == {
        "placement": "center_no_resize",
        "offset_x": 1,
        "offset_y": 1,
        "preserved_pixel_count": 2,
        "preserved_fraction": 0.125,
        "foreground_pixel_identity_verified": True,
        "max_abs_foreground_error": 0,
    }


def test_hard_foreground_composite_rejects_non_binary_mask():
    mod = _module()
    with pytest.raises(ValueError, match="binary mask must contain only 0 and 255"):
        mod.composite_hard_foreground(
            Image.new("RGB", (2, 2)),
            Image.new("RGB", (2, 2)),
            Image.fromarray(np.array([[0, 128], [255, 0]], dtype=np.uint8), mode="L"),
        )


def test_hard_foreground_composite_rejects_source_larger_than_background():
    mod = _module()
    with pytest.raises(ValueError, match="source image must fit inside background without resizing"):
        mod.composite_hard_foreground(
            Image.new("RGB", (2, 2)),
            Image.new("RGB", (3, 2)),
            Image.new("L", (3, 2), 255),
        )


def test_hard_foreground_composite_rejects_mask_size_mismatch():
    mod = _module()
    with pytest.raises(ValueError, match="mask size must match source image size"):
        mod.composite_hard_foreground(
            Image.new("RGB", (4, 4)),
            Image.new("RGB", (2, 2)),
            Image.new("L", (1, 2), 255),
        )


def test_white_background_mask_keeps_subject_and_drops_near_white_pixels():
    mod = _module()
    source = Image.fromarray(
        np.array(
            [[[255, 255, 255], [248, 245, 244], [220, 220, 220], [255, 0, 0]]],
            dtype=np.uint8,
        ),
        mode="RGB",
    )

    mask = mod.build_white_background_binary_mask(source, foreground_distance=20)

    assert np.asarray(mask).tolist() == [[0, 0, 255, 255]]


def test_white_background_mask_raises_threshold_near_bottom_shadow_band():
    mod = _module()
    pixels = np.full((4, 2, 3), 225, dtype=np.uint8)
    pixels[:, 1] = 0
    source = Image.fromarray(pixels, mode="RGB")

    mask = mod.build_white_background_binary_mask(
        source,
        foreground_distance=20,
        bottom_start_y=2,
        bottom_foreground_distance=80,
    )

    assert np.asarray(mask).tolist() == [
        [255, 255],
        [255, 255],
        [0, 255],
        [0, 255],
    ]


def test_white_background_alpha_has_transparent_soft_and_pixel_locked_regions():
    mod = _module()
    source = Image.fromarray(
        np.array(
            [[[255, 255, 255], [244, 244, 244], [220, 220, 220], [200, 200, 200], [255, 0, 0]]],
            dtype=np.uint8,
        ),
        mode="RGB",
    )

    alpha = mod.build_white_background_alpha_mask(
        source,
        transparent_distance=16,
        opaque_distance=48,
        edge_feather_radius=1,
        core_neighbor_distance=32,
    )

    values = np.asarray(alpha)[0].tolist()
    assert values[0] == 0
    assert values[1] == 0
    assert 0 < values[2] < 255
    assert 0 < values[3] < 255
    assert values[4] == 255


def test_white_background_alpha_feathers_spatial_boundary_but_locks_interior():
    mod = _module()
    pixels = np.full((5, 5, 3), 255, dtype=np.uint8)
    pixels[1:4, 1:4] = 100
    source = Image.fromarray(pixels, mode="RGB")

    alpha = mod.build_white_background_alpha_mask(
        source,
        transparent_distance=16,
        opaque_distance=96,
        edge_feather_radius=1,
        core_neighbor_distance=64,
    )

    values = np.asarray(alpha)
    assert values[2, 2] == 255
    assert 0 < values[1, 1] < 255
    assert values[0, 0] == 0


def test_white_background_alpha_recovers_known_black_foreground_coverage():
    mod = _module()
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    source_values = np.rint((1.0 - expected) * 255.0).astype(np.uint8)
    source = Image.fromarray(
        np.repeat(source_values[None, :, None], 3, axis=2),
        mode="RGB",
    )

    alpha = mod.build_white_background_alpha_mask(
        source,
        transparent_distance=0,
        opaque_distance=96,
        edge_feather_radius=1,
        core_neighbor_distance=64,
    )

    actual = np.asarray(alpha, dtype=np.float32)[0] / 255.0
    assert np.max(np.abs(actual - expected)) <= (1.0 / 255.0)


def test_white_background_alpha_removes_pixels_below_bottom_cutoff():
    mod = _module()
    source = Image.new("RGB", (3, 5), (80, 80, 80))

    alpha = mod.build_white_background_alpha_mask(
        source,
        transparent_distance=6,
        opaque_distance=96,
        edge_feather_radius=1,
        core_neighbor_distance=64,
        bottom_start_y=2,
        bottom_end_y=4,
        bottom_transparent_distance=128,
        bottom_opaque_distance=180,
        bottom_core_neighbor_distance=96,
    )

    values = np.asarray(alpha)
    assert np.any(values[2] > 0)
    assert np.all(values[4] == 0)


def test_soft_composite_decontaminates_white_edge_but_keeps_core_exact():
    mod = _module()
    background = Image.new("RGB", (3, 1), (20, 100, 40))
    source = Image.fromarray(
        np.array([[[1, 2, 3], [200, 200, 200], [250, 250, 250]]], dtype=np.uint8),
        mode="RGB",
    )
    alpha = Image.fromarray(np.array([[255, 128, 0]], dtype=np.uint8), mode="L")

    result, audit = mod.composite_pixel_locked_foreground(background, source, alpha)

    pixels = np.asarray(result)[0]
    assert pixels[0].tolist() == [1, 2, 3]
    assert pixels[2].tolist() == [20, 100, 40]
    assert pixels[1].tolist() not in ([200, 200, 200], [20, 100, 40])
    assert audit["locked_pixel_count"] == 1
    assert audit["soft_edge_pixel_count"] == 1
    assert audit["foreground_pixel_identity_verified"] is True
    assert audit["max_abs_foreground_error"] == 0


def test_save_background_replacement_artifacts_reloads_and_verifies_locked_pixels(tmp_path):
    mod = _module()
    source_path = tmp_path / "source.png"
    source = Image.fromarray(
        np.array(
            [
                [[200, 1, 2], [3, 210, 4]],
                [[5, 6, 220], [230, 7, 8]],
            ],
            dtype=np.uint8,
        ),
        mode="RGB",
    )
    source.save(source_path)
    mask = Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8), mode="L")
    background = Image.new("RGB", (4, 4), (10, 20, 30))

    result = mod.save_background_replacement_artifacts(
        output_dir=tmp_path / "out",
        background=background,
        source=source,
        mask=mask,
        source_path=source_path,
        config_path="demo.yaml",
        model_id="local-model",
        background_prompt="empty forest trail",
        seed=42,
        num_inference_steps=30,
        mask_parameters={"foreground_distance": 20},
    )

    metadata = json.loads((tmp_path / "out" / "metadata.json").read_text(encoding="utf-8"))
    reopened = np.asarray(Image.open(result["image_path"]).convert("RGB"))
    assert reopened[1, 1].tolist() == [200, 1, 2]
    assert reopened[2, 2].tolist() == [230, 7, 8]
    assert metadata["mode"] == "pixel_locked_soft_edge_composite_v2"
    assert metadata["checkpoint"] is None
    assert metadata["pixel_preservation"]["foreground_pixel_identity_verified"] is True
    assert metadata["pixel_preservation"]["max_abs_foreground_error"] == 0
    assert metadata["pixel_preservation"]["preserved_pixel_count"] == 2
    assert metadata["mask_parameters"] == {"foreground_distance": 20}
    assert len(metadata["source_sha256"]) == 64
    assert len(metadata["output_sha256"]) == 64
    assert (tmp_path / "out" / "background.png").is_file()
    assert (tmp_path / "out" / "foreground_mask.png").is_file()
