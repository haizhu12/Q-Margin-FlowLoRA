import importlib
import json
from types import SimpleNamespace

from PIL import Image


def _module():
    return importlib.import_module("scripts.run_background_replace")


def test_run_generates_empty_background_then_hard_composites_source(tmp_path):
    mod = _module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  inference_model_id: local-model\n  torch_dtype: bf16\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "source.png"
    source = Image.new("RGB", (2, 2), "white")
    source.putpixel((0, 0), (0, 0, 0))
    source.putpixel((1, 0), (220, 220, 220))
    source.save(source_path)
    calls = {}

    class FakePipeline:
        def __call__(self, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (4, 4), (20, 40, 60))])

    def pipeline_loader(model_id, dtype_name, device):
        calls["load"] = (model_id, dtype_name, device)
        return FakePipeline()

    def generator_factory(seed):
        calls["seed"] = seed
        return "fake-generator"

    args = SimpleNamespace(
        config=str(config_path),
        source_image=str(source_path),
        output_dir=str(tmp_path / "out"),
        background_prompt="empty alpine forest trail, no bicycle",
        width=4,
        height=4,
        num_inference_steps=3,
        guidance_scale=1.0,
        seed=42,
        device="cpu",
        transparent_distance=6,
        opaque_distance=96,
        edge_feather_radius=1,
        core_neighbor_distance=64,
        bottom_start_y=None,
        bottom_end_y=None,
        bottom_transparent_distance=None,
        bottom_opaque_distance=None,
        bottom_core_neighbor_distance=None,
    )

    result = mod.run(
        args,
        pipeline_loader=pipeline_loader,
        generator_factory=generator_factory,
    )

    assert calls["load"] == ("local-model", "bf16", "cpu")
    assert calls["seed"] == 42
    assert calls["generate"] == {
        "prompt": "empty alpine forest trail, no bicycle",
        "height": 4,
        "width": 4,
        "num_inference_steps": 3,
        "guidance_scale": 1.0,
        "generator": "fake-generator",
    }
    output = Image.open(result["image_path"]).convert("RGB")
    assert output.getpixel((1, 1)) == (0, 0, 0)
    assert output.getpixel((2, 1)) not in ((220, 220, 220), (20, 40, 60))
    assert output.getpixel((1, 2)) == (20, 40, 60)
    saved_alpha = Image.open(result["foreground_mask_path"]).convert("L")
    assert 0 < saved_alpha.getpixel((1, 0)) < 255
    metadata = json.loads((tmp_path / "out" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pixel_preservation"]["foreground_pixel_identity_verified"] is True
    assert metadata["pixel_preservation"]["max_abs_foreground_error"] == 0
    assert metadata["pixel_preservation"]["soft_edge_pixel_count"] == 1
    assert metadata["mask_parameters"]["edge_feather_radius"] == 1
    assert metadata["mask_parameters"]["core_neighbor_distance"] == 64


def test_extract_pipeline_image_requires_exactly_one_pil_image():
    mod = _module()
    try:
        mod._extract_pipeline_image(SimpleNamespace(images=[]))
    except RuntimeError as exc:
        assert "exactly one PIL image" in str(exc)
    else:
        raise AssertionError("empty pipeline output must be rejected")
