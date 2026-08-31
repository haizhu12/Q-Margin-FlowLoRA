import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.evaluate_fixed_reference_metrics import (
    MethodSpec,
    _as_feature_tensor,
    evaluate_fixed_reference_metrics,
    parse_method_spec,
)


def _save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def _write_eval_set(path: Path, root: Path) -> list[dict]:
    rows = [
        {
            "case_id": "case_a",
            "subject_id": "subject_a",
            "category": "toy",
            "prompt": "A studio product photo of the same toy.",
            "target_image": str(root / "data" / "case_a" / "target.png"),
            "single_reference_images": [str(root / "data" / "case_a" / "ref0.png")],
            "multi_reference_images": [
                str(root / "data" / "case_a" / "ref0.png"),
                str(root / "data" / "case_a" / "ref1.png"),
            ],
        },
        {
            "case_id": "case_b",
            "subject_id": "subject_b",
            "category": "toy",
            "prompt": "A studio product photo of the same product.",
            "target_image": str(root / "data" / "case_b" / "target.png"),
            "single_reference_images": [str(root / "data" / "case_b" / "ref0.png")],
            "multi_reference_images": [
                str(root / "data" / "case_b" / "ref0.png"),
                str(root / "data" / "case_b" / "ref1.png"),
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return rows


def _build_tiny_outputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    eval_set = tmp_path / "eval.jsonl"
    rows = _write_eval_set(eval_set, tmp_path)
    colors = {
        "case_a": {"target": (10, 20, 30), "ref0": (20, 20, 30), "ref1": (40, 20, 30)},
        "case_b": {"target": (100, 90, 80), "ref0": (110, 90, 80), "ref1": (130, 90, 80)},
    }
    for row in rows:
        case_colors = colors[row["case_id"]]
        _save_image(Path(row["target_image"]), case_colors["target"])
        _save_image(Path(row["single_reference_images"][0]), case_colors["ref0"])
        _save_image(Path(row["multi_reference_images"][1]), case_colors["ref1"])

    output_root = tmp_path / "outputs"
    for row in rows:
        case_id = row["case_id"]
        _save_image(output_root / case_id / "same_target" / "000000.png", colors[case_id]["target"])
        _save_image(output_root / case_id / "copy_ref" / "000000.png", colors[case_id]["ref0"])
    return eval_set, output_root, tmp_path / "metrics"


def test_parse_method_spec_requires_three_parts():
    spec = parse_method_spec(["native_text", "outputs/native", "native_text_only"])

    assert spec == MethodSpec(name="native_text", root=Path("outputs/native"), method_dir="native_text_only")

    with pytest.raises(ValueError, match="requires exactly three values"):
        parse_method_spec(["bad", "only_two"])


def test_evaluate_fixed_reference_metrics_writes_case_and_summary_outputs(tmp_path):
    eval_set, output_root, output_dir = _build_tiny_outputs(tmp_path)
    methods = [
        MethodSpec(name="same_target", root=output_root, method_dir="same_target"),
        MethodSpec(name="copy_ref", root=output_root, method_dir="copy_ref"),
    ]

    result = evaluate_fixed_reference_metrics(
        eval_set_path=eval_set,
        methods=methods,
        output_dir=output_dir,
        feature_backend="rgb_stats",
        prompt_backend="none",
        device="cpu",
    )

    assert result["case_count"] == 2
    assert result["method_count"] == 2
    assert (output_dir / "case_metrics.csv").is_file()
    assert (output_dir / "method_summary.csv").is_file()
    assert (output_dir / "summary.json").is_file()

    case_rows = list(csv.DictReader((output_dir / "case_metrics.csv").open(newline="", encoding="utf-8")))
    same_target = [row for row in case_rows if row["method"] == "same_target"]
    copy_ref = [row for row in case_rows if row["method"] == "copy_ref"]

    assert len(case_rows) == 4
    assert all(float(row["pixel_mae_to_target"]) == pytest.approx(0.0) for row in same_target)
    assert all(float(row["ref_copy_pixel_mae_min"]) == pytest.approx(0.0) for row in copy_ref)
    assert all(float(row["ssim_to_target"]) == pytest.approx(1.0) for row in same_target)
    assert "dino_sim_to_multi_ref_mean" in case_rows[0]
    assert "clip_text_image_sim" in case_rows[0]

    summary_rows = list(csv.DictReader((output_dir / "method_summary.csv").open(newline="", encoding="utf-8")))
    target_summary = next(row for row in summary_rows if row["method"] == "same_target")
    copy_summary = next(row for row in summary_rows if row["method"] == "copy_ref")
    assert float(target_summary["pixel_mae_to_target_mean"]) == pytest.approx(0.0)
    assert float(copy_summary["ref_copy_pixel_mae_min_mean"]) == pytest.approx(0.0)


def test_evaluate_fixed_reference_metrics_fails_on_missing_method_image(tmp_path):
    eval_set, output_root, output_dir = _build_tiny_outputs(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing_method"):
        evaluate_fixed_reference_metrics(
            eval_set_path=eval_set,
            methods=[MethodSpec(name="missing", root=output_root, method_dir="missing_method")],
            output_dir=output_dir,
            feature_backend="none",
            prompt_backend="none",
            device="cpu",
        )


def test_as_feature_tensor_accepts_pooler_output_objects():
    import torch

    class Output:
        def __init__(self):
            self.pooler_output = torch.tensor([[1.0, 2.0, 3.0]])

    tensor = _as_feature_tensor(Output())

    assert tensor.shape == (1, 3)
    assert tensor.tolist() == [[1.0, 2.0, 3.0]]
