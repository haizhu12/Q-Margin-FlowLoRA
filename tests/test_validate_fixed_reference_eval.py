import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.validate_fixed_reference_eval import validate_fixed_reference_eval


def write_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(path)
    return str(path)


def write_eval_set(path: Path, row: dict) -> Path:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def make_row(tmp_path: Path) -> dict:
    return {
        "case_id": "case_001",
        "subject_id": "s1",
        "category": "toy",
        "prompt": "A studio product photo.",
        "seed": 123,
        "target_image": write_image(tmp_path / "target.png"),
        "single_reference_images": [write_image(tmp_path / "ref1.png")],
        "multi_reference_images": [
            write_image(tmp_path / "ref1.png"),
            write_image(tmp_path / "ref2.png"),
        ],
    }


def test_validate_fixed_reference_eval_accepts_valid_jsonl(tmp_path):
    path = write_eval_set(tmp_path / "eval.jsonl", make_row(tmp_path))

    summary = validate_fixed_reference_eval(path)

    assert summary["case_count"] == 1
    assert summary["subject_count"] == 1
    assert summary["case_ids"] == ["case_001"]


def test_validate_fixed_reference_eval_rejects_bad_reference_cardinality(tmp_path):
    row = make_row(tmp_path)
    row["multi_reference_images"] = [row["single_reference_images"][0]]
    path = write_eval_set(tmp_path / "eval.jsonl", row)

    with pytest.raises(ValueError, match="at least two"):
        validate_fixed_reference_eval(path)


def test_validate_fixed_reference_eval_rejects_missing_image(tmp_path):
    row = make_row(tmp_path)
    row["target_image"] = str(tmp_path / "missing.png")
    path = write_eval_set(tmp_path / "eval.jsonl", row)

    with pytest.raises(FileNotFoundError):
        validate_fixed_reference_eval(path)
