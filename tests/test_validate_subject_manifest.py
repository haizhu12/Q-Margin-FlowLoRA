import json

import pytest
from PIL import Image

from scripts.validate_subject_manifest import (
    ManifestValidationError,
    summarize_manifest,
    validate_manifest,
    write_summary,
)


def make_image(path, color=(20, 40, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path)


def write_manifest(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def subject_row(subject_id, split="train", category="toy", image_count=2):
    return {
        "subject_id": subject_id,
        "category": category,
        "split": split,
        "images": [
            {"path": f"{subject_id}/{idx}.png", "caption": f"{subject_id} view {idx}", "quality": 1.0}
            for idx in range(image_count)
        ],
    }


def materialize_images(root, rows):
    for row in rows:
        for image in row["images"]:
            make_image(root / image["path"])


def test_validate_manifest_accepts_valid_rows_and_returns_summary(tmp_path):
    root = tmp_path / "subjects"
    rows = [
        subject_row("s1", split="train", category="toy", image_count=2),
        subject_row("s2", split="train", category="toy", image_count=3),
        subject_row("s3", split="val", category="shoe", image_count=2),
    ]
    materialize_images(root, rows)
    manifest = tmp_path / "subjects.jsonl"
    write_manifest(manifest, rows)

    records = validate_manifest(root, manifest)
    summary = summarize_manifest(records)

    assert summary["subject_count"] == 3
    assert summary["image_count"] == 7
    assert summary["split_counts"] == {"train": 2, "val": 1}
    assert summary["category_counts"] == {"shoe": 1, "toy": 2}
    assert summary["images_per_subject"]["min"] == 2
    assert summary["images_per_subject"]["max"] == 3
    assert summary["images_per_subject"]["mean"] == pytest.approx(7 / 3)


def test_validate_manifest_reports_missing_required_fields(tmp_path):
    root = tmp_path / "subjects"
    manifest = tmp_path / "subjects.jsonl"
    write_manifest(manifest, [{"subject_id": "s1", "split": "train", "images": []}])

    with pytest.raises(ManifestValidationError, match="line 1: missing required field: category"):
        validate_manifest(root, manifest)


def test_validate_manifest_resolves_paths_and_rejects_missing_images(tmp_path):
    root = tmp_path / "subjects"
    manifest = tmp_path / "subjects.jsonl"
    write_manifest(
        manifest,
        [
            {
                "subject_id": "s1",
                "category": "toy",
                "split": "train",
                "images": [{"path": "s1/missing.png", "caption": "front", "quality": 1.0}],
            }
        ],
    )

    with pytest.raises(ManifestValidationError, match="image path does not exist"):
        validate_manifest(root, manifest, allow_single_subject_debug=True)


def test_validate_manifest_rejects_subject_split_leakage(tmp_path):
    root = tmp_path / "subjects"
    rows = [
        subject_row("s1", split="train", category="toy", image_count=2),
        subject_row("s1", split="val", category="toy", image_count=2),
        subject_row("s2", split="train", category="toy", image_count=2),
    ]
    materialize_images(root, rows)
    manifest = tmp_path / "subjects.jsonl"
    write_manifest(manifest, rows)

    with pytest.raises(ManifestValidationError, match="appears in multiple splits"):
        validate_manifest(root, manifest)


def test_validate_manifest_requires_two_subjects_unless_debug(tmp_path):
    root = tmp_path / "subjects"
    rows = [subject_row("s1", split="train", category="toy", image_count=2)]
    materialize_images(root, rows)
    manifest = tmp_path / "subjects.jsonl"
    write_manifest(manifest, rows)

    with pytest.raises(ManifestValidationError, match="at least 2 subjects"):
        validate_manifest(root, manifest)

    records = validate_manifest(root, manifest, allow_single_subject_debug=True)
    assert len(records) == 1


def test_write_summary_creates_json_file(tmp_path):
    summary = {"subject_count": 2, "image_count": 4}
    out = tmp_path / "summary.json"

    write_summary(summary, out)

    assert json.loads(out.read_text(encoding="utf-8")) == summary
