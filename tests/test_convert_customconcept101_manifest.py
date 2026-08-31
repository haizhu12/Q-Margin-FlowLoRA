import json

from PIL import Image

from scripts.convert_customconcept101_manifest import convert_customconcept101, main
from scripts.validate_subject_manifest import summarize_manifest, validate_manifest


def make_image(path, color=(30, 60, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_fake_customconcept101(input_root):
    make_image(input_root / "toys" / "red robot" / "000.jpg")
    make_image(input_root / "toys" / "red robot" / "001.png")
    make_image(input_root / "shoes" / "blue sneaker" / "a.jpeg")
    make_image(input_root / "shoes" / "blue sneaker" / "b.jpg")
    (input_root / "shoes" / "blue sneaker" / "README.txt").write_text("ignore me", encoding="utf-8")
    make_image(input_root / "toys" / "single view toy" / "only.jpg")


def test_convert_customconcept101_copies_images_and_writes_valid_manifest(tmp_path):
    input_root = tmp_path / "customconcept101"
    output_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects.jsonl"
    make_fake_customconcept101(input_root)

    rows = convert_customconcept101(
        input_root=input_root,
        output_root=output_root,
        manifest_path=manifest,
        split="train",
        category_from_parent=True,
        copy_images=True,
    )

    assert [row["subject_id"] for row in rows] == ["blue_sneaker", "red_robot"]
    assert {row["category"] for row in rows} == {"shoes", "toys"}
    assert all(len(row["images"]) == 2 for row in rows)
    assert all((output_root / image["path"]).exists() for row in rows for image in row["images"])
    assert rows[0]["images"][0]["caption"].startswith("A reference image of blue sneaker")
    records = validate_manifest(output_root, manifest)
    summary = summarize_manifest(records)
    assert summary["subject_count"] == 2
    assert summary["image_count"] == 4


def test_convert_customconcept101_skips_concepts_with_too_few_images(tmp_path):
    input_root = tmp_path / "customconcept101"
    output_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects.jsonl"
    make_fake_customconcept101(input_root)

    rows = convert_customconcept101(
        input_root=input_root,
        output_root=output_root,
        manifest_path=manifest,
        split="train",
        category_from_parent=True,
        copy_images=True,
    )

    assert "single_view_toy" not in {row["subject_id"] for row in rows}


def test_convert_customconcept101_without_copy_uses_paths_relative_to_input_root(tmp_path):
    input_root = tmp_path / "customconcept101"
    manifest = tmp_path / "subjects.jsonl"
    make_fake_customconcept101(input_root)

    rows = convert_customconcept101(
        input_root=input_root,
        output_root=input_root,
        manifest_path=manifest,
        split="val",
        category_from_parent=True,
        copy_images=False,
    )

    assert rows[0]["split"] == "val"
    assert rows[0]["images"][0]["path"].startswith("shoes/blue sneaker/")
    records = validate_manifest(input_root, manifest)
    assert summarize_manifest(records)["split_counts"] == {"val": 2}


def test_cli_writes_manifest_and_summary_json(tmp_path):
    input_root = tmp_path / "customconcept101"
    output_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects.jsonl"
    summary_path = tmp_path / "summary.json"
    make_fake_customconcept101(input_root)

    summary = main(
        [
            "--input_root",
            str(input_root),
            "--output_root",
            str(output_root),
            "--manifest_path",
            str(manifest),
            "--split",
            "train",
            "--category_from_parent",
            "--copy_images",
            "--summary_json",
            str(summary_path),
        ]
    )

    assert manifest.exists()
    assert summary_path.exists()
    assert len(read_jsonl(manifest)) == 2
    assert summary["subject_count"] == 2
    assert json.loads(summary_path.read_text(encoding="utf-8"))["image_count"] == 4
