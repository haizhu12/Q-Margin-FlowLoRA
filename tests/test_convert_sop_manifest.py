import json

from PIL import Image

from scripts.convert_sop_manifest import convert_sop, main, parse_sop_metadata
from scripts.validate_subject_manifest import summarize_manifest, validate_manifest


def make_image(path, color=(50, 80, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_fake_sop(image_root, metadata_path):
    make_image(image_root / "bicycle_final" / "100_a.jpg")
    make_image(image_root / "bicycle_final" / "100_b.jpg")
    make_image(image_root / "chair_final" / "200_a.jpg")
    make_image(image_root / "chair_final" / "200_b.jpg")
    make_image(image_root / "chair_final" / "200_c.jpg")
    make_image(image_root / "toaster_final" / "300_only.jpg")
    metadata_path.write_text(
        "\n".join(
            [
                "image_id class_id super_class_id path",
                "1 100 10 bicycle_final/100_a.jpg",
                "2 100 10 bicycle_final/100_b.jpg",
                "3 200 20 chair_final/200_a.jpg",
                "4 200 20 chair_final/200_b.jpg",
                "5 200 20 chair_final/200_c.jpg",
                "6 300 30 toaster_final/300_only.jpg",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_sop_metadata_reads_headered_whitespace_file(tmp_path):
    image_root = tmp_path / "images"
    metadata = tmp_path / "Ebay_train.txt"
    make_fake_sop(image_root, metadata)

    rows = parse_sop_metadata(metadata)

    assert rows[0]["class_id"] == "100"
    assert rows[0]["super_class_id"] == "10"
    assert rows[0]["path"] == "bicycle_final/100_a.jpg"
    assert len(rows) == 6


def test_convert_sop_copies_images_and_writes_valid_manifest(tmp_path):
    image_root = tmp_path / "sop_images"
    metadata = tmp_path / "Ebay_train.txt"
    output_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects_sop.jsonl"
    make_fake_sop(image_root, metadata)

    rows = convert_sop(
        image_root=image_root,
        metadata_path=metadata,
        output_root=output_root,
        manifest_path=manifest,
        split="train",
        copy_images=True,
    )

    assert [row["subject_id"] for row in rows] == ["sop_100", "sop_200"]
    assert {row["category"] for row in rows} == {"sop_super_10", "sop_super_20"}
    assert [len(row["images"]) for row in rows] == [2, 3]
    assert all((output_root / image["path"]).exists() for row in rows for image in row["images"])
    assert rows[0]["images"][0]["caption"] == "A reference image of Stanford Online Products item 100."
    records = validate_manifest(output_root, manifest)
    summary = summarize_manifest(records)
    assert summary["subject_count"] == 2
    assert summary["image_count"] == 5


def test_convert_sop_without_copy_uses_paths_relative_to_image_root(tmp_path):
    image_root = tmp_path / "sop_images"
    metadata = tmp_path / "Ebay_test.txt"
    manifest = tmp_path / "subjects_sop.jsonl"
    make_fake_sop(image_root, metadata)

    rows = convert_sop(
        image_root=image_root,
        metadata_path=metadata,
        output_root=image_root,
        manifest_path=manifest,
        split="test",
        copy_images=False,
    )

    assert rows[0]["split"] == "test"
    assert rows[0]["images"][0]["path"] == "bicycle_final/100_a.jpg"
    records = validate_manifest(image_root, manifest)
    assert summarize_manifest(records)["split_counts"] == {"test": 2}


def test_convert_sop_limits_subjects_and_images_per_subject(tmp_path):
    image_root = tmp_path / "sop_images"
    metadata = tmp_path / "Ebay_train.txt"
    output_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects_sop.jsonl"
    make_fake_sop(image_root, metadata)

    rows = convert_sop(
        image_root=image_root,
        metadata_path=metadata,
        output_root=output_root,
        manifest_path=manifest,
        split="train",
        copy_images=True,
        max_subjects=1,
        max_images_per_subject=2,
        subject_seed=7,
    )

    assert len(rows) == 1
    assert len(rows[0]["images"]) == 2
    assert all((output_root / image["path"]).exists() for image in rows[0]["images"])
    records = validate_manifest(output_root, manifest, allow_single_subject_debug=True)
    summary = summarize_manifest(records)
    assert summary["subject_count"] == 1
    assert summary["image_count"] == 2


def test_convert_sop_copies_only_selected_subset_images(tmp_path):
    image_root = tmp_path / "sop_images"
    metadata = tmp_path / "Ebay_train.txt"
    output_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects_sop.jsonl"
    make_fake_sop(image_root, metadata)

    rows = convert_sop(
        image_root=image_root,
        metadata_path=metadata,
        output_root=output_root,
        manifest_path=manifest,
        split="train",
        copy_images=True,
        max_subjects=1,
        max_images_per_subject=2,
        subject_seed=7,
    )

    copied_images = [path for path in output_root.rglob("*") if path.is_file() and path.suffix.lower() == ".jpg"]
    manifest_paths = {image["path"] for row in rows for image in row["images"]}
    copied_rel_paths = {path.relative_to(output_root).as_posix() for path in copied_images}
    assert copied_rel_paths == manifest_paths


def test_convert_sop_rejects_image_cap_below_min_images(tmp_path):
    image_root = tmp_path / "sop_images"
    metadata = tmp_path / "Ebay_train.txt"
    manifest = tmp_path / "subjects_sop.jsonl"
    make_fake_sop(image_root, metadata)

    try:
        convert_sop(
            image_root=image_root,
            metadata_path=metadata,
            output_root=image_root,
            manifest_path=manifest,
            min_images_per_subject=2,
            max_images_per_subject=1,
        )
    except ValueError as exc:
        assert "max_images_per_subject" in str(exc)
    else:
        raise AssertionError("Expected max_images_per_subject below min_images_per_subject to fail.")


def test_cli_writes_manifest_and_summary_json(tmp_path):
    image_root = tmp_path / "sop_images"
    metadata = tmp_path / "Ebay_train.txt"
    output_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects_sop.jsonl"
    summary_path = tmp_path / "summary.json"
    make_fake_sop(image_root, metadata)

    summary = main(
        [
            "--image_root",
            str(image_root),
            "--metadata_path",
            str(metadata),
            "--output_root",
            str(output_root),
            "--manifest_path",
            str(manifest),
            "--split",
            "train",
            "--copy_images",
            "--summary_json",
            str(summary_path),
        ]
    )

    assert manifest.exists()
    assert summary_path.exists()
    assert len(read_jsonl(manifest)) == 2
    assert summary["subject_count"] == 2
    assert json.loads(summary_path.read_text(encoding="utf-8"))["image_count"] == 5


def test_cli_accepts_subset_controls(tmp_path):
    image_root = tmp_path / "sop_images"
    metadata = tmp_path / "Ebay_train.txt"
    output_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects_sop.jsonl"
    summary_path = tmp_path / "summary.json"
    make_fake_sop(image_root, metadata)

    summary = main(
        [
            "--image_root",
            str(image_root),
            "--metadata_path",
            str(metadata),
            "--output_root",
            str(output_root),
            "--manifest_path",
            str(manifest),
            "--split",
            "train",
            "--copy_images",
            "--max_subjects",
            "1",
            "--max_images_per_subject",
            "2",
            "--subject_seed",
            "7",
            "--summary_json",
            str(summary_path),
            "--allow_single_subject_debug",
        ]
    )

    assert summary["subject_count"] == 1
    assert summary["image_count"] == 2
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
