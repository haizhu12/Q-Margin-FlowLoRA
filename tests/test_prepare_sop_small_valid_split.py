import json

from PIL import Image

from scripts.prepare_sop_small_valid_split import main, prepare_sop_small_valid_split
from scripts.validate_subject_manifest import summarize_manifest, validate_manifest


def make_image(path, color=(50, 80, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path)


def make_category_subjects(image_root, rows, category_folder, super_id, class_start, subject_count, image_count):
    for subject_offset in range(subject_count):
        class_id = class_start + subject_offset
        for image_idx in range(image_count):
            rel_path = f"{category_folder}_final/{class_id}_{image_idx}.jpg"
            make_image(image_root / rel_path, color=(class_id % 255, image_idx * 20 % 255, super_id * 20 % 255))
            rows.append(f"{len(rows) + 1} {class_id} {super_id} {rel_path}")


def write_fake_sop(image_root, train_metadata, test_metadata):
    train_rows = ["image_id class_id super_class_id path"]
    test_rows = ["image_id class_id super_class_id path"]
    for folder, super_id, class_start in [
        ("bicycle", 1, 1000),
        ("chair", 2, 2000),
    ]:
        make_category_subjects(image_root, train_rows, folder, super_id, class_start, subject_count=4, image_count=3)
        make_category_subjects(image_root, test_rows, folder, super_id, class_start + 100, subject_count=2, image_count=3)
    train_metadata.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    test_metadata.write_text("\n".join(test_rows) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prepare_sop_small_valid_split_writes_subject_disjoint_captioned_manifest(tmp_path):
    image_root = tmp_path / "raw_sop"
    train_metadata = tmp_path / "Ebay_train.txt"
    test_metadata = tmp_path / "Ebay_test.txt"
    output_root = tmp_path / "subjects" / "sop_small_valid"
    manifest = tmp_path / "subjects" / "subjects_sop_small_valid.jsonl"
    write_fake_sop(image_root, train_metadata, test_metadata)

    rows = prepare_sop_small_valid_split(
        image_root=image_root,
        train_metadata_path=train_metadata,
        test_metadata_path=test_metadata,
        output_root=output_root,
        manifest_path=manifest,
        train_subjects_per_category=2,
        val_subjects_per_category=1,
        test_subjects_per_category=1,
        min_images_per_subject=2,
        max_images_per_subject=2,
        subject_seed=7,
        copy_images=True,
    )

    assert manifest.exists()
    assert len(rows) == 8
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    records = validate_manifest(output_root, manifest)
    summary = summarize_manifest(records)
    assert summary["split_counts"] == {"test": 2, "train": 4, "val": 2}
    assert summary["category_counts"] == {"sop_super_1": 4, "sop_super_2": 4}
    assert summary["images_per_subject"]["min"] == 2
    assert summary["images_per_subject"]["max"] == 2
    for row in read_jsonl(manifest):
        captions = {image["caption"] for image in row["images"]}
        assert len(captions) == 1
        assert "Stanford Online Products" in next(iter(captions))
        assert any(name in next(iter(captions)) for name in ("bicycle", "chair"))
        assert all((output_root / image["path"]).exists() for image in row["images"])


def test_prepare_sop_small_valid_split_cli_writes_summary(tmp_path):
    image_root = tmp_path / "raw_sop"
    train_metadata = tmp_path / "Ebay_train.txt"
    test_metadata = tmp_path / "Ebay_test.txt"
    output_root = tmp_path / "subjects" / "sop_small_valid"
    manifest = tmp_path / "subjects" / "subjects_sop_small_valid.jsonl"
    summary_path = tmp_path / "subjects" / "subjects_sop_small_valid_summary.json"
    write_fake_sop(image_root, train_metadata, test_metadata)

    summary = main(
        [
            "--image_root",
            str(image_root),
            "--train_metadata_path",
            str(train_metadata),
            "--test_metadata_path",
            str(test_metadata),
            "--output_root",
            str(output_root),
            "--manifest_path",
            str(manifest),
            "--copy_images",
            "--train_subjects_per_category",
            "2",
            "--val_subjects_per_category",
            "1",
            "--test_subjects_per_category",
            "1",
            "--min_images_per_subject",
            "2",
            "--max_images_per_subject",
            "2",
            "--summary_json",
            str(summary_path),
        ]
    )

    assert summary_path.exists()
    assert summary["subject_count"] == 8
    assert json.loads(summary_path.read_text(encoding="utf-8"))["split_counts"] == {"test": 2, "train": 4, "val": 2}
