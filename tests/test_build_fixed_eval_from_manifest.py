import json

from PIL import Image

from scripts.build_fixed_eval_from_manifest import build_fixed_eval_from_manifest, main
from scripts.validate_fixed_reference_eval import validate_fixed_reference_eval


def make_image(path, color=(30, 60, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path)


def write_manifest(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def subject_row(subject_id, category, split, image_count=4):
    return {
        "subject_id": subject_id,
        "category": category,
        "split": split,
        "images": [
            {
                "path": f"{category}/{subject_id}/{idx}.png",
                "caption": f"A product photo of a {category} from Stanford Online Products, item {subject_id}.",
                "quality": 1.0,
            }
            for idx in range(image_count)
        ],
    }


def materialize_images(root, rows):
    for row in rows:
        for image in row["images"]:
            make_image(root / image["path"])


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_fixed_eval_from_manifest_uses_test_split_and_absolute_project_relative_paths(tmp_path):
    data_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects.jsonl"
    output = tmp_path / "eval.jsonl"
    rows = [
        subject_row("train_bike", "bike", "train"),
        subject_row("test_bike", "bike", "test"),
        subject_row("test_chair", "chair", "test"),
        subject_row("val_chair", "chair", "val"),
    ]
    materialize_images(data_root, rows)
    write_manifest(manifest, rows)

    cases = build_fixed_eval_from_manifest(
        data_root=data_root,
        manifest_path=manifest,
        output_path=output,
        split="test",
        cases_per_category=1,
        seed=123,
        root=tmp_path,
    )

    assert [case["subject_id"] for case in cases] == ["test_bike", "test_chair"]
    assert all("train" not in case["subject_id"] for case in cases)
    assert all(case["target_image"].startswith("subjects/") for case in cases)
    assert all(len(case["single_reference_images"]) == 1 for case in cases)
    assert all(len(case["multi_reference_images"]) == 2 for case in cases)
    assert all("same item" in case["prompt"] for case in cases)
    summary = validate_fixed_reference_eval(output, root=tmp_path)
    assert summary["case_count"] == 2
    assert summary["category_count"] == 2


def test_build_fixed_eval_from_manifest_cli_writes_jsonl(tmp_path):
    data_root = tmp_path / "subjects"
    manifest = tmp_path / "subjects.jsonl"
    output = tmp_path / "eval.jsonl"
    rows = [
        subject_row("test_bike", "bike", "test"),
        subject_row("test_chair", "chair", "test"),
    ]
    materialize_images(data_root, rows)
    write_manifest(manifest, rows)

    summary = main(
        [
            "--data_root",
            str(data_root),
            "--manifest_path",
            str(manifest),
            "--output_path",
            str(output),
            "--split",
            "test",
            "--cases_per_category",
            "1",
            "--root",
            str(tmp_path),
        ]
    )

    assert output.exists()
    assert len(read_jsonl(output)) == 2
    assert summary["case_count"] == 2
