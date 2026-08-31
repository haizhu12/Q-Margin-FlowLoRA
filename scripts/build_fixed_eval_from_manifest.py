from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_fixed_reference_eval import validate_fixed_reference_eval
from scripts.validate_subject_manifest import ManifestRecord, load_manifest_records


def _path_for_eval(data_root: Path, image_path: str, root: Path) -> str:
    path = Path(image_path)
    if not path.is_absolute():
        path = data_root / path
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _semantic_name(record: ManifestRecord) -> str:
    if record.images:
        caption = record.images[0].caption
        match = re.search(r"product photo of an? (.+?) from Stanford Online Products", caption)
        if match:
            return match.group(1).strip()
    category = record.category
    if category.startswith("sop_super_"):
        return "product"
    return category.replace("_", " ")


def _case_for_record(record: ManifestRecord, index: int, data_root: Path, root: Path, seed: int) -> dict:
    images = sorted(record.images, key=lambda image: image.path)
    if len(images) < 3:
        raise ValueError(f"subject_id {record.subject_id!r} needs at least 3 images for fixed eval.")
    semantic_name = _semantic_name(record)
    return {
        "case_id": f"sop_small_valid_{index:03d}_{record.subject_id}",
        "subject_id": record.subject_id,
        "category": record.category,
        "prompt": f"A studio product photo of the same item, a {semantic_name}, on a clean white background.",
        "seed": int(seed) + index,
        "target_image": _path_for_eval(data_root, images[2].path, root),
        "single_reference_images": [_path_for_eval(data_root, images[0].path, root)],
        "multi_reference_images": [
            _path_for_eval(data_root, images[0].path, root),
            _path_for_eval(data_root, images[1].path, root),
        ],
        "notes": "SOP small-valid held-out fixed-eval case; subject_id is from the manifest test split.",
    }


def build_fixed_eval_from_manifest(
    data_root: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    split: str = "test",
    cases_per_category: int = 1,
    seed: int = 123,
    root: str | Path = ".",
) -> list[dict]:
    data_root = Path(data_root)
    output_path = Path(output_path)
    root = Path(root)
    cases_per_category = int(cases_per_category)
    if cases_per_category < 1:
        raise ValueError("cases_per_category must be positive.")
    records = [record for record in load_manifest_records(data_root, manifest_path) if record.split == split]
    if not records:
        raise ValueError(f"No records found for split={split!r}.")

    by_category: dict[str, list[ManifestRecord]] = {}
    for record in records:
        if len(record.images) >= 3:
            by_category.setdefault(record.category, []).append(record)
    if not by_category:
        raise ValueError("No eligible records with at least 3 images were found.")

    selected: list[ManifestRecord] = []
    for category in sorted(by_category):
        items = sorted(by_category[category], key=lambda record: record.subject_id)
        if len(items) < cases_per_category:
            raise ValueError(
                f"category {category!r} has {len(items)} eligible records, "
                f"but {cases_per_category} cases were requested."
            )
        rng = random.Random(f"{int(seed)}:{category}")
        selected.extend(sorted(rng.sample(items, k=cases_per_category), key=lambda record: record.subject_id))

    cases = [
        _case_for_record(record, index=index, data_root=data_root, root=root, seed=seed)
        for index, record in enumerate(selected)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")
    return cases


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a fixed reference eval set from a subject manifest split.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--cases_per_category", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--root", default=".")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    build_fixed_eval_from_manifest(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        split=args.split,
        cases_per_category=args.cases_per_category,
        seed=args.seed,
        root=args.root,
    )
    summary = validate_fixed_reference_eval(args.output_path, root=args.root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
