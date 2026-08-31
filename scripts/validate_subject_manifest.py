from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestImage:
    path: str
    caption: str
    quality: float


@dataclass(frozen=True)
class ManifestRecord:
    subject_id: str
    category: str
    split: str
    images: list[ManifestImage]
    line_no: int


def _resolve_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = data_root / path
    return path


def _require(obj: dict, key: str, line_no: int):
    if key not in obj:
        raise ManifestValidationError(f"line {line_no}: missing required field: {key}")
    return obj[key]


def _parse_image(image_obj: dict, line_no: int, image_idx: int, data_root: Path) -> ManifestImage:
    if not isinstance(image_obj, dict):
        raise ManifestValidationError(f"line {line_no}: images[{image_idx}] must be an object")
    path = _require(image_obj, "path", line_no)
    caption = _require(image_obj, "caption", line_no)
    quality = _require(image_obj, "quality", line_no)
    if not isinstance(path, str) or not path:
        raise ManifestValidationError(f"line {line_no}: images[{image_idx}].path must be a non-empty string")
    if not isinstance(caption, str) or not caption.strip():
        raise ManifestValidationError(f"line {line_no}: images[{image_idx}].caption must be a non-empty string")
    try:
        quality = float(quality)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"line {line_no}: images[{image_idx}].quality must be numeric") from exc
    resolved = _resolve_path(data_root, path)
    if not resolved.exists():
        raise ManifestValidationError(f"line {line_no}: image path does not exist: {resolved}")
    return ManifestImage(path=path, caption=caption, quality=quality)


def _parse_record(obj: dict, line_no: int, data_root: Path) -> ManifestRecord:
    if not isinstance(obj, dict):
        raise ManifestValidationError(f"line {line_no}: row must be a JSON object")
    subject_id = _require(obj, "subject_id", line_no)
    category = _require(obj, "category", line_no)
    split = _require(obj, "split", line_no)
    images = _require(obj, "images", line_no)

    if not isinstance(subject_id, str) or not subject_id:
        raise ManifestValidationError(f"line {line_no}: subject_id must be a non-empty string")
    if not isinstance(category, str) or not category:
        raise ManifestValidationError(f"line {line_no}: category must be a non-empty string")
    if split not in {"train", "val", "test"}:
        raise ManifestValidationError(f"line {line_no}: split must be one of: train, val, test")
    if not isinstance(images, list) or not images:
        raise ManifestValidationError(f"line {line_no}: images must be a non-empty list")

    parsed_images = [_parse_image(image, line_no, idx, data_root) for idx, image in enumerate(images)]
    return ManifestRecord(
        subject_id=subject_id,
        category=category,
        split=split,
        images=parsed_images,
        line_no=line_no,
    )


def load_manifest_records(data_root: str | Path, manifest_path: str | Path) -> list[ManifestRecord]:
    data_root = Path(data_root)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise ManifestValidationError(f"manifest path does not exist: {manifest_path}")

    records = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            records.append(_parse_record(obj, line_no, data_root))
    if not records:
        raise ManifestValidationError("manifest contains no records")
    return records


def validate_manifest(
    data_root: str | Path,
    manifest_path: str | Path,
    allow_single_subject_debug: bool = False,
) -> list[ManifestRecord]:
    records = load_manifest_records(data_root, manifest_path)
    subject_to_splits: dict[str, set[str]] = {}
    subject_image_counts: dict[str, int] = {}

    for record in records:
        subject_to_splits.setdefault(record.subject_id, set()).add(record.split)
        subject_image_counts[record.subject_id] = subject_image_counts.get(record.subject_id, 0) + len(record.images)

    if len(subject_to_splits) < 2 and not allow_single_subject_debug:
        raise ManifestValidationError(
            "RefFlowLoRA manifests require at least 2 subjects. "
            "Use --allow_single_subject_debug only for smoke/debug validation."
        )

    for subject_id, splits in sorted(subject_to_splits.items()):
        if len(splits) > 1:
            raise ManifestValidationError(f"subject_id {subject_id!r} appears in multiple splits: {sorted(splits)}")
        if subject_image_counts[subject_id] < 2:
            raise ManifestValidationError(f"subject_id {subject_id!r} needs at least 2 images")

    return records


def summarize_manifest(records: list[ManifestRecord]) -> dict:
    split_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    images_by_subject: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        category_counts[record.category] = category_counts.get(record.category, 0) + 1
        images_by_subject[record.subject_id] = images_by_subject.get(record.subject_id, 0) + len(record.images)

    image_counts = list(images_by_subject.values())
    image_count = sum(image_counts)
    return {
        "subject_count": len(images_by_subject),
        "record_count": len(records),
        "image_count": image_count,
        "split_counts": dict(sorted(split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "images_per_subject": {
            "min": min(image_counts),
            "max": max(image_counts),
            "mean": image_count / len(image_counts),
        },
    }


def write_summary(summary: dict, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate a Q-Margin RefFlowLoRA subject manifest.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--allow_single_subject_debug", action="store_true")
    parser.add_argument("--summary_json", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    records = validate_manifest(
        args.data_root,
        args.manifest_path,
        allow_single_subject_debug=args.allow_single_subject_debug,
    )
    summary = summarize_manifest(records)
    if args.summary_json:
        write_summary(summary, args.summary_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
