from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.convert_sop_manifest import (
    CATEGORY_COLUMNS,
    CLASS_COLUMNS,
    PATH_COLUMNS,
    _first_existing,
    parse_sop_metadata,
    slugify,
)
from scripts.validate_subject_manifest import summarize_manifest, validate_manifest, write_summary


@dataclass(frozen=True)
class SopSubjectCandidate:
    subject_id: str
    class_id: str
    category: str
    semantic_category: str
    image_paths: tuple[Path, ...]


def _semantic_category_from_path(image_path: str) -> str:
    folder = Path(image_path).parts[0] if Path(image_path).parts else "product"
    if folder.endswith("_final"):
        folder = folder[: -len("_final")]
    return folder.replace("_", " ") or "product"


def _load_candidates(
    image_root: Path,
    metadata_path: Path,
    min_images_per_subject: int,
) -> dict[str, list[SopSubjectCandidate]]:
    grouped: dict[str, dict] = {}
    for row in parse_sop_metadata(metadata_path):
        class_id = _first_existing(row, CLASS_COLUMNS)
        image_path = _first_existing(row, PATH_COLUMNS)
        if class_id is None:
            raise ValueError("SOP metadata row is missing a class/product id column.")
        if image_path is None:
            raise ValueError("SOP metadata row is missing an image path column.")
        super_class = _first_existing(row, CATEGORY_COLUMNS)
        category = f"sop_super_{slugify(super_class)}" if super_class is not None else "sop"
        subject_id = f"sop_{slugify(class_id)}"
        src = image_root / image_path
        if not src.exists():
            raise FileNotFoundError(f"SOP image path does not exist: {src}")
        entry = grouped.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "class_id": str(class_id),
                "category": category,
                "semantic_category": _semantic_category_from_path(image_path),
                "image_paths": [],
            },
        )
        entry["image_paths"].append(src)

    by_category: dict[str, list[SopSubjectCandidate]] = {}
    for subject_id in sorted(grouped):
        entry = grouped[subject_id]
        image_paths = tuple(sorted(entry["image_paths"], key=lambda path: path.as_posix()))
        if len(image_paths) < min_images_per_subject:
            continue
        candidate = SopSubjectCandidate(
            subject_id=entry["subject_id"],
            class_id=entry["class_id"],
            category=entry["category"],
            semantic_category=entry["semantic_category"],
            image_paths=image_paths,
        )
        by_category.setdefault(candidate.category, []).append(candidate)
    return {category: sorted(items, key=lambda item: item.subject_id) for category, items in sorted(by_category.items())}


def _copy_or_ref_image(src: Path, output_root: Path, candidate: SopSubjectCandidate, copy_images: bool) -> str:
    if copy_images:
        dst = output_root / candidate.category / candidate.subject_id / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst.relative_to(output_root).as_posix()
    try:
        return src.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise ValueError("--copy_images is required when selected images are not under --output_root.") from exc


def _caption_for(candidate: SopSubjectCandidate) -> str:
    return (
        f"A product photo of a {candidate.semantic_category} from Stanford Online Products, "
        f"item {candidate.class_id}."
    )


def _row_for_candidate(
    candidate: SopSubjectCandidate,
    split: str,
    output_root: Path,
    copy_images: bool,
    max_images_per_subject: int,
    quality: float,
) -> dict:
    caption = _caption_for(candidate)
    image_paths = candidate.image_paths[:max_images_per_subject]
    return {
        "subject_id": candidate.subject_id,
        "category": candidate.category,
        "split": split,
        "images": [
            {
                "path": _copy_or_ref_image(src, output_root, candidate, copy_images=copy_images),
                "caption": caption,
                "quality": float(quality),
            }
            for src in image_paths
        ],
    }


def _pick(items: list[SopSubjectCandidate], count: int, rng: random.Random) -> tuple[list[SopSubjectCandidate], list[SopSubjectCandidate]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    return sorted(shuffled[:count], key=lambda item: item.subject_id), shuffled[count:]


def prepare_sop_small_valid_split(
    image_root: str | Path,
    train_metadata_path: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    test_metadata_path: str | Path | None = None,
    train_subjects_per_category: int = 20,
    val_subjects_per_category: int = 4,
    test_subjects_per_category: int = 4,
    min_images_per_subject: int = 5,
    max_images_per_subject: int = 10,
    subject_seed: int = 42,
    copy_images: bool = False,
    quality: float = 1.0,
) -> list[dict]:
    image_root = Path(image_root)
    train_metadata_path = Path(train_metadata_path)
    output_root = Path(output_root)
    manifest_path = Path(manifest_path)
    if not image_root.exists():
        raise FileNotFoundError(f"image_root does not exist: {image_root}")
    min_images_per_subject = int(min_images_per_subject)
    max_images_per_subject = int(max_images_per_subject)
    if min_images_per_subject < 2:
        raise ValueError("min_images_per_subject must be at least 2 for target/ref sampling.")
    if max_images_per_subject < min_images_per_subject:
        raise ValueError("max_images_per_subject must be >= min_images_per_subject.")

    train_pool = _load_candidates(image_root, train_metadata_path, min_images_per_subject=min_images_per_subject)
    if test_metadata_path is not None:
        test_pool = _load_candidates(image_root, Path(test_metadata_path), min_images_per_subject=min_images_per_subject)
    else:
        test_pool = {}

    rows: list[dict] = []
    categories = sorted(train_pool)
    for category in categories:
        rng = random.Random(f"{int(subject_seed)}:{category}")
        available = train_pool[category]
        needed_from_train = int(train_subjects_per_category) + int(val_subjects_per_category)
        if test_metadata_path is None:
            needed_from_train += int(test_subjects_per_category)
        if len(available) < needed_from_train:
            raise ValueError(
                f"category {category!r} has {len(available)} eligible train subjects, "
                f"but {needed_from_train} are required."
            )
        train_items, remaining = _pick(available, int(train_subjects_per_category), rng)
        val_items, remaining = _pick(remaining, int(val_subjects_per_category), rng)
        if test_metadata_path is None:
            test_items, _remaining = _pick(remaining, int(test_subjects_per_category), rng)
        else:
            test_available = test_pool.get(category, [])
            if len(test_available) < int(test_subjects_per_category):
                raise ValueError(
                    f"category {category!r} has {len(test_available)} eligible test subjects, "
                    f"but {test_subjects_per_category} are required."
                )
            test_items, _remaining = _pick(test_available, int(test_subjects_per_category), rng)

        for split, items in (("train", train_items), ("val", val_items), ("test", test_items)):
            for candidate in items:
                rows.append(
                    _row_for_candidate(
                        candidate,
                        split=split,
                        output_root=output_root,
                        copy_images=copy_images,
                        max_images_per_subject=max_images_per_subject,
                        quality=quality,
                    )
                )

    rows = sorted(rows, key=lambda row: ({"train": 0, "val": 1, "test": 2}[row["split"]], row["category"], row["subject_id"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a subject-disjoint SOP small-valid manifest.")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--train_metadata_path", required=True)
    parser.add_argument("--test_metadata_path", default=None)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--train_subjects_per_category", type=int, default=20)
    parser.add_argument("--val_subjects_per_category", type=int, default=4)
    parser.add_argument("--test_subjects_per_category", type=int, default=4)
    parser.add_argument("--min_images_per_subject", type=int, default=5)
    parser.add_argument("--max_images_per_subject", type=int, default=10)
    parser.add_argument("--subject_seed", type=int, default=42)
    parser.add_argument("--quality", type=float, default=1.0)
    parser.add_argument("--summary_json", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    prepare_sop_small_valid_split(
        image_root=args.image_root,
        train_metadata_path=args.train_metadata_path,
        test_metadata_path=args.test_metadata_path,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
        train_subjects_per_category=args.train_subjects_per_category,
        val_subjects_per_category=args.val_subjects_per_category,
        test_subjects_per_category=args.test_subjects_per_category,
        min_images_per_subject=args.min_images_per_subject,
        max_images_per_subject=args.max_images_per_subject,
        subject_seed=args.subject_seed,
        copy_images=args.copy_images,
        quality=args.quality,
    )
    records = validate_manifest(args.output_root, args.manifest_path)
    summary = summarize_manifest(records)
    if args.summary_json:
        write_summary(summary, args.summary_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
