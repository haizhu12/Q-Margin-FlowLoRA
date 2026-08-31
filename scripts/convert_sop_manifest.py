from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_subject_manifest import summarize_manifest, validate_manifest, write_summary


PATH_COLUMNS = ("path", "image_path", "filepath", "file_name", "filename")
CLASS_COLUMNS = ("class_id", "product_id", "item_id", "subject_id")
CATEGORY_COLUMNS = ("super_class_id", "category", "category_id", "super_category")
DEFAULT_SOP_HEADER = ["image_id", "class_id", "super_class_id", "path"]


def slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def _split_metadata_line(line: str) -> list[str]:
    if "," in line:
        return [part.strip() for part in line.split(",")]
    if "\t" in line:
        return [part.strip() for part in line.split("\t")]
    return line.split()


def _looks_like_header(parts: list[str]) -> bool:
    lowered = {part.lower() for part in parts}
    return bool(lowered.intersection({"class_id", "product_id", "item_id", "path", "image_path"}))


def parse_sop_metadata(metadata_path: str | Path) -> list[dict[str, str]]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_path does not exist: {metadata_path}")
    raw_lines = [line.strip() for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError(f"metadata file is empty: {metadata_path}")

    first = _split_metadata_line(raw_lines[0])
    if _looks_like_header(first):
        header = [part.lower() for part in first]
        data_lines = raw_lines[1:]
    else:
        header = DEFAULT_SOP_HEADER
        data_lines = raw_lines

    rows = []
    for line_no, line in enumerate(data_lines, start=2 if _looks_like_header(first) else 1):
        parts = _split_metadata_line(line)
        if len(parts) < len(header):
            raise ValueError(f"line {line_no}: expected at least {len(header)} columns, got {len(parts)}")
        row = {header[idx]: parts[idx] for idx in range(len(header))}
        rows.append(row)
    return rows


def _first_existing(row: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    return None


def _copy_or_ref_path(src: Path, image_root: Path, output_root: Path, subject_id: str, category: str, copy_images: bool) -> str:
    if copy_images:
        dst = output_root / category / subject_id / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst.relative_to(output_root).as_posix()
    try:
        return src.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise ValueError("--copy_images is required when image files are not already under --output_root.") from exc


def convert_sop(
    image_root: str | Path,
    metadata_path: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    split: str = "train",
    copy_images: bool = False,
    min_images_per_subject: int = 2,
    max_subjects: int | None = None,
    max_images_per_subject: int | None = None,
    subject_seed: int = 42,
    quality: float = 1.0,
) -> list[dict]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    min_images_per_subject = int(min_images_per_subject)
    if min_images_per_subject < 1:
        raise ValueError("min_images_per_subject must be positive")
    if max_subjects is not None:
        max_subjects = int(max_subjects)
        if max_subjects < 1:
            raise ValueError("max_subjects must be positive")
    if max_images_per_subject is not None:
        max_images_per_subject = int(max_images_per_subject)
        if max_images_per_subject < min_images_per_subject:
            raise ValueError("max_images_per_subject must be >= min_images_per_subject")
    image_root = Path(image_root)
    output_root = Path(output_root)
    manifest_path = Path(manifest_path)
    if not image_root.exists():
        raise FileNotFoundError(f"image_root does not exist: {image_root}")

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
                "category": category,
                "split": split,
                "images": [],
                "_class_id": str(class_id),
            },
        )
        entry["images"].append(
            {
                "_src": src,
                "caption": f"A reference image of Stanford Online Products item {class_id}.",
                "quality": float(quality),
            }
        )

    rows = []
    for subject_id in sorted(grouped):
        row = grouped[subject_id]
        if len(row["images"]) < min_images_per_subject:
            continue
        row = {key: value for key, value in row.items() if not key.startswith("_")}
        row["images"] = sorted(row["images"], key=lambda image: str(image["_src"]))
        if max_images_per_subject is not None:
            row["images"] = row["images"][:max_images_per_subject]
        rows.append(row)

    if not rows:
        raise ValueError(f"No SOP products with at least {min_images_per_subject} images were found.")
    if max_subjects is not None and len(rows) > max_subjects:
        rng = random.Random(int(subject_seed))
        rows = sorted(rng.sample(rows, k=max_subjects), key=lambda row: (row["category"], row["subject_id"]))
    for row in rows:
        finalized_images = []
        for image in row["images"]:
            src = image["_src"]
            finalized_images.append(
                {
                    "path": _copy_or_ref_path(
                        src=src,
                        image_root=image_root,
                        output_root=output_root,
                        subject_id=row["subject_id"],
                        category=row["category"],
                        copy_images=copy_images,
                    ),
                    "caption": image["caption"],
                    "quality": image["quality"],
                }
            )
        row["images"] = finalized_images
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Convert Stanford Online Products metadata to RefFlowLoRA manifest.")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--min_images_per_subject", type=int, default=2)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--max_images_per_subject", type=int, default=None)
    parser.add_argument("--subject_seed", type=int, default=42)
    parser.add_argument("--quality", type=float, default=1.0)
    parser.add_argument("--allow_single_subject_debug", action="store_true")
    parser.add_argument("--summary_json", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    convert_sop(
        image_root=args.image_root,
        metadata_path=args.metadata_path,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
        split=args.split,
        copy_images=args.copy_images,
        min_images_per_subject=args.min_images_per_subject,
        max_subjects=args.max_subjects,
        max_images_per_subject=args.max_images_per_subject,
        subject_seed=args.subject_seed,
        quality=args.quality,
    )
    records = validate_manifest(
        args.output_root,
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
