from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_subject_manifest import summarize_manifest, validate_manifest, write_summary


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def display_name(value: str) -> str:
    return re.sub(r"[_-]+", " ", value).strip().lower()


def find_concept_folders(input_root: str | Path, min_images_per_subject: int = 2) -> list[tuple[Path, list[Path]]]:
    input_root = Path(input_root)
    concepts = []
    for folder in sorted([path for path in input_root.rglob("*") if path.is_dir()]):
        images = sorted(
            [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda path: path.name.lower(),
        )
        if len(images) >= int(min_images_per_subject):
            concepts.append((folder, images))
    return concepts


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _copy_or_link_path(
    src: Path,
    concept_folder: Path,
    input_root: Path,
    output_root: Path,
    subject_id: str,
    category: str,
    copy_images: bool,
) -> str:
    if copy_images:
        rel_dir = Path(category) / subject_id if category != "customconcept101" else Path(subject_id)
        dst = output_root / rel_dir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return _relative_path(dst, output_root)

    try:
        return _relative_path(src, output_root)
    except ValueError as exc:
        raise ValueError(
            "--copy_images is required when image files are not already under --output_root."
        ) from exc


def convert_customconcept101(
    input_root: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    split: str = "train",
    category_from_parent: bool = False,
    copy_images: bool = False,
    min_images_per_subject: int = 2,
    quality: float = 1.0,
) -> list[dict]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    manifest_path = Path(manifest_path)
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")

    rows = []
    for concept_folder, image_paths in find_concept_folders(input_root, min_images_per_subject=min_images_per_subject):
        subject_id = slugify(concept_folder.name)
        if category_from_parent and concept_folder.parent != input_root:
            category = slugify(concept_folder.parent.name)
        else:
            category = "customconcept101"
        concept_name = display_name(concept_folder.name)
        images = []
        for src in image_paths:
            rel_path = _copy_or_link_path(
                src=src,
                concept_folder=concept_folder,
                input_root=input_root,
                output_root=output_root,
                subject_id=subject_id,
                category=category,
                copy_images=copy_images,
            )
            images.append(
                {
                    "path": rel_path,
                    "caption": f"A reference image of {concept_name}.",
                    "quality": float(quality),
                }
            )
        rows.append(
            {
                "subject_id": subject_id,
                "category": category,
                "split": split,
                "images": images,
            }
        )

    rows = sorted(rows, key=lambda row: (row["category"], row["subject_id"]))
    if not rows:
        raise ValueError(f"No concept folders with at least {min_images_per_subject} images found under {input_root}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Convert a local CustomConcept101-style folder to RefFlowLoRA manifest.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--category_from_parent", action="store_true")
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--min_images_per_subject", type=int, default=2)
    parser.add_argument("--quality", type=float, default=1.0)
    parser.add_argument("--summary_json", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    convert_customconcept101(
        input_root=args.input_root,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
        split=args.split,
        category_from_parent=args.category_from_parent,
        copy_images=args.copy_images,
        min_images_per_subject=args.min_images_per_subject,
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
