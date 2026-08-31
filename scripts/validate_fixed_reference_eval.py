from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_KEYS = {
    "case_id",
    "subject_id",
    "category",
    "prompt",
    "seed",
    "target_image",
    "single_reference_images",
    "multi_reference_images",
}


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return rows


def validate_fixed_reference_eval(path: str | Path, root: str | Path = ".") -> dict:
    root = Path(root)
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError("Fixed reference eval set is empty.")

    seen_case_ids = set()
    subject_ids = set()
    categories = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_KEYS - set(row)
        if missing:
            raise ValueError(f"Row {index} missing keys: {sorted(missing)}")
        case_id = str(row["case_id"])
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        subject_ids.add(str(row["subject_id"]))
        categories.add(str(row["category"]))

        prompt = str(row["prompt"]).strip()
        if not prompt:
            raise ValueError(f"Row {case_id} has an empty prompt.")

        try:
            int(row["seed"])
        except Exception as exc:
            raise ValueError(f"Row {case_id} seed must be an integer.") from exc

        single_refs = row["single_reference_images"]
        multi_refs = row["multi_reference_images"]
        if not isinstance(single_refs, list) or len(single_refs) != 1:
            raise ValueError(f"Row {case_id} single_reference_images must contain exactly one path.")
        if not isinstance(multi_refs, list) or len(multi_refs) < 2:
            raise ValueError(f"Row {case_id} multi_reference_images must contain at least two paths.")

        paths = [row["target_image"], *single_refs, *multi_refs]
        for image_path in paths:
            resolved = Path(image_path)
            if not resolved.is_absolute():
                resolved = root / resolved
            if not resolved.exists():
                raise FileNotFoundError(f"Row {case_id} image path does not exist: {resolved}")

    return {
        "case_count": len(rows),
        "subject_count": len(subject_ids),
        "category_count": len(categories),
        "case_ids": sorted(seen_case_ids),
        "subjects": sorted(subject_ids),
        "categories": sorted(categories),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate a fixed reference evaluation JSONL file.")
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--root", default=".")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = validate_fixed_reference_eval(args.eval_set, root=args.root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
