from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROUTE_METHODS = [
    "v2_8a_static_a064_d080",
    "v2_8a_static_a048_d096",
    "v2_8a_static_a032_d112",
    "v2_8a_static_a016_d128",
    "v2_8a_static_a000_d144",
]

ROUTE_LABELS = {
    "v2_8a_static_a064_d080": "Route I",
    "v2_8a_static_a048_d096": "Route II",
    "v2_8a_static_a032_d112": "Route III",
    "v2_8a_static_a016_d128": "Route IV",
    "v2_8a_static_a000_d144": "Route V",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf"),
        Path(r"C:\Windows\Fonts\Times New Roman.ttf"),
        "DejaVuSerif.ttf",
    ]:
        if isinstance(candidate, Path) and not candidate.exists():
            continue
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["case_id"]] = row
    return rows


def _read_predictions(results_dir: Path) -> list[dict[str, str]]:
    path = results_dir / "v2_14_ablation_phase1_official_locked_hybrid_predictions.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _resolve(path: str | Path, project_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root / path


def _fit_resize_pad(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(size / width, size / height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        paste_at = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
        canvas.paste(image, paste_at)
        return canvas


def _column_positions(n_cols: int, cell: int, margin: int, gap_x: int, group_gap_after: set[int]) -> list[int]:
    positions = []
    x = margin
    for col in range(n_cols):
        positions.append(x)
        x += cell
        if col < n_cols - 1:
            x += gap_x * 2 if col in group_gap_after else gap_x
    return positions


def _draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill=(20, 20, 20)):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) / 2
    y = top + (bottom - top - text_h) / 2 - 1
    draw.text((x, y), text, font=font, fill=fill)


def _save_figure(image: Image.Image, stem: str, figures_dir: Path) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, kwargs in [
        (".png", {}),
        (".tiff", {"compression": "tiff_lzw"}),
        (".pdf", {}),
    ]:
        path = figures_dir / f"{stem}{suffix}"
        if suffix == ".pdf":
            image.save(path, "PDF", resolution=600)
        else:
            image.save(path, dpi=(600, 600), **kwargs)
        outputs[suffix.lstrip(".")] = str(path)
    return outputs


def create_route_diversity_figure(
    *, project_root: Path, results_dir: Path, figures_dir: Path
) -> dict[str, str]:
    eval_rows = _read_jsonl(project_root / "data/eval/sop_small_valid_fixed_test48_20260621.jsonl")
    predictions = [row for row in _read_predictions(results_dir) if row["run"] == "regen_test48"]
    pred_by_case = {row["case_id"]: row for row in predictions}

    cases = [
        "sop_small_valid_000_sop_11422",
        "sop_small_valid_001_sop_11704",
        "sop_small_valid_032_sop_15805",
        "sop_small_valid_020_sop_12978",
    ]
    headers = ["Ref. 1", "Ref. 2", *[ROUTE_LABELS[m] for m in ROUTE_METHODS], "Target"]
    cell = 384
    header_h = 72
    gap_x = 26
    gap_y = 46
    margin = 40
    x_positions = _column_positions(len(headers), cell, margin, gap_x, {1, 6})
    width = x_positions[-1] + cell + margin
    height = margin * 2 + header_h + len(cases) * cell + (len(cases) - 1) * gap_y
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(38)
    border_default = (218, 218, 218)
    border_selected = (27, 115, 73)

    for col, header in enumerate(headers):
        x = x_positions[col]
        _draw_centered_text(draw, (x, margin, x + cell, margin + header_h), header, font)

    for row_idx, case_id in enumerate(cases):
        row = eval_rows[case_id]
        prediction = pred_by_case[case_id]
        y = margin + header_h + row_idx * (cell + gap_y)
        paths: list[tuple[str, Path, bool]] = [
            ("Ref. 1", _resolve(row["multi_reference_images"][0], project_root), False),
            ("Ref. 2", _resolve(row["multi_reference_images"][1], project_root), False),
        ]
        generation_root = project_root / "outputs/v2_14_candidate_ranker_r13_regen_test48_generation_20260622"
        for method in ROUTE_METHODS:
            paths.append((method, generation_root / case_id / method / "000000.png", method == prediction["selected_method"]))
        paths.append(("Target", _resolve(row["target_image"], project_root), False))

        for col, (_label, path, selected) in enumerate(paths):
            x = x_positions[col]
            image = _fit_resize_pad(path, cell)
            canvas.paste(image, (x, y))
            color = border_selected if selected else border_default
            line_w = 8 if selected else 2
            for offset in range(line_w):
                draw.rectangle((x - offset, y - offset, x + cell - 1 + offset, y + cell - 1 + offset), outline=color)

    return _save_figure(
        canvas,
        "v2_14_ablation_figA_route_diversity_casewise_selection_4rows_20260624",
        figures_dir,
    )


def _sampling_method_path(project_root: Path, case_id: str, method: str) -> Path:
    if method == "v2_14_a5_same_a064_seed0":
        return (
            project_root
            / "outputs/v2_14_candidate_ranker_r13_regen_test48_generation_20260622"
            / case_id
            / "v2_8a_static_a064_d080"
            / "000000.png"
        )
    return project_root / "outputs/v2_14_ablation_phase2_test48_generation_20260624" / case_id / method / "000000.png"


def create_sampling_copy_risk_figure(
    *, project_root: Path, figures_dir: Path
) -> dict[str, str]:
    eval_rows = _read_jsonl(project_root / "data/eval/sop_small_valid_fixed_test48_20260621.jsonl")
    case_specs = [
        {
            "case_id": "sop_small_valid_004_sop_19647",
            "structured_method": "v2_8a_static_a000_d144",
            "sampling_method": "v2_14_a5_same_a064_seed3",
        },
        {
            "case_id": "sop_small_valid_018_sop_12224",
            "structured_method": "v2_8a_static_a032_d112",
            "sampling_method": "v2_14_a5_same_a064_seed3",
        },
        {
            "case_id": "sop_small_valid_032_sop_15805",
            "structured_method": "v2_8a_static_a000_d144",
            "sampling_method": "v2_14_a5_same_a064_seed3",
        },
        {
            "case_id": "sop_small_valid_035_sop_16188",
            "structured_method": "v2_8a_static_a016_d128",
            "sampling_method": "v2_14_a5_same_a064_seed4",
        },
    ]
    official_root = project_root / "outputs/v2_14_candidate_ranker_r13_regen_test48_generation_20260622"
    headers = ["Ref. 1", "Ref. 2", "Baseline", "Structured", "Same-route", "Target"]
    cell = 420
    header_h = 74
    gap_x = 32
    gap_y = 48
    margin = 42
    x_positions = _column_positions(len(headers), cell, margin, gap_x, {1, 4})
    width = x_positions[-1] + cell + margin
    height = margin * 2 + header_h + len(case_specs) * cell + (len(case_specs) - 1) * gap_y
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(40)
    border_default = (218, 218, 218)
    border_risk = (169, 58, 47)
    border_safe = (27, 115, 73)

    for col, header in enumerate(headers):
        x = x_positions[col]
        _draw_centered_text(draw, (x, margin, x + cell, margin + header_h), header, font)

    for row_idx, spec in enumerate(case_specs):
        case_id = spec["case_id"]
        row = eval_rows[case_id]
        paths = [
            (_resolve(row["multi_reference_images"][0], project_root), None),
            (_resolve(row["multi_reference_images"][1], project_root), None),
            (official_root / case_id / "v2_8a_static_a064_d080" / "000000.png", None),
            (official_root / case_id / spec["structured_method"] / "000000.png", "safe"),
            (_sampling_method_path(project_root, case_id, spec["sampling_method"]), "risk"),
            (_resolve(row["target_image"], project_root), None),
        ]
        y = margin + header_h + row_idx * (cell + gap_y)
        for col, (path, role) in enumerate(paths):
            x = x_positions[col]
            image = _fit_resize_pad(path, cell)
            canvas.paste(image, (x, y))
            if role == "risk":
                color, line_w = border_risk, 8
            elif role == "safe":
                color, line_w = border_safe, 8
            else:
                color, line_w = border_default, 2
            for offset in range(line_w):
                draw.rectangle((x - offset, y - offset, x + cell - 1 + offset, y + cell - 1 + offset), outline=color)

    return _save_figure(
        canvas,
        "v2_14_ablation_figB_sampling_copy_risk_4rows_20260624",
        figures_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the v2.14 ablation visual figures and their manifest."
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root containing data and generated model outputs.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing the locked hybrid predictions CSV.",
    )
    parser.add_argument(
        "--figures_dir",
        type=Path,
        required=True,
        help="Directory where figure files and the manifest will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = args.results_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    outputs = {
        "route_diversity": create_route_diversity_figure(
            project_root=project_root,
            results_dir=results_dir,
            figures_dir=figures_dir,
        ),
        "sampling_copy_risk": create_sampling_copy_risk_figure(
            project_root=project_root,
            figures_dir=figures_dir,
        ),
    }
    manifest = figures_dir / "v2_14_ablation_visual_figures_manifest_20260624.json"
    manifest.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
