from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUANT_DIR = PROJECT_ROOT / "outputs/quant_main_v2_14_20260626/phase1_trackI/r14_replay"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/quant_main_v2_14_20260626/manifests/phase0_protocol_manifest.json"

METHOD_LABELS = {
    "v2_8a_static_a064_d080": "Static 64/80",
    "v2_8a_static_a048_d096": "Static 48/96",
    "v2_8a_static_a032_d112": "Static 32/112",
    "v2_8a_static_a016_d128": "Static 16/128",
    "v2_8a_static_a000_d144": "Static 0/144",
}

PAPER_ROWS = [
    "Static 48/96",
    "Static 32/112",
    "Static 16/128",
    "Static 0/144",
    "Global-Best-Static",
    "CLIP-only",
    "Pure RefMax",
    "Always Guard",
    "Locked Hybrid",
    "Target Oracle",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["case_id"])] = row
    return rows


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for suffix, kwargs in [
        ("png", {"dpi": 450}),
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
    ]:
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        written[suffix] = str(path)
    plt.close(fig)
    return written


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _build_run_context(manifest: dict[str, Any]) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    metric_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    eval_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for run in manifest["runs"]:
        label = str(run["label"])
        for row in _read_csv(run["case_metrics"]):
            metric_by_key[(label, str(row["case_id"]), str(row["method"]))] = row
        for case_id, row in _read_jsonl(run["eval_jsonl"]).items():
            eval_by_key[(label, case_id)] = row
    return metric_by_key, eval_by_key


def _pick_case_rows(case_rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    heldout = [row for row in case_rows if row["role"] == "heldout"]
    high_gain = max(
        [row for row in heldout if _float(row, "dino_delta") > 0.10 and _float(row, "copy_ssim_delta") < 0.02],
        key=lambda row: _float(row, "dino_delta"),
    )
    used = {(high_gain["run"], high_gain["case_id"])}
    triggered_pool = [
        row
        for row in heldout
        if str(row.get("triggered")) == "True" and (row["run"], row["case_id"]) not in used
    ]
    triggered = max(
        triggered_pool or [row for row in heldout if str(row.get("triggered")) == "True"],
        key=lambda row: _float(row, "dino_delta"),
    )
    used.add((triggered["run"], triggered["case_id"]))
    failure_pool = [row for row in heldout if (row["run"], row["case_id"]) not in used]
    failure = min(failure_pool or heldout, key=lambda row: _float(row, "dino_delta"))
    selected: list[tuple[str, dict[str, str]]] = [
        ("Large gain", high_gain),
        ("Guard triggered", triggered),
        ("Failure / boundary", failure),
    ]
    deduped: list[tuple[str, dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    for label, row in selected:
        key = (row["run"], row["case_id"])
        if key not in seen:
            deduped.append((label, row))
            seen.add(key)
    return deduped


def _image(path: Path, size: int = 256) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        left = (size - image.width) // 2
        top = (size - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas


def _oracle_method(metric_by_key: dict[tuple[str, str, str], dict[str, str]], run: str, case_id: str, methods: list[str]) -> str:
    return max(methods, key=lambda method: _float(metric_by_key[(run, case_id, method)], "dino_sim_to_target"))


def create_qualitative_grid(
    quant_dir: Path,
    manifest: dict[str, Any],
    figures_dir: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    metric_by_key, eval_by_key = _build_run_context(manifest)
    methods = list(manifest["official_method"]["candidate_methods"])
    baseline_method = str(manifest["official_method"]["baseline_method"])
    case_rows = _read_csv(quant_dir / "case_level_predictions.csv")
    picked = _pick_case_rows(case_rows)

    columns = ["Ref. 1", "Ref. 2", "Target", "Baseline", "Pure RefMax", "Locked Hybrid", "Target Oracle"]
    fig, axes = plt.subplots(
        len(picked),
        len(columns),
        figsize=(7.0, 1.12 * len(picked)),
        gridspec_kw={"wspace": 0.02, "hspace": 0.04},
    )
    if len(picked) == 1:
        axes = axes[None, :]

    case_manifest_rows: list[dict[str, Any]] = []
    for row_idx, (case_type, row) in enumerate(picked):
        run = row["run"]
        case_id = row["case_id"]
        eval_row = eval_by_key[(run, case_id)]
        pure_method = row["primary_selected"]
        locked_method = row["selected_method"]
        oracle_method = _oracle_method(metric_by_key, run, case_id, methods)
        paths = [
            _resolve(eval_row["multi_reference_images"][0]),
            _resolve(eval_row["multi_reference_images"][1]),
            _resolve(eval_row["target_image"]),
            _resolve(metric_by_key[(run, case_id, baseline_method)]["image_path"]),
            _resolve(metric_by_key[(run, case_id, pure_method)]["image_path"]),
            _resolve(metric_by_key[(run, case_id, locked_method)]["image_path"]),
            _resolve(metric_by_key[(run, case_id, oracle_method)]["image_path"]),
        ]
        labels = [
            "ref1",
            "ref2",
            "target",
            baseline_method,
            pure_method,
            locked_method,
            oracle_method,
        ]
        for col_idx, path in enumerate(paths):
            ax = axes[row_idx, col_idx]
            ax.imshow(_image(path, 256))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=7.5, pad=3)
        case_manifest_rows.append(
            {
                "case_type": case_type,
                "run": run,
                "case_id": case_id,
                "category": row["category"],
                "dino_delta": _float(row, "dino_delta"),
                "copy_ssim_delta": _float(row, "copy_ssim_delta"),
                "pure_refmax_method": pure_method,
                "locked_hybrid_method": locked_method,
                "target_oracle_method": oracle_method,
                "triggered": row["triggered"],
                "panel_labels": labels,
            }
        )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.005)
    return _save(fig, figures_dir / "trackI_qualitative_case_grid"), case_manifest_rows


def _plot_method_delta(ax: plt.Axes, aggregate: list[dict[str, str]]) -> None:
    rows = [row for row in aggregate if row["comparison"] in PAPER_ROWS]
    rows = sorted(rows, key=lambda row: PAPER_ROWS.index(row["comparison"]))
    y = list(range(len(rows)))
    estimates = [_float(row, "dino_delta_mean") for row in rows]
    low = [_float(row, "ci95_low") for row in rows]
    high = [_float(row, "ci95_high") for row in rows]
    colors = ["#B9B9B9"] * len(rows)
    for idx, row in enumerate(rows):
        if row["comparison"] == "Locked Hybrid":
            colors[idx] = "#2B7A5B"
        elif row["comparison"] in {"Pure RefMax", "Target Oracle"}:
            colors[idx] = "#6D8FB3"
        elif row["comparison"] in {"Always Guard"}:
            colors[idx] = "#8F7AAE"
    xerr = [[est - lo for est, lo in zip(estimates, low)], [hi - est for est, hi in zip(estimates, high)]]
    ax.barh(y, estimates, color=colors, edgecolor="none", height=0.65, alpha=0.88)
    ax.errorbar(estimates, y, xerr=xerr, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    ax.axvline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([row["comparison"] for row in rows], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("DINO target similarity delta vs Static 64/80")
    ax.set_title("A  Method effect with 95% CI", loc="left", fontweight="bold")


def _plot_tradeoff(ax: plt.Axes, case_rows: list[dict[str, str]]) -> None:
    xs = [_float(row, "copy_ssim_delta") for row in case_rows]
    ys = [_float(row, "dino_delta") for row in case_rows]
    triggered = [str(row.get("triggered")) == "True" for row in case_rows]
    colors = ["#2B7A5B" if flag else "#A7A7A7" for flag in triggered]
    ax.scatter(xs, ys, s=18, c=colors, edgecolor="white", linewidth=0.3, alpha=0.82)
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.axvline(0.015, color="#B35A4B", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Copy SSIM delta vs Static 64/80")
    ax.set_ylabel("DINO target delta")
    ax.set_title("B  Case-level gain and copy-risk tradeoff", loc="left", fontweight="bold")
    ax.text(0.017, max(ys) * 0.82, "copy-risk\nthreshold", color="#8F3E32", fontsize=6)


def _tau_value(row: dict[str, str]) -> float:
    return 0.225 if str(row["tau_dino"]) == "+inf" else float(row["tau_dino"])


def _plot_threshold(ax: plt.Axes, quant_dir: Path) -> None:
    tables_dir = quant_dir / "tables"
    validation = sorted(_read_csv(tables_dir / "threshold_validation_summary.csv"), key=_tau_value)
    heldout = sorted(_read_csv(tables_dir / "threshold_heldout_summary.csv"), key=_tau_value)
    for rows, label, color in [
        (validation, "validation", "#6D8FB3"),
        (heldout, "heldout", "#B35A4B"),
    ]:
        xs = [_tau_value(row) for row in rows]
        ys = [_float(row, "dino_delta_mean") if "dino_delta_mean" in row else _float(row, "estimate") for row in rows]
        lo = [_float(row, "ci95_low") for row in rows]
        hi = [_float(row, "ci95_high") for row in rows]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=label, color=color)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.axvline(0.10, color="#2B7A5B", linewidth=0.8, linestyle=":")
    ax.set_xlabel("DINO-copy relative threshold")
    ax.set_ylabel("DINO target delta")
    ax.set_title("C  Guard threshold sweep", loc="left", fontweight="bold")
    ax.legend(loc="lower left", fontsize=6)


def _plot_category(ax: plt.Axes, quant_dir: Path) -> None:
    rows = sorted(_read_csv(quant_dir / "category_breakdown.csv"), key=lambda row: _float(row, "estimate"))
    y = list(range(len(rows)))
    estimates = [_float(row, "estimate") for row in rows]
    low = [_float(row, "ci95_low") for row in rows]
    high = [_float(row, "ci95_high") for row in rows]
    xerr = [[est - lo for est, lo in zip(estimates, low)], [hi - est for est, hi in zip(estimates, high)]]
    ax.errorbar(estimates, y, xerr=xerr, fmt="o", markersize=3, color="#2B7A5B", ecolor="#A8C8B8", capsize=2)
    ax.axvline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([row["variant"].replace("sop_super_", "cat ") for row in rows], fontsize=6)
    ax.set_xlabel("DINO target delta")
    ax.set_title("D  Category-level consistency", loc="left", fontweight="bold")


def create_statistics_panel(quant_dir: Path, figures_dir: Path) -> dict[str, str]:
    aggregate = _read_csv(quant_dir / "aggregate_table.csv")
    case_rows = _read_csv(quant_dir / "case_level_predictions.csv")
    fig = plt.figure(figsize=(7.1, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.08, 0.92], wspace=0.36, hspace=0.52)
    _plot_method_delta(fig.add_subplot(grid[:, 0]), aggregate)
    _plot_tradeoff(fig.add_subplot(grid[0, 1]), case_rows)
    _plot_threshold(fig.add_subplot(grid[1, 1]), quant_dir)
    fig.suptitle("Track I same-backbone mechanism validation", fontsize=9, y=0.995)
    return _save(fig, figures_dir / "trackI_statistics_panel")


def create_category_breakdown(quant_dir: Path, figures_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    _plot_category(ax, quant_dir)
    return _save(fig, figures_dir / "trackI_category_breakdown")


def create_figures(quant_dir: Path, manifest_path: Path, figures_dir: Path) -> dict[str, Any]:
    _setup_style()
    manifest = _load_manifest(manifest_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {
        "statistics_panel": create_statistics_panel(quant_dir, figures_dir),
        "category_breakdown": create_category_breakdown(quant_dir, figures_dir),
    }
    qualitative_paths, case_manifest = create_qualitative_grid(quant_dir, manifest, figures_dir)
    written["qualitative_case_grid"] = qualitative_paths
    written["qualitative_cases"] = case_manifest
    manifest_out = figures_dir / "trackI_figures_manifest.json"
    manifest_out.write_text(json.dumps(written, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["manifest_json"] = str(manifest_out)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Track I quantitative and qualitative figures.")
    parser.add_argument("--quant_dir", default=str(DEFAULT_QUANT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--figures_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quant_dir = Path(args.quant_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else quant_dir / "figures"
    written = create_figures(quant_dir, Path(args.manifest), figures_dir)
    print(json.dumps(written, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
