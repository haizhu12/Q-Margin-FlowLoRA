from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_v2_14_ablation_phase1 import (  # noqa: E402
    DEFAULT_BASELINE,
    analyze_phase1_offline,
    read_csv,
    summarize_prediction_rows,
    write_csv,
)
from scripts.analyze_v2_14_compressibility import DEFAULT_METHODS  # noqa: E402


STATIC_LABELS = {
    "v2_8a_static_a064_d080": "Static 64/80",
    "v2_8a_static_a048_d096": "Static 48/96",
    "v2_8a_static_a032_d112": "Static 32/112",
    "v2_8a_static_a016_d128": "Static 16/128",
    "v2_8a_static_a000_d144": "Static 0/144",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _table_by_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("variant")) == variant:
            return row
    return None


def _paper_row(comparison: str, source_table: str, source: dict[str, Any], *, scope: str) -> dict[str, Any]:
    keys = (
        "clip_delta_mean",
        "ci90_low",
        "resampling_unit",
        "n_bootstrap",
        "copy_risk_rate_ci95_low",
        "n_subjects",
        "trigger_count",
        "n_cases",
        "false_intervention_rate",
        "dino_delta_median",
        "ci95_high",
        "baseline_variant",
        "ci90_high",
        "win_rate_ci95_high",
        "copy_ssim_delta_ci95_high",
        "copy_ssim_delta_ci95_low",
        "interval_method",
        "copy_risk_rate_ci95_high",
        "false_intervention_count",
        "win_rate",
        "trigger_rate",
        "copy_risk_rate",
        "clip_delta_ci95_high",
        "ci95_low",
        "copy_risk_count",
        "bootstrap_seed",
        "copy_ssim_delta_mean",
        "ci_level",
        "dino_delta_trimmed_mean_10pct",
        "selected_method_counts",
        "win_rate_ci95_low",
        "clip_delta_ci95_low",
    )
    row = {
        "comparison": comparison,
        "scope": scope,
        "source_table": source_table,
        "source_variant": source.get("variant", ""),
        "dino_delta_mean": source.get("estimate", ""),
    }
    for key in keys:
        row[key] = source.get(key, "")
    if "selected_on_validation_method" in source:
        row["selected_on_validation_method"] = source.get("selected_on_validation_method", "")
    return row


def _validate_manifest_runs(manifest: dict[str, Any]) -> dict[str, Any]:
    candidate_methods = set(manifest["official_method"].get("candidate_methods", DEFAULT_METHODS))
    missing_paths: list[str] = []
    run_checks: list[dict[str, Any]] = []
    bad_candidate_case_count = 0
    case_count_mismatch_count = 0

    for run in manifest.get("runs", []):
        for key in ("eval_jsonl", "case_metrics"):
            if key in run and not Path(run[key]).exists():
                missing_paths.append(str(run[key]))
        metrics_path = Path(run["case_metrics"])
        if not metrics_path.exists():
            continue
        rows = read_csv(metrics_path)
        methods_by_case: dict[str, set[str]] = {}
        for row in rows:
            method = str(row.get("method", ""))
            if method not in candidate_methods:
                continue
            methods_by_case.setdefault(str(row.get("case_id", "")), set()).add(method)
        bad_cases = [
            case_id
            for case_id, methods in methods_by_case.items()
            if methods != candidate_methods
        ]
        expected_cases = run.get("case_count")
        case_count_mismatch = expected_cases is not None and int(expected_cases) != len(methods_by_case)
        if case_count_mismatch:
            case_count_mismatch_count += 1
        bad_candidate_case_count += len(bad_cases)
        run_checks.append(
            {
                "label": run.get("label", ""),
                "role": run.get("role", ""),
                "case_metrics": str(metrics_path),
                "expected_case_count": expected_cases,
                "candidate_case_count": len(methods_by_case),
                "metric_row_count": len(rows),
                "bad_candidate_case_count": len(bad_cases),
                "bad_candidate_case_examples": bad_cases[:5],
                "case_count_mismatch": case_count_mismatch,
            }
        )

    return {
        "missing_paths": sorted(missing_paths),
        "run_checks": run_checks,
        "bad_candidate_case_count": bad_candidate_case_count,
        "case_count_mismatch_count": case_count_mismatch_count,
    }


def _build_runs(manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (str(run["label"]), str(run.get("role", "heldout")), str(run["case_metrics"]))
        for run in manifest.get("runs", [])
    ]


def _build_aggregate_table(result: dict[str, Any], *, bootstrap_iterations: int, seed: int) -> list[dict[str, Any]]:
    tables = result["tables"]
    rows: list[dict[str, Any]] = []

    for source in tables.get("static_method_summary", []):
        method = str(source.get("method") or source.get("variant"))
        rows.append(
            _paper_row(
                STATIC_LABELS.get(method, f"Static {method}"),
                "static_method_summary",
                source,
                scope="all_replay",
            )
        )

    global_static = _table_by_variant(
        tables.get("global_best_static_summary", []),
        "global_best_static_on_heldout",
    )
    if global_static:
        rows.append(_paper_row("Global-Best-Static", "global_best_static_summary", global_static, scope="heldout"))

    selector_rows = tables.get("selector_signal_summary", [])
    selector_specs = [
        ("Random-Split", "E1-0__random_candidate"),
        ("CLIP-only", "E1-1__clip_t_only"),
    ]
    for comparison, variant in selector_specs:
        source = _table_by_variant(selector_rows, variant)
        if source:
            rows.append(_paper_row(comparison, "selector_signal_summary", source, scope="all_replay"))

    guard_rows = tables.get("guard_strategy_summary", [])
    guard_specs = [
        ("Pure RefMax", "E3-0__pure_refmax_no_guard"),
        ("Always Guard", "E3-1__always_guarded_score"),
        ("Locked Hybrid", "E3-7__trigger_then_guard_rerank_official"),
    ]
    for comparison, variant in guard_specs:
        source = _table_by_variant(guard_rows, variant)
        if source:
            rows.append(_paper_row(comparison, "guard_strategy_summary", source, scope="all_replay"))

    oracle_rows = result.get("predictions", {}).get("target_oracle_full", [])
    if oracle_rows:
        oracle_summary = summarize_prediction_rows(
            oracle_rows,
            table="quant_trackI_aggregate",
            variant="target_oracle_full",
            baseline_variant=str(result["metadata"]["baseline_method"]),
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        rows.append(_paper_row("Target Oracle", "predictions", oracle_summary, scope="all_replay"))

    return rows


def _environment_text() -> str:
    return "\n".join(
        [
            f"python: {sys.version.split()[0]}",
            f"python_executable: {sys.executable}",
            f"platform: {platform.platform()}",
            f"cwd: {Path.cwd()}",
            f"project_root: {PROJECT_ROOT}",
        ]
    ) + "\n"


def _write_result_md(path: Path, aggregate_rows: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    locked = next((row for row in aggregate_rows if row["comparison"] == "Locked Hybrid"), {})
    pure = next((row for row in aggregate_rows if row["comparison"] == "Pure RefMax"), {})
    oracle = next((row for row in aggregate_rows if row["comparison"] == "Target Oracle"), {})

    def fmt(row: dict[str, Any]) -> str:
        if not row:
            return ""
        return (
            f"{float(row['dino_delta_mean']):+.6f} "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}]"
        )

    lines = [
        "# Q-Margin v2.14 Quant Track I Phase 1A R14 Replay",
        "",
        "This run replays existing R14 metric caches only. No new images were generated.",
        "",
        "## Key Rows",
        "",
        "| Comparison | DINO target delta mean with 95% CI | Copy-risk |",
        "|---|---:|---:|",
        f"| Locked Hybrid | {fmt(locked)} | {locked.get('copy_risk_count', '')}/{locked.get('n_cases', '')} |",
        f"| Pure RefMax | {fmt(pure)} | {pure.get('copy_risk_count', '')}/{pure.get('n_cases', '')} |",
        f"| Target Oracle | {fmt(oracle)} | {oracle.get('copy_risk_count', '')}/{oracle.get('n_cases', '')} |",
        "",
        "## Validation",
        "",
        f"- Missing paths: {len(validation['missing_paths'])}",
        f"- Bad candidate cases: {validation['bad_candidate_case_count']}",
        f"- Case-count mismatches: {validation['case_count_mismatch_count']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quant_tracki_outputs(
    result: dict[str, Any],
    *,
    output_dir: str | Path,
    manifest_path: str | Path,
    validation: dict[str, Any],
    bootstrap_iterations: int,
    seed: int,
    command_line: str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    case_lists_dir = output_dir / "case_lists"
    tables_dir.mkdir(parents=True, exist_ok=True)
    case_lists_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = _build_aggregate_table(result, bootstrap_iterations=bootstrap_iterations, seed=seed)
    official_rows = result["predictions"].get(
        "E3-7__trigger_then_guard_rerank_official",
        result["predictions"].get("locked_hybrid_full", []),
    )

    written: dict[str, str] = {}
    root_csvs = {
        "aggregate_table_csv": (output_dir / "aggregate_table.csv", aggregate_rows),
        "case_level_predictions_csv": (output_dir / "case_level_predictions.csv", official_rows),
        "split_breakdown_csv": (output_dir / "split_breakdown.csv", result["tables"].get("split_breakdown_summary", [])),
        "category_breakdown_csv": (output_dir / "category_breakdown.csv", result["tables"].get("category_breakdown_summary", [])),
        "guard_breakdown_csv": (output_dir / "guard_breakdown.csv", result["tables"].get("guard_trigger_breakdown_summary", [])),
        "copy_risk_cases_csv": (output_dir / "copy_risk_cases.csv", result["case_lists"].get("copy_risk_cases", [])),
        "worst_cases_csv": (output_dir / "worst_cases.csv", result["case_lists"].get("worst_cases", [])),
    }
    for key, (path, rows) in root_csvs.items():
        write_csv(path, rows)
        written[key] = str(path)

    for table_name, rows in result["tables"].items():
        path = tables_dir / f"{table_name}.csv"
        write_csv(path, rows)
        written[f"table_{table_name}_csv"] = str(path)

    for list_name, rows in result["case_lists"].items():
        path = case_lists_dir / f"{list_name}.csv"
        write_csv(path, rows)
        written[f"case_list_{list_name}_csv"] = str(path)

    summary = {
        "metadata": {
            **result["metadata"],
            "analysis": "quant_trackI_phase1a_r14_replay",
            "source_manifest": str(Path(manifest_path)),
        },
        "validation": validation,
        "aggregate_table": aggregate_rows,
        "tables": result["tables"],
        "case_lists": result["case_lists"],
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    written["summary_json"] = str(summary_path)

    result_md = output_dir / "result.md"
    _write_result_md(result_md, aggregate_rows, validation)
    written["result_md"] = str(result_md)

    run_command = output_dir / "run_command.txt"
    run_command.write_text(command_line.rstrip() + "\n", encoding="utf-8")
    written["run_command_txt"] = str(run_command)

    environment = output_dir / "environment.txt"
    environment.write_text(_environment_text(), encoding="utf-8")
    written["environment_txt"] = str(environment)

    output_manifest = output_dir / "output_manifest.json"
    _write_json(output_manifest, written)
    written["output_manifest_json"] = str(output_manifest)
    return written


def run_quant_tracki_from_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    bootstrap_iterations: int | None = None,
    seed: int | None = None,
    random_seed: int | None = None,
    command_line: str | None = None,
) -> dict[str, str]:
    manifest = _load_json(manifest_path)
    validation = _validate_manifest_runs(manifest)
    if validation["missing_paths"]:
        raise FileNotFoundError(f"Missing input paths: {validation['missing_paths']}")
    if validation["bad_candidate_case_count"]:
        raise ValueError(f"Some cases do not contain the complete candidate method set: {validation}")
    if validation["case_count_mismatch_count"]:
        raise ValueError(f"Manifest case_count does not match metric cache case counts: {validation}")

    official = manifest.get("official_method", {})
    statistics = manifest.get("statistics", {})
    resolved_output_dir = output_dir or manifest.get("outputs", {}).get("phase1a_output_dir")
    if not resolved_output_dir:
        raise ValueError("output_dir is required when manifest.outputs.phase1a_output_dir is absent.")

    resolved_bootstrap_iterations = int(bootstrap_iterations or statistics.get("bootstrap_iterations", 10000))
    resolved_seed = int(seed or statistics.get("bootstrap_seed", 20260624))
    resolved_random_seed = int(random_seed or resolved_seed)
    baseline_method = str(official.get("baseline_method", DEFAULT_BASELINE))
    candidate_methods = list(official.get("candidate_methods") or DEFAULT_METHODS)

    result = analyze_phase1_offline(
        runs=_build_runs(manifest),
        baseline_method=baseline_method,
        candidate_methods=candidate_methods,
        bootstrap_iterations=resolved_bootstrap_iterations,
        seed=resolved_seed,
        random_seed=resolved_random_seed,
    )
    return write_quant_tracki_outputs(
        result,
        output_dir=resolved_output_dir,
        manifest_path=manifest_path,
        validation=validation,
        bootstrap_iterations=resolved_bootstrap_iterations,
        seed=resolved_seed,
        command_line=command_line or f"{sys.executable} {' '.join(sys.argv)}",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2.14 quantitative Track I Phase 1A replay from a frozen manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bootstrap_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = parse_args(argv)
    command_line = f"{sys.executable} {' '.join(sys.argv)}"
    written = run_quant_tracki_from_manifest(
        args.manifest,
        output_dir=args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        random_seed=args.random_seed,
        command_line=command_line,
    )
    print(json.dumps(written, ensure_ascii=False, indent=2))
    return written


if __name__ == "__main__":
    main()
