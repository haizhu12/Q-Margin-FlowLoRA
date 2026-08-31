from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class MethodSpec:
    name: str
    root: Path
    method_dir: str


def parse_method_spec(values: list[str] | tuple[str, ...]) -> MethodSpec:
    if len(values) != 3:
        raise ValueError("--method requires exactly three values: <name> <output_root> <method_dir>.")
    return MethodSpec(name=str(values[0]), root=Path(values[1]), method_dir=str(values[2]))


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "case_id" not in row:
                raise ValueError(f"Missing case_id in {path} line {line_no}.")
            rows.append(row)
    if not rows:
        raise ValueError(f"No eval cases found in {path}.")
    return rows


def _resolve_path(path: str | Path, project_root: str | Path = PROJECT_ROOT) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path(project_root) / value


def _method_image_path(case_id: str, spec: MethodSpec, project_root: str | Path = PROJECT_ROOT) -> Path:
    root = _resolve_path(spec.root, project_root)
    path = root / case_id / spec.method_dir / "000000.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing image for method_dir={spec.method_dir!r}, case_id={case_id!r}: {path}")
    return path


def _load_rgb_array(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def _pixel_mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).mean() * 255.0)


def _global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1, 3)
    right = right.reshape(-1, 3)
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(3):
        x = left[:, channel]
        y = right[:, channel]
        mux = float(x.mean())
        muy = float(y.mean())
        vx = float(((x - mux) ** 2).mean())
        vy = float(((y - muy) ** 2).mean())
        cov = float(((x - mux) * (y - muy)).mean())
        numerator = (2.0 * mux * muy + c1) * (2.0 * cov + c2)
        denominator = (mux * mux + muy * muy + c1) * (vx + vy + c2)
        scores.append(numerator / denominator if denominator else 1.0)
    return float(np.clip(np.mean(scores), -1.0, 1.0))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def _as_feature_tensor(value):
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    return value


def _rgb_stats_feature(path: Path) -> np.ndarray:
    arr = _load_rgb_array(path)
    flat = arr.reshape(-1, 3)
    feature = np.concatenate([flat.mean(axis=0), flat.std(axis=0)], axis=0).astype(np.float32)
    norm = np.linalg.norm(feature)
    return feature / max(float(norm), 1e-12)


def _encode_dino_features(
    paths: list[Path],
    model_id: str,
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> dict[Path, np.ndarray]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device_obj = torch.device(device if str(device) == "cpu" or torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only).to(device_obj)
    model.eval()

    features: dict[Path, np.ndarray] = {}
    unique_paths = list(dict.fromkeys(paths))
    for start in range(0, len(unique_paths), int(batch_size)):
        batch_paths = unique_paths[start : start + int(batch_size)]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device_obj) for key, value in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("DINO model did not return last_hidden_state.")
        batch = hidden[:, 0, :].detach().float().cpu().numpy()
        for path, vector in zip(batch_paths, batch):
            vector = vector.astype(np.float32)
            features[path] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return features


def _collect_paths(cases: list[dict], methods: list[MethodSpec], project_root: Path) -> list[Path]:
    paths = []
    for case in cases:
        paths.append(_resolve_path(case["target_image"], project_root))
        paths.extend(_resolve_path(path, project_root) for path in case.get("single_reference_images", []))
        paths.extend(_resolve_path(path, project_root) for path in case.get("multi_reference_images", []))
        paths.extend(_method_image_path(case["case_id"], spec, project_root) for spec in methods)
    return list(dict.fromkeys(paths))


def _build_feature_lookup(
    cases: list[dict],
    methods: list[MethodSpec],
    project_root: Path,
    feature_backend: str,
    feature_model_id: str,
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> dict[Path, np.ndarray]:
    if feature_backend == "none":
        return {}
    paths = _collect_paths(cases, methods, project_root)
    if feature_backend == "rgb_stats":
        return {path: _rgb_stats_feature(path) for path in paths}
    if feature_backend == "dino":
        return _encode_dino_features(paths, feature_model_id, device, batch_size, local_files_only)
    raise ValueError(f"Unsupported feature_backend={feature_backend!r}.")


def _encode_clip_prompt_scores(
    cases: list[dict],
    methods: list[MethodSpec],
    project_root: Path,
    model_id: str,
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> dict[tuple[str, str], float]:
    import torch
    from transformers import AutoModel, AutoProcessor

    device_obj = torch.device(device if str(device) == "cpu" or torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only).to(device_obj)
    model.eval()
    if not hasattr(model, "get_image_features") or not hasattr(model, "get_text_features"):
        raise RuntimeError(f"Model {model_id!r} does not expose CLIP image/text feature methods.")

    scores: dict[tuple[str, str], float] = {}
    for case in cases:
        prompt = case.get("prompt", "")
        with torch.no_grad():
            text_inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
            text_inputs = {key: value.to(device_obj) for key, value in text_inputs.items()}
            text_features = _as_feature_tensor(model.get_text_features(**text_inputs))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        method_paths = [(spec.name, _method_image_path(case["case_id"], spec, project_root)) for spec in methods]
        for start in range(0, len(method_paths), int(batch_size)):
            chunk = method_paths[start : start + int(batch_size)]
            images = []
            for _, path in chunk:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {key: value.to(device_obj) for key, value in image_inputs.items()}
            with torch.no_grad():
                image_features = _as_feature_tensor(model.get_image_features(**image_inputs))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                sims = (image_features @ text_features.T).squeeze(-1).detach().float().cpu().tolist()
            for (method_name, _), score in zip(chunk, sims):
                scores[(case["case_id"], method_name)] = float(score)
    return scores


def _mean(values: Iterable[float]) -> float:
    kept = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not kept:
        return math.nan
    return float(sum(kept) / len(kept))


def _format_value(value):
    if value is None:
        return ""
    try:
        if math.isnan(float(value)):
            return ""
    except (TypeError, ValueError):
        return value
    return f"{float(value):.6f}"


def _feature_sim(feature_lookup: dict[Path, np.ndarray], generated: Path, references: list[Path]) -> float:
    if not feature_lookup or generated not in feature_lookup or not references:
        return math.nan
    return _mean(_cosine(feature_lookup[generated], feature_lookup[path]) for path in references if path in feature_lookup)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in fieldnames})


def evaluate_fixed_reference_metrics(
    eval_set_path: str | Path,
    methods: list[MethodSpec],
    output_dir: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    feature_backend: str = "none",
    feature_model_id: str = "facebook/dinov2-base",
    prompt_backend: str = "none",
    clip_model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    batch_size: int = 16,
    local_files_only: bool = False,
) -> dict:
    if not methods:
        raise ValueError("At least one method must be provided.")
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    cases = _load_jsonl(eval_set_path)

    feature_lookup = _build_feature_lookup(
        cases=cases,
        methods=methods,
        project_root=project_root,
        feature_backend=feature_backend,
        feature_model_id=feature_model_id,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )
    if prompt_backend == "none":
        prompt_scores = {}
    elif prompt_backend == "clip":
        prompt_scores = _encode_clip_prompt_scores(
            cases=cases,
            methods=methods,
            project_root=project_root,
            model_id=clip_model_id,
            device=device,
            batch_size=batch_size,
            local_files_only=local_files_only,
        )
    else:
        raise ValueError(f"Unsupported prompt_backend={prompt_backend!r}.")

    case_rows: list[dict] = []
    for case in cases:
        case_id = case["case_id"]
        target_path = _resolve_path(case["target_image"], project_root)
        single_refs = [_resolve_path(path, project_root) for path in case.get("single_reference_images", [])]
        multi_refs = [_resolve_path(path, project_root) for path in case.get("multi_reference_images", [])]
        for spec in methods:
            generated_path = _method_image_path(case_id, spec, project_root)
            generated_image = _load_rgb_array(generated_path)
            size = (generated_image.shape[1], generated_image.shape[0])
            target_image = _load_rgb_array(target_path, size=size)
            single_images = [_load_rgb_array(path, size=size) for path in single_refs]
            multi_images = [_load_rgb_array(path, size=size) for path in multi_refs]

            pixel_to_single = [_pixel_mae(generated_image, image) for image in single_images]
            pixel_to_multi = [_pixel_mae(generated_image, image) for image in multi_images]
            ssim_to_single = [_global_ssim(generated_image, image) for image in single_images]
            ssim_to_multi = [_global_ssim(generated_image, image) for image in multi_images]

            row = {
                "case_id": case_id,
                "subject_id": case.get("subject_id", ""),
                "category": case.get("category", ""),
                "method": spec.name,
                "prompt": case.get("prompt", ""),
                "image_path": str(generated_path),
                "pixel_mae_to_target": _pixel_mae(generated_image, target_image),
                "pixel_mae_to_single_ref_mean": _mean(pixel_to_single),
                "pixel_mae_to_multi_ref_mean": _mean(pixel_to_multi),
                "ref_copy_pixel_mae_min": min(pixel_to_multi) if pixel_to_multi else math.nan,
                "ssim_to_target": _global_ssim(generated_image, target_image),
                "ssim_to_single_ref_mean": _mean(ssim_to_single),
                "ssim_to_multi_ref_mean": _mean(ssim_to_multi),
                "ref_copy_ssim_max": max(ssim_to_multi) if ssim_to_multi else math.nan,
                "dino_sim_to_target": _feature_sim(feature_lookup, generated_path, [target_path]),
                "dino_sim_to_single_ref_mean": _feature_sim(feature_lookup, generated_path, single_refs),
                "dino_sim_to_multi_ref_mean": _feature_sim(feature_lookup, generated_path, multi_refs),
                "dino_ref_copy_sim_max": max(
                    [
                        _cosine(feature_lookup[generated_path], feature_lookup[path])
                        for path in multi_refs
                        if feature_lookup and generated_path in feature_lookup and path in feature_lookup
                    ],
                    default=math.nan,
                ),
                "clip_text_image_sim": prompt_scores.get((case_id, spec.name), math.nan),
            }
            case_rows.append(row)

    numeric_keys = [
        "pixel_mae_to_target",
        "pixel_mae_to_single_ref_mean",
        "pixel_mae_to_multi_ref_mean",
        "ref_copy_pixel_mae_min",
        "ssim_to_target",
        "ssim_to_single_ref_mean",
        "ssim_to_multi_ref_mean",
        "ref_copy_ssim_max",
        "dino_sim_to_target",
        "dino_sim_to_single_ref_mean",
        "dino_sim_to_multi_ref_mean",
        "dino_ref_copy_sim_max",
        "clip_text_image_sim",
    ]
    summary_rows = []
    for spec in methods:
        rows = [row for row in case_rows if row["method"] == spec.name]
        summary = {"method": spec.name, "case_count": len(rows)}
        for key in numeric_keys:
            summary[f"{key}_mean"] = _mean(row[key] for row in rows)
        summary_rows.append(summary)

    case_fields = [
        "case_id",
        "subject_id",
        "category",
        "method",
        "prompt",
        "image_path",
        *numeric_keys,
    ]
    summary_fields = ["method", "case_count", *[f"{key}_mean" for key in numeric_keys]]

    _write_csv(output_dir / "case_metrics.csv", case_rows, case_fields)
    _write_csv(output_dir / "method_summary.csv", summary_rows, summary_fields)

    result = {
        "eval_set": str(eval_set_path),
        "output_dir": str(output_dir),
        "case_count": len(cases),
        "method_count": len(methods),
        "feature_backend": feature_backend,
        "feature_model_id": feature_model_id if feature_backend != "none" else None,
        "prompt_backend": prompt_backend,
        "clip_model_id": clip_model_id if prompt_backend == "clip" else None,
        "case_metrics": str(output_dir / "case_metrics.csv"),
        "method_summary": str(output_dir / "method_summary.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate fixed reference-generation outputs with compact metrics.")
    parser.add_argument("--eval_set", required=True, help="Fixed eval JSONL path.")
    parser.add_argument(
        "--method",
        nargs=3,
        action="append",
        metavar=("NAME", "OUTPUT_ROOT", "METHOD_DIR"),
        required=True,
        help="Method spec. Example: --method native_text outputs/root native_text_only",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--feature_backend", choices=["none", "rgb_stats", "dino"], default="none")
    parser.add_argument("--feature_model_id", default="facebook/dinov2-base")
    parser.add_argument("--prompt_backend", choices=["none", "clip"], default="none")
    parser.add_argument("--clip_model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    result = evaluate_fixed_reference_metrics(
        eval_set_path=args.eval_set,
        methods=[parse_method_spec(values) for values in args.method],
        output_dir=args.output_dir,
        project_root=args.project_root,
        feature_backend=args.feature_backend,
        feature_model_id=args.feature_model_id,
        prompt_backend=args.prompt_backend,
        clip_model_id=args.clip_model_id,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
