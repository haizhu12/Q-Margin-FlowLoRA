# Reproducing Q-Margin

This guide runs the frozen Q-Margin generation and target-free selection protocol from local model and data assets. Commands assume PowerShell, Python 3.11+, the full CUDA environment from the root README, and execution from the repository root.

Formal generated images, metric tables, figures, and evidence bundles are local-only outputs. They are ignored by Git and are not claimed to be versioned with this repository.

## 1. Freeze the protocol inputs

The protocol uses:

- frozen `FLUX.2-klein-base-4B` in `bf16`, with no checkpoint or adapter;
- `512 x 512` output, 30 inference steps, and guidance scale `1.0`;
- seed `42` stored in every evaluation JSONL record;
- RGB references center-cropped/resized to `256 x 256` before native VAE encoding;
- `K=144` reference tokens for each of five `(anchor, detail)` routes: `(64,80)`, `(48,96)`, `(32,112)`, `(16,128)`, and `(0,144)`;
- RefMax selection followed by the locked DINO-copy Guard at delta `0.10`, relative to Static `(64,80)`.

The generator has no `--seed` option: it reads `seed` from each manifest record. Use a local fixed-evaluation manifest whose records all contain `"seed": 42`; do not treat other example manifest seeds as the frozen paper setting.

## 2. Provide model, data, and a fixed-reference manifest

The `fixed_reference` schema requires these fields in every JSONL record:

```text
case_id
subject_id
category
prompt
seed
target_image
single_reference_images
multi_reference_images
```

`single_reference_images` must contain exactly one path and `multi_reference_images` at least two paths. Extra metadata fields are permitted. Target images are evaluation-only and are not read by RefMax or Guard.

For generation-only deployment, `single_reference_no_target` instead requires `case_id`, `prompt`, and exactly one entry in `ref_paths` (or `single_reference_images`). It does not require a target, subject, category, or multi-reference list. That schema cannot be passed to `evaluate_fixed_reference_metrics.py`, which requires the fixed-reference target and reference fields.

Set portable local paths once:

```powershell
$project = (Get-Location)
$evalSet = [System.IO.Path]::GetFullPath($env:QMARGIN_EVAL_SET)
$candidateRoot = $project / "outputs/reproduction/candidates"
$metricRoot = $project / "outputs/reproduction/metrics"
$selectionRoot = $project / "outputs/reproduction/refmax_guard"
```

Set `QMARGIN_EVAL_SET` to your seed-42 JSONL before running the block. Relative paths inside that JSONL resolve against `--root` for generation and `--project_root` for metric evaluation, so the commands deliberately pass `$project` to both.

Validate the manifest and local images first:

```powershell
.venv\Scripts\python.exe scripts/validate_fixed_reference_eval.py `
  --eval_set $evalSet `
  --root $project
```

## 3. Generate all five equal-budget routes

```powershell
.venv\Scripts\python.exe scripts/run_v2_8_ctnr_oracle.py `
  --config ($project / "configs/qm_ref_sop_small_valid_native_vae_native_coreset_k144_eval.yaml") `
  --eval_set $evalSet `
  --root $project `
  --output_root $candidateRoot `
  --height 512 `
  --width 512 `
  --num_inference_steps 30 `
  --guidance_scale 1.0 `
  --methods v2_8a_static_a064_d080 v2_8a_static_a048_d096 v2_8a_static_a032_d112 v2_8a_static_a016_d128 v2_8a_static_a000_d144 `
  --device cuda
```

Do not pass `--checkpoint`; the frozen runner rejects checkpoints. Expected local outputs are `$candidateRoot/summary.json` and one `CASE_ID/METHOD/000000.png` for every case/route pair.

## 4. Calculate fixed-reference metrics

```powershell
.venv\Scripts\python.exe scripts/evaluate_fixed_reference_metrics.py `
  --eval_set $evalSet `
  --project_root $project `
  --method v2_8a_static_a064_d080 $candidateRoot v2_8a_static_a064_d080 `
  --method v2_8a_static_a048_d096 $candidateRoot v2_8a_static_a048_d096 `
  --method v2_8a_static_a032_d112 $candidateRoot v2_8a_static_a032_d112 `
  --method v2_8a_static_a016_d128 $candidateRoot v2_8a_static_a016_d128 `
  --method v2_8a_static_a000_d144 $candidateRoot v2_8a_static_a000_d144 `
  --output_dir $metricRoot `
  --feature_backend dino `
  --prompt_backend clip `
  --device cuda
```

Expected local outputs are `case_metrics.csv`, `method_summary.csv`, and `summary.json` under `$metricRoot`.

This step uses `facebook/dinov2-base` and `openai/clip-vit-base-patch32`. First use requires network access unless both are already cached by Hugging Face. For an offline cached run, set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `DIFFUSERS_OFFLINE=1`, and add `--local_files_only`. Offline mode fails intentionally if a required model is missing from the cache.

The evaluator's `ref_copy_ssim_max` is a global RGB SSIM used by the local selector analysis; it is not a claim about an external or reported sliding-window Copy SSIM implementation.

## 5. Apply RefMax + Guard

```powershell
.venv\Scripts\python.exe scripts/analyze_v2_14_refmax_guard_hybrid.py `
  --eval_run frozen_seed42 ($metricRoot / "case_metrics.csv") `
  --output_dir $selectionRoot `
  --baseline_method v2_8a_static_a064_d080 `
  --candidate_methods v2_8a_static_a064_d080 v2_8a_static_a048_d096 v2_8a_static_a032_d112 v2_8a_static_a016_d128 v2_8a_static_a000_d144 `
  --dino_copy_delta_threshold 0.10 `
  --bootstrap_iterations 10000
```

Expected local outputs are `v2_14_refmax_guard_hybrid_predictions.csv` and `v2_14_refmax_guard_hybrid_summary.json`. Selection uses reference-side metrics; target-side metrics are used only to evaluate the selected outputs.

## 6. Optional external wrapper

An independently maintained formal wrapper can be exercised only by explicit opt-in:

```powershell
$wrapperPath = Read-Host "Path to the external wrapper.py"
$env:QMARGIN_EXTERNAL_WRAPPER = (Resolve-Path $wrapperPath).Path
.venv\Scripts\python.exe -m pytest -q tests/test_reference_conditioning_modes.py
```

The wrapper is not bundled. When `QMARGIN_EXTERNAL_WRAPPER` is unset, its integration tests skip. Do not commit wrapper outputs, generated images, downloaded weights, dataset images, caches, or formal evidence bundles.
