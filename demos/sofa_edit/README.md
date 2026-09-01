# Sofa img2img editing demo

This demo performs full-image FlowMatch img2img editing with the frozen local `FLUX.2-klein-base-4B` pipeline. It starts from one reference image, preserves a black three-seat leather sofa as far as the img2img trajectory allows, and replaces the white studio background with a warm modern living room.

## Required local input

The manifest at `demos/sofa_edit/edit_manifest.jsonl` references this local image:

```text
data/subjects/sop_small_valid/sop_super_9/sop_18845/201386573034_0.JPG
```

The source image and generated result are local, ignored artifacts; they are not redistributed by this repository.

## Run the edit

Run from the repository root after installing the full CUDA environment:

```powershell
$project = (Get-Location).Path
$python = Join-Path $project ".venv\Scripts\python.exe"
$config = Join-Path $project "demos\sofa_edit\edit_config_512.yaml"
$manifest = Join-Path $project "demos\sofa_edit\edit_manifest.jsonl"
$output = Join-Path $project "outputs\sofa_img2img_edit"

& $python (Join-Path $project "scripts\run_v2_8_ctnr_oracle.py") `
  --config $config `
  --eval_set $manifest `
  --root $project `
  --output_root $output `
  --height 512 `
  --width 512 `
  --num_inference_steps 30 `
  --guidance_scale 1.0 `
  --manifest_schema single_reference_no_target `
  --edit_strength 0.96 `
  --methods v2_8a_static_a064_d080 `
  --device cuda
```

The edited image is written to:

```text
outputs/sofa_img2img_edit/sofa_living_room_edit/v2_8a_static_a064_d080/000000.png
```

The manifest fixes seed `42`. With 30 configured inference steps and edit strength `0.96`, the current scheduler starts at `t_start=2` and executes 28 edit steps. No checkpoint or adapter is loaded.

## Limitation

This is full-image rewriting, not mask-local inpainting or identity-locked compositing. Fine geometry, upholstery details, colors, or object proportions can change. Use the bicycle compositing demo when pixel-level foreground preservation is required and the source is suitable for white-screen extraction.
