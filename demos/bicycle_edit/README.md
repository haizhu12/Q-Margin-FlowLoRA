# Bicycle background-editing demo

This deployment demo preserves a red-and-black mountain bicycle by generating only an empty background, then compositing the original foreground. A white-screen matte locks the conservative bicycle core pixel-for-pixel and decontaminates its antialiased boundary. It does not ask the diffusion model to redraw the bicycle.

The sales labels are deliberately retained because they overlap the frame; removing them would require inventing hidden bicycle structure.

## Required local input

The command expects this locally obtained source image relative to the repository root:

```text
data/subjects/sop_small_valid/sop_super_1/sop_11768/331617272124_0.JPG
```

The image is not redistributed by this repository. The output directory is also local and ignored by Git.

## Run the edit

Run from the repository root after installing the full CUDA environment:

```powershell
$project = (Get-Location).Path
$python = Join-Path $project ".venv\Scripts\python.exe"
$config = Join-Path $project "demos\bicycle_edit\edit_config_512.yaml"
$source = Join-Path $project "data\subjects\sop_small_valid\sop_super_1\sop_11768\331617272124_0.JPG"
$output = Join-Path $project "outputs\bicycle_identity_edit"

& $python (Join-Path $project "scripts\run_background_replace.py") `
  --config $config `
  --source_image $source `
  --output_dir $output `
  --background_prompt "An empty photorealistic alpine forest trail at golden hour, warm sunlight filtering through pine trees, natural ground and a clear open space in the center for a product, realistic depth, no bicycle, no bike, no vehicle, no people, no animals, no text, no watermark." `
  --width 512 `
  --height 512 `
  --num_inference_steps 30 `
  --guidance_scale 1.0 `
  --seed 314159 `
  --device cuda `
  --transparent_distance 6 `
  --opaque_distance 96 `
  --edge_feather_radius 1 `
  --core_neighbor_distance 64 `
  --bottom_start_y 295 `
  --bottom_end_y 315 `
  --bottom_transparent_distance 128 `
  --bottom_opaque_distance 180 `
  --bottom_core_neighbor_distance 96
```

The source is placed without resizing at offset `(6, 89)` on the `512 x 512` background. The command loads no checkpoint or adapter.

Expected local files under `outputs/bicycle_identity_edit/` are `background.png`, `foreground_mask.png`, `000000.png`, and `metadata.json`.

## Limitations

This is white-screen extraction, not general segmentation or inpainting. The retained local seven-image review is `3 pass / 1 warn / 3 fail`. Dark or colored bicycles on white backgrounds generally work; a white bicycle on a white or near-white backdrop can lose pale frame structure, while a non-white source background can be copied as foreground. Those review images and audits remain local and are not versioned.

This demo is not part of the quantitative paper protocol.
