# README bicycle demo provenance

This record documents how the versioned README images were created. It is a provenance record, not a substitute for the terms governing the image-generation service, FLUX.2 weights, or repository source code.

## Input asset

`bicycle_input.png` was generated on 2026-09-01 with OpenAI's built-in image-generation tool from a text-only prompt. No source or reference image was supplied. The generated square image was resized to `500 x 500` with Pillow/Lanczos so that the project compositor could place it inside a `512 x 512` canvas without resizing at edit time.

- SHA-256: `9e0e9285e88d2b28c9c27f4288850038e2131bc0160d7cc3eccbe24489ebd481`
- Visible-content review: one unbranded red-and-black mountain bicycle on white; no people, watermark, readable text, or visible trademark.

Prompt:

```text
Use case: product-mockup
Asset type: rights-clear input image for a research-code README image-editing demo
Primary request: create a photorealistic studio product photograph of one modern unbranded mountain bicycle, shown in a clean near-perfect side view.
Scene/backdrop: seamless pure white background with only a very faint neutral contact shadow beneath the tires.
Subject: one complete bicycle with a vivid red frame, matte-black fork, saddle, handlebar, drivetrain, and black tires; realistic thin metallic spokes; both wheels, pedals, cables, and the entire frame fully visible.
Style/medium: high-resolution realistic catalog product photography, natural materials and mechanically plausible bicycle geometry.
Composition/framing: square canvas, bicycle centered, generous white margin on every side, both tires entirely inside the frame, no cropping, level camera, no perspective distortion.
Lighting/mood: bright soft studio lighting, high contrast between bicycle and white background.
Constraints: exactly one bicycle; no people; no props; no logos; no trademarks; no labels; no text; no watermark; no colored backdrop; preserve a crisp silhouette suitable for foreground extraction.
Avoid: white or pale bicycle parts, motion blur, dramatic shadows, clutter, duplicate wheels, malformed frame geometry, kickstand crossing the frame.
```

The phrase “rights-clear” above is part of the generation prompt, not a legal conclusion. The README describes this asset as synthetic and links to this record instead of making an independent licensing claim.

## Background and edited result

`bicycle_generated_background.png` and `bicycle_edited_result.png` were produced on 2026-09-01 by `scripts/run_background_replace.py` with the local `models/FLUX.2-klein-base-4B` Diffusers directory. The command is versioned in the root README.

- GPU: NVIDIA GeForce RTX 4090
- Runtime used for this saved example: PyTorch `2.5.1+cu121`, Diffusers `0.38.0`, Transformers `4.57.6`
- Resolution: `512 x 512`
- Inference steps: `30`
- Guidance scale: `1.0`
- Seed: `20260901`
- Generated background SHA-256: `5f6d674abba67599c84d1da0bdbf7bfd5f60d0bdbee0f855191e98e60d78671d`
- Final edit SHA-256: `1d8934ede3ebb38fd3a13af4dbbcf3962701d1942fa9b637055b2eb1b8fa2858`
- Fully locked foreground pixels: `26,855`
- Soft-edge pixels: `15,959`
- Maximum absolute RGB error over locked pixels after saving: `0`

The upstream FLUX.2 model revision was not recorded for this local snapshot, so the saved images are reference outputs rather than a bitwise reproducibility claim.
