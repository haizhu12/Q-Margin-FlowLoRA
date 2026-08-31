# Q-Margin

Q-Margin is a training-free, target-free selection protocol for reference-guided generation with a frozen `FLUX.2-klein-base-4B` model. It generates five equal-budget reference-token routes, selects the candidate with the highest mean DINO reference similarity (RefMax), and conditionally falls back through a copy-risk Guard.

No LoRA, adapter, or other learned checkpoint is loaded. Target images are used only for offline evaluation, never for candidate construction or selection.

## Method and frozen protocol

Each reference is converted to RGB, center-cropped/resized to `256 x 256`, and encoded by the model's native VAE. The total reference-token budget is fixed at `K=144`; the five `(anchor, detail)` routes are `(64,80)`, `(48,96)`, `(32,112)`, `(16,128)`, and `(0,144)`. RefMax chooses the largest mean DINO similarity across references. Guard reranks only when the selected candidate's maximum single-reference DINO similarity exceeds that of Static `(64,80)` by more than `0.10`.

The frozen reproduction setting is `512 x 512`, 30 inference steps, guidance scale `1.0`, `bf16`, and seed `42` in every evaluation-manifest record. The route runner reads the seed from each record; it has no seed command-line option.

## Repository layout

```text
qmargin/                         inference and native token-routing runtime
scripts/run_v2_8_ctnr_oracle.py  five-route generation
scripts/evaluate_fixed_reference_metrics.py
                                 local candidate metrics
scripts/analyze_v2_14_refmax_guard_hybrid.py
                                 target-free RefMax + Guard selection
scripts/create_v2_14_quant_trackI_figures.py
                                 figure generation from local metric artifacts
configs/                         frozen runtime configuration
data/eval/                       versioned JSONL evaluation manifests
demos/                           local-input editing recipes
docs/REPRODUCIBILITY.md          complete reproduction procedure
tests/                           CPU unit and contract tests
```

Model files, source images, generated candidates, metrics, and figures are local artifacts and are ignored by Git. See [`data/README.md`](data/README.md) for the data boundary.

## CPU quick start

Python 3.11 or newer is required. The test suite does not download models or require a GPU.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[test]"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:DIFFUSERS_OFFLINE = "1"
.venv\Scripts\python.exe -m pytest -q
```

## Full CUDA environment

`requirements.freeze.txt` is the validated direct-dependency snapshot for CUDA 12.8. Its editable install is intentional, so run this command from the repository root:

```powershell
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.freeze.txt
```

The frozen core includes `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, Diffusers `0.38.0`, Transformers `5.10.2`, and Matplotlib `3.11.0`. Use a driver/runtime compatible with the official PyTorch CUDA 12.8 wheels.

## Required local model and data

From the repository root, provide authorized model files and dataset images at the paths referenced by the configuration and manifests:

```text
models/
  FLUX.2-klein-base-4B/
data/
  eval/*.jsonl
  subjects/sop_small_valid/<category>/<subject>/<image>
```

Relative paths in fixed-reference manifests resolve against `--root` during generation and against `--project_root` during metrics. Pass the same repository/data root to both. The project does not redistribute model weights, dataset images, or raw archives; obtain them under their respective terms.

## Run and reproduce

The task-oriented command sequence, manifest schemas, local cache behavior, expected files, and RefMax + Guard invocation are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Two deployment examples are documented separately:

- [`demos/bicycle_edit/README.md`](demos/bicycle_edit/README.md): identity-preserving foreground compositing over a generated background.
- [`demos/sofa_edit/README.md`](demos/sofa_edit/README.md): full-image FlowMatch img2img editing.

Both demos require local source images and write ignored local results. The retained local bicycle review is `3 pass / 1 warn / 3 fail`: pale bicycles on pale backgrounds and non-white source backgrounds remain known failure modes.

## Optional external wrapper integration

Some integration tests can exercise a separately obtained formal wrapper. Opt in by pointing `QMARGIN_EXTERNAL_WRAPPER` to that local Python file:

```powershell
$wrapperPath = Read-Host "Path to the external wrapper.py"
$env:QMARGIN_EXTERNAL_WRAPPER = (Resolve-Path $wrapperPath).Path
.venv\Scripts\python.exe -m pytest -q tests/test_reference_conditioning_modes.py
```

When the variable is unset, external-wrapper tests skip. The wrapper and its outputs are not part of this repository.

## Scope

This repository contains the frozen Q-Margin runtime, manifests, tests, and public reproduction instructions. It does not publish private working notes, local evidence bundles, generated outputs, a paper source tree, or external evaluation wrappers.
