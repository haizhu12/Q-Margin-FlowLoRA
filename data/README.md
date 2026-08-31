# Data boundary and local layout

Only lightweight JSONL manifests and the canonical validated subject index are versioned under `data/`:

```text
data/eval/*.jsonl
data/subjects/subjects_sop_small_valid.jsonl
```

Dataset images, copied subject trees, raw archives, metric caches, and generated data are local-only and ignored by Git. They are not redistributed by this repository.

## Relative-path rules

Paths inside fixed-reference evaluation JSONL are interpreted relative to the root passed as `--root` to `scripts/run_v2_8_ctnr_oracle.py` and `scripts/validate_fixed_reference_eval.py`. The metric evaluator interprets the same paths relative to `--project_root`. Pass the same root to generation and metrics.

Paths inside the canonical subject manifest are interpreted relative to `--data_root` by subject-manifest preparation and validation tools. Absolute paths are accepted by the code but should not be committed because they are machine-specific.

## Required local layout

A typical local checkout has:

```text
data/
  eval/
    your_fixed_eval.jsonl
  subjects/
    subjects_sop_small_valid.jsonl
    sop_small_valid/
      <category>/<subject>/<image files>
  raw/
    sop/<locally obtained archive contents>
models/
  FLUX.2-klein-base-4B/<authorized model files>
outputs/
  <generated candidates, metrics, and figures>
```

The `raw/`, subject-image, model, and output trees remain ignored even when present locally.

## Rights and protocol responsibilities

The Stanford Online Products dataset terms, the model provider's terms, and the Q-Margin experimental SOP are separate obligations. A versioned manifest records paths and protocol metadata; it does not grant rights to dataset images or model weights, and it does not replace the reproduction SOP. Obtain each asset from its authorized source, comply with its terms, and keep restricted assets outside publication history.
