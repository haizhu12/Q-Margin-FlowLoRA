from pathlib import Path

import pytest
import yaml

from qmargin.config_utils import validate_config


CURRENT_CONFIG = Path("configs/qm_ref_sop_small_valid_native_vae_native_coreset_k144_eval.yaml")


def minimal_frozen_config():
    return {
        "model": {
            "inference_model_id": "models/FLUX.2-klein-base-4B",
            "torch_dtype": "bf16",
        },
        "reference_adapter": {
            "source": "native_vae",
            "mode": "latent_prefix",
            "apply_native_adapter": False,
            "tokens_per_image": 144,
            "pool_grid": [12, 12],
            "ref_latent_image_size": 256,
            "coreset_anchor_tokens": 64,
            "txt_id_strategy": "zero_prefix",
        },
        "semantic_reference": {"enabled": False},
    }


def test_validate_config_accepts_frozen_native_vae_contract():
    cfg = validate_config(minimal_frozen_config())

    assert cfg["model"]["inference_model_id"].endswith("FLUX.2-klein-base-4B")
    assert cfg["reference_adapter"]["tokens_per_image"] == 144
    assert cfg["reference_adapter"]["pool_grid"] == [12, 12]


@pytest.mark.parametrize(
    "legacy_key",
    ["train", "loss", "teacher", "global_lora", "checkpoint", "negative_reference"],
)
def test_validate_config_rejects_legacy_training_sections(legacy_key):
    cfg = minimal_frozen_config()
    cfg[legacy_key] = {}

    with pytest.raises(ValueError, match="unsupported top-level"):
        validate_config(cfg)


def test_current_config_contains_only_frozen_inference_sections():
    raw = yaml.safe_load(CURRENT_CONFIG.read_text(encoding="utf-8"))

    assert set(raw) == {"model", "reference_adapter", "semantic_reference"}
    validate_config(raw)
