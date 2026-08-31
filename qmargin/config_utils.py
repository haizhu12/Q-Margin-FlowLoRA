from __future__ import annotations

from copy import deepcopy


SUPPORTED_TOP_LEVEL_KEYS = {"model", "reference_adapter", "semantic_reference"}
SUPPORTED_DTYPES = {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}


def validate_config(cfg: dict) -> dict:
    """Validate the frozen native-VAE inference contract used by the paper runner."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    cfg = deepcopy(cfg)

    unsupported = sorted(set(cfg) - SUPPORTED_TOP_LEVEL_KEYS)
    if unsupported:
        raise ValueError(f"unsupported top-level config sections: {', '.join(unsupported)}")

    model = cfg.get("model")
    if not isinstance(model, dict) or not model.get("inference_model_id"):
        raise ValueError("model.inference_model_id is required")
    dtype = str(model.get("torch_dtype", "bf16")).lower()
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError("model.torch_dtype must be one of: bf16, fp16, fp32")
    model["torch_dtype"] = dtype

    adapter = cfg.get("reference_adapter")
    if not isinstance(adapter, dict):
        raise ValueError("reference_adapter is required")
    if adapter.get("source") != "native_vae":
        raise ValueError("reference_adapter.source must be 'native_vae'")
    if adapter.get("mode") != "latent_prefix":
        raise ValueError("reference_adapter.mode must be 'latent_prefix'")
    if bool(adapter.get("apply_native_adapter", False)):
        raise ValueError("reference_adapter.apply_native_adapter must be false")

    tokens_per_image = int(adapter.get("tokens_per_image", 144))
    if tokens_per_image < 1:
        raise ValueError("reference_adapter.tokens_per_image must be >= 1")
    adapter["tokens_per_image"] = tokens_per_image

    pool_grid = adapter.get("pool_grid")
    if not isinstance(pool_grid, (list, tuple)) or len(pool_grid) != 2:
        raise ValueError("reference_adapter.pool_grid must be [height, width]")
    grid_h, grid_w = (int(pool_grid[0]), int(pool_grid[1]))
    if grid_h < 1 or grid_w < 1 or grid_h * grid_w != tokens_per_image:
        raise ValueError("reference_adapter.pool_grid area must equal tokens_per_image")
    adapter["pool_grid"] = [grid_h, grid_w]

    image_size = int(adapter.get("ref_latent_image_size", 256))
    if image_size < 1:
        raise ValueError("reference_adapter.ref_latent_image_size must be >= 1")
    adapter["ref_latent_image_size"] = image_size

    anchor_tokens = int(adapter.get("coreset_anchor_tokens", 64))
    if not 1 <= anchor_tokens <= tokens_per_image:
        raise ValueError("reference_adapter.coreset_anchor_tokens must be within [1, tokens_per_image]")
    adapter["coreset_anchor_tokens"] = anchor_tokens

    txt_id_strategy = str(adapter.get("txt_id_strategy", "zero_prefix"))
    if txt_id_strategy != "zero_prefix":
        raise ValueError("reference_adapter.txt_id_strategy must be 'zero_prefix'")
    adapter["txt_id_strategy"] = txt_id_strategy

    semantic_reference = cfg.setdefault("semantic_reference", {"enabled": False})
    if not isinstance(semantic_reference, dict):
        raise ValueError("semantic_reference must be a mapping")
    if bool(semantic_reference.get("enabled", False)):
        raise ValueError("semantic_reference.enabled must be false for frozen paper inference")
    semantic_reference["enabled"] = False
    return cfg
