import ast
from pathlib import Path


def test_flux2_utils_excludes_unreferenced_training_and_legacy_bridges():
    source_path = Path("qmargin/flux2_utils.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }

    forbidden = {
        "FlowTrainingPair",
        "assert_only_ref_modules_trainable",
        "normalize_timestep",
        "normalize_sigma_from_schedule",
        "build_condition_bundle",
        "build_semantic_native_reference_condition_bundle",
        "encode_images_with_vae",
        "pool_reference_latent_tokens",
        "pack_vae_latents_for_flux2",
        "unpack_transformer_latents_compat",
        "decode_latents_with_vae",
        "make_flow_training_pair",
        "_condition_from_legacy_args",
    }
    assert defined.isdisjoint(forbidden), sorted(defined & forbidden)


def test_predict_velocity_requires_an_explicit_condition_bundle():
    source_path = Path("qmargin/flux2_utils.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    predict = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "predict_velocity_compat"
    )
    argument_names = [argument.arg for argument in predict.args.args]

    assert "condition" in argument_names
    assert "prompts" not in argument_names
    assert "encoded" not in argument_names
    assert "img_ids" not in argument_names
