import pytest
import torch

from qmargin.flux2_utils import assert_text_ids_match_condition, extend_txt_ids_for_ref_tokens


def test_extend_txt_ids_for_ref_tokens_2d_prepends_zero_ids():
    text_ids = torch.ones(512, 4)

    out = extend_txt_ids_for_ref_tokens(text_ids, num_ref_tokens=32, device="cpu")

    assert out.shape == (544, 4)
    assert torch.equal(out[:32], torch.zeros(32, 4))
    assert torch.equal(out[32:], text_ids)


def test_extend_txt_ids_for_ref_tokens_3d_prepends_zero_ids_per_batch():
    text_ids = torch.ones(2, 512, 4)

    out = extend_txt_ids_for_ref_tokens(text_ids, num_ref_tokens=32, device="cpu")

    assert out.shape == (2, 544, 4)
    assert torch.equal(out[:, :32], torch.zeros(2, 32, 4))
    assert torch.equal(out[:, 32:], text_ids)


def test_extend_txt_ids_requires_existing_text_ids():
    with pytest.raises(RuntimeError, match="text_ids are required"):
        extend_txt_ids_for_ref_tokens(None, num_ref_tokens=4, device="cpu")


def test_assert_text_ids_match_condition_checks_sequence_length():
    combined = torch.zeros(2, 10, 8)
    txt_ids = torch.zeros(2, 9, 4)

    with pytest.raises(AssertionError, match="combined_prompt_embeds length"):
        assert_text_ids_match_condition(combined, txt_ids)
