import torch

from qmargin.flux2_utils import PromptBundle, encode_prompt_compat


class TuplePromptPipe:
    def encode_prompt(self, prompt, device):
        return torch.ones(1, 5, 8), torch.zeros(5, 4)


class DictPromptPipe:
    def encode_prompt(self, prompt, device, max_sequence_length=None):
        return {
            "prompt_embeds": torch.ones(1, 5, 8),
            "txt_ids": torch.zeros(1, 5, 4),
            "pooled_prompt_embeds": torch.ones(1, 8),
        }


def test_encode_prompt_tuple_second_item_is_text_ids_not_pooled():
    bundle = encode_prompt_compat(TuplePromptPipe(), "a product photo", device="cpu")

    assert isinstance(bundle, PromptBundle)
    assert bundle.prompt_embeds.shape == (1, 5, 8)
    assert bundle.internal_text_ids.shape == (5, 4)
    assert bundle.pooled_prompt_embeds is None


def test_encode_prompt_dict_preserves_text_ids_and_pooled_embeds():
    bundle = encode_prompt_compat(DictPromptPipe(), "a product photo", device="cpu", max_sequence_length=16)

    assert bundle.internal_text_ids.shape == (1, 5, 4)
    assert bundle.pooled_prompt_embeds.shape == (1, 8)
