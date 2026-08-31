import pytest
import torch

from qmargin.flux2_utils import (
    ConditionBundle,
    normalize_timestep_from_schedule,
    predict_velocity_compat,
)


class ForwardRecorder:
    def __init__(self):
        self.kwargs = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        return_dict=True,
        num_ref_tokens=0,
        ref_fixed_timestep=0.0,
    ):
        self.kwargs = dict(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=return_dict,
            num_ref_tokens=num_ref_tokens,
            ref_fixed_timestep=ref_fixed_timestep,
        )
        return (hidden_states + 1,)


class FakePipe:
    def __init__(self):
        self.transformer = ForwardRecorder()


def make_condition(img_ids=None, txt_ids=None):
    return ConditionBundle(
        prompt_embeds=torch.zeros(1, 5, 8),
        combined_prompt_embeds=torch.zeros(1, 7, 8),
        txt_ids=txt_ids,
        img_ids=img_ids,
        guidance=torch.ones(1),
        ref_tokens=torch.zeros(1, 2, 8),
        num_ref_tokens=2,
    )


def test_predict_velocity_requires_supported_img_and_txt_ids():
    latents = torch.zeros(1, 4, 8)
    condition = make_condition(img_ids=None, txt_ids=torch.zeros(7, 4))

    with pytest.raises(RuntimeError, match="img_ids"):
        predict_velocity_compat(FakePipe(), latents, torch.tensor([0.5]), condition)


def test_predict_velocity_filters_and_forwards_condition_bundle():
    latents = torch.zeros(1, 4, 8)
    img_ids = torch.zeros(1, 4, 4)
    txt_ids = torch.zeros(7, 4)
    pipe = FakePipe()

    pred = predict_velocity_compat(pipe, latents, torch.tensor([0.5]), make_condition(img_ids, txt_ids))

    assert pred.shape == latents.shape
    assert pipe.transformer.kwargs["encoder_hidden_states"].shape == (1, 7, 8)
    assert pipe.transformer.kwargs["img_ids"] is img_ids
    assert pipe.transformer.kwargs["txt_ids"] is txt_ids
    assert pipe.transformer.kwargs["num_ref_tokens"] == 2


def test_schedule_timestep_normalization_not_constant():
    timesteps = torch.tensor([1000, 960, 920, 500, 0], dtype=torch.float32)

    values = torch.stack([normalize_timestep_from_schedule(t, timesteps) for t in timesteps])

    assert values[0].item() == pytest.approx(1.0)
    assert values[-1].item() == pytest.approx(0.0)
    assert values.std().item() > 1e-4


def test_scalar_timestep_normalization_forbidden():
    with pytest.raises(RuntimeError, match="full scheduler timesteps"):
        normalize_timestep_from_schedule(torch.tensor(1000.0), torch.tensor([1000.0]))
