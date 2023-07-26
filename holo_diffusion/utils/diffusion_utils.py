# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This module adapts the diffusion functionality 
from OpenAI's guided diffusion package to work with implicitron. Please note 
that we have made some changes to the guided_diffusion code as well"""

import torch

from pytorch3d.implicitron.tools.config import registry, ReplaceableBase, Configurable
from pytorch3d.implicitron.models.implicit_function.decoding_functions import (
    _xavier_init,
)
from typing import Optional, Tuple

from ..guided_diffusion.unet import UNetModel
from ..guided_diffusion.timestep_sampler import create_named_schedule_sampler
from ..guided_diffusion.gaussian_diffusion import (
    ModelMeanType,
    LossType,
    ModelVarType,
    GaussianDiffusion,
    get_named_beta_schedule,
)


class Unet3DBase(ReplaceableBase, torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError()


@registry.register
class SimpleUnet3D(Unet3DBase):
    image_size: int = 64
    in_channels: int = 128
    out_channels: int = 128
    model_channels: int = 128
    num_res_blocks: int = 2
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    attention_resolutions: Tuple[int, ...] = (8, 16)
    num_heads: int = 2
    dropout: float = 0.0
    # 3d down/upsamples have the same size in all 3 dims
    homogeneous_resample: bool = True

    def __post_init__(self):
        self._net = UNetModel(
            dims=3,
            image_size=self.image_size,
            in_channels=self.in_channels,
            model_channels=self.model_channels,
            out_channels=self.out_channels,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=self.attention_resolutions,
            dropout=self.dropout,
            channel_mult=self.channel_mult,
            num_classes=None,
            use_checkpoint=False,
            num_heads=self.num_heads,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            zero_last_conv=False,
            homogeneous_resample=self.homogeneous_resample,
        )

        for m in self._net.modules():
            if isinstance(m, (torch.nn.Conv3d, torch.nn.Linear)):
                _xavier_init(m)
                m.bias.data[:] = 0.0

    def forward(self, x, timesteps, cond_features=None):
        if cond_features is not None:
            x = torch.cat([x, cond_features], dim=1)
        y = self._net(x, timesteps=timesteps)
        return y


class ImplicitronGaussianDiffusion(Configurable):
    beta_schedule_type: str = "linear"
    num_steps: int = 1000
    beta_start_unscaled: float = 0.0001
    beta_end_unscaled: float = 0.02
    # Note that we use START_X here because of the photometric-loss we use
    model_mean_type: ModelMeanType = ModelMeanType.START_X
    model_var_type: ModelVarType = ModelVarType.FIXED_SMALL
    schedule_sampler_type: str = "uniform"

    def __post_init__(self):
        self._diffusion = GaussianDiffusion(
            betas=get_named_beta_schedule(
                self.beta_schedule_type,
                self.num_steps,
                self.beta_start_unscaled,
                self.beta_end_unscaled,
            ),
            model_mean_type=self.model_mean_type,
            model_var_type=self.model_var_type,
            # LossType is not used in holo_diffusion. We define our own loss
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )
        self._schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, self._diffusion
        )

    def training_losses(self, *args, **kwargs):
        # We don't use the loss from the diffusion model, but
        # this has been exposed just in case.
        return self._diffusion.training_losses(*args, **kwargs)

    # diffusion related methods:
    def q_sample(self, *args, **kwargs):
        return self._diffusion.q_sample(*args, **kwargs)

    def p_mean_variance(self, *args, **kwargs):
        return self._diffusion.p_mean_variance(*args, **kwargs)

    def p_sample(self, *args, **kwargs):
        return self._diffusion.p_sample(*args, **kwargs)

    def p_sample_loop(self, *args, **kwargs):
        return self._diffusion.p_sample_loop(*args, **kwargs)

    def p_sample_loop_progressive(self, *args, **kwargs):
        return self._diffusion.p_sample_loop_progressive(*args, **kwargs)

    # schedule sampler related methods:
    def sample_timesteps(self, *args, **kwargs):
        return self._schedule_sampler.sample(*args, **kwargs)
