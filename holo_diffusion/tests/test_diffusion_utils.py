# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from typing import Tuple

from ..utils.diffusion_utils import SimpleUnet3D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _setup_simple_unet_3d() -> Tuple[SimpleUnet3D, torch.Tensor, torch.Tensor]:
    simple_unet_3d = SimpleUnet3D().to(device)
    test_batch_size = 1  # only one voxel-grid fits on one gpu mostly
    resol = (32, 32, 32)
    dummy_x = torch.randn(
        test_batch_size,
        simple_unet_3d.in_channels,
        *resol,
        dtype=torch.float32,
        device=device,
    )
    dummy_timesteps = torch.randint(
        0, 1000, (test_batch_size,), dtype=torch.long, device=device
    )
    return simple_unet_3d, dummy_x, dummy_timesteps


def test_SimpleUnet3D_forward() -> None:
    simple_unet_3d, dummy_x, dummy_timesteps = _setup_simple_unet_3d()

    # go forward:
    output_features = simple_unet_3d(x=dummy_x, timesteps=dummy_timesteps)

    # CHECK: No NaNs in outputs
    print(
        f"shape: {output_features.shape}, min: {output_features.min().item()}, "
        f"max: {output_features.max().item()}"
    )
    torch.testing.assert_close(torch.isnan(output_features).sum().item(), 0)


def test_SimpleUnet3D_backward() -> None:
    simple_unet_3d, dummy_x, dummy_timestpes = _setup_simple_unet_3d()

    # go forward:
    output_features = simple_unet_3d(x=dummy_x, timesteps=dummy_timestpes)
    simple_unet_3d_weights = list(simple_unet_3d.parameters())

    # CHECK: No gradients in the MLP weights
    for weight in simple_unet_3d_weights:
        assert weight.grad is None

    # go backward:
    output_features.mean().backward()

    # CHECK: No NaNs in the gradients of the MLP weights
    for weight in simple_unet_3d_weights:
        print(
            f"grad shape: {weight.grad.shape}, grad_norm: {weight.grad.norm().item()}"
        )
        torch.testing.assert_close(torch.isnan(weight.grad).sum().item(), 0)
