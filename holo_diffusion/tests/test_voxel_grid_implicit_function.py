# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from typing import Tuple

from ..holo_voxel_grid_implicit_function import RenderMLP, HoloVoxelGridImplicitFunction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _setup_render_mlp() -> Tuple[RenderMLP, torch.Tensor, torch.Tensor]:
    render_mlp = RenderMLP().to(device)
    test_batch_size = 16
    dummy_features = torch.randn(
        test_batch_size, render_mlp.input_dims, dtype=torch.float32, device=device
    )
    dummy_viewdirs = torch.randn(test_batch_size, 3, dtype=torch.float32, device=device)
    dummy_viewdirs = dummy_viewdirs / dummy_viewdirs.norm(dim=-1, keepdim=True)
    return render_mlp, dummy_features, dummy_viewdirs


def _setup_voxel_grid_implicit_function() -> (
    Tuple[HoloVoxelGridImplicitFunction, torch.Tensor, torch.Tensor]
):
    vox_grid_if = HoloVoxelGridImplicitFunction().to(device)
    dummy_pts_3d = torch.rand(4, 64, 64, 16, 3, dtype=torch.float32, device=device)
    dummy_pts_3d = (dummy_pts_3d * 2.0 - 1.0) * (vox_grid_if.volume_extent / 2.0)
    dummy_features = torch.randn(
        1,
        vox_grid_if.n_hidden,
        *((vox_grid_if.resol,) * 3),
        dtype=torch.float32,
        device=device,
    )
    return vox_grid_if, dummy_pts_3d, dummy_features


def test_RenderMLP_forward() -> None:
    render_mlp, dummy_features, dummy_viewdirs = _setup_render_mlp()

    # go forward:
    densities, rad_feats, vp_idp_feats = render_mlp(dummy_features, dummy_viewdirs)

    # CHECK: No NaNs in outputs
    for tensor in [densities, rad_feats, vp_idp_feats]:
        print(
            f"shape: {tensor.shape}, min: {tensor.min().item()}, max: {tensor.max().item()}"
        )
        torch.testing.assert_close(torch.isnan(tensor).sum().item(), 0)


def test_RenderMLP_backward() -> None:
    render_mlp, dummy_features, dummy_viewdirs = _setup_render_mlp()

    # go forward:
    densities, rad_feats, vp_idp_feats = render_mlp(dummy_features, dummy_viewdirs)
    render_mlp_weights = list(render_mlp.parameters())

    # CHECK: No gradients in the MLP weights
    for weight in render_mlp_weights:
        assert weight.grad is None

    # go backward:
    (densities.mean() + rad_feats.mean() + vp_idp_feats.mean()).backward()

    # CHECK: No NaNs in the gradients of the MLP weights
    for weight in render_mlp_weights:
        print(
            f"grad shape: {weight.grad.shape}, grad_norm: {weight.grad.norm().item()}"
        )
        torch.testing.assert_close(torch.isnan(weight.grad).sum().item(), 0)


def test_VoxelGridImplicitFunction_forward() -> None:
    vox_grid_if, dummy_pts_3d, dummy_features = _setup_voxel_grid_implicit_function()

    # go forward:
    densities, features, _ = vox_grid_if(
        pts_3d=dummy_pts_3d, voxel_grid_features=dummy_features
    )

    # CHECK: No NaNs in outputs
    for tensor in [densities, features]:
        print(
            f"shape: {tensor.shape}, min: {tensor.min().item()}, max: {tensor.max().item()}"
        )
        torch.testing.assert_close(torch.isnan(tensor).sum().item(), 0)


def test_VoxelGridImplicitFunction_backward() -> None:
    vox_grid_if, dummy_pts_3d, dummy_features = _setup_voxel_grid_implicit_function()

    # go forward:
    densities, features, _ = vox_grid_if(
        pts_3d=dummy_pts_3d, voxel_grid_features=dummy_features
    )
    vox_grid_if_weights = list(vox_grid_if.parameters())

    # CHECK: No gradients in the voxel_grid_implicit_function weights
    for weight in vox_grid_if_weights:
        assert weight.grad is None

    # go backward:
    (densities.mean() + features.mean()).backward()

    # CHECK: No NaNs in the gradients of the voxel_grid_implicit_function weights
    for weight in vox_grid_if_weights:
        print(
            f"grad shape: {weight.grad.shape}, grad_norm: {weight.grad.norm().item()}"
        )
        torch.testing.assert_close(torch.isnan(weight.grad).sum().item(), 0)
