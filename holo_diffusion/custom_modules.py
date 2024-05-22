# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from enum import Enum
from pytorch3d.implicitron.models.implicit_function.decoding_functions import (
    _xavier_init,
    DecoderActivation,
)
from pytorch3d.implicitron.models.view_pooler.feature_aggregator import (
    FeatureAggregatorBase,
    _mask_target_view_features, 
    _get_view_sampling_mask,
    _avgmaxstd_reduction_function,
    ReductionFunction,
)
from pytorch3d.implicitron.models.view_pooler.view_sampler import (
    cameras_points_cartesian_product,
)
from pytorch3d.renderer import HarmonicEmbedding
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.implicitron.tools.config import Configurable, registry
from typing import Optional, Tuple, Union, Dict


class HiddenActivation(Enum):
    RELU = "relu"
    SOFTPLUS = "softplus"
    LEAKYRELU = "leakyrelu"


class LazyLinearWithXavierInit(torch.nn.LazyLinear):
    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            _xavier_init(self)
            self.bias.data[:] = 0.0


class MLPWithInputSkips(Configurable, torch.nn.Module):
    """Same as `pytorch3d.implicitron.models.implicit_function.decoding_functions.MLPWithInputSkips`
    but with the option to specify a different activation function for the hidden layers.
    """

    n_layers: int = 8
    input_dim: int = 39
    output_dim: int = 256
    skip_dim: int = 39
    hidden_dim: int = 256
    input_skips: Tuple[int, ...] = (5,)
    skip_affine_trans: bool = False
    last_layer_bias_init: Optional[float] = None
    hidden_activation: HiddenActivation = HiddenActivation.LEAKYRELU
    last_activation: DecoderActivation = DecoderActivation.SOFTPLUS
    use_xavier_init: bool = True

    def __post_init__(self):
        try:
            last_activation = {
                DecoderActivation.RELU: torch.nn.ReLU(True),
                DecoderActivation.SOFTPLUS: torch.nn.Softplus(),
                DecoderActivation.SIGMOID: torch.nn.Sigmoid(),
                DecoderActivation.IDENTITY: torch.nn.Identity(),
            }[self.last_activation]
        except KeyError as e:
            raise ValueError(
                "`last_activation` can only be `RELU`,"
                " `SOFTPLUS`, `SIGMOID` or `IDENTITY`."
            ) from e

        try:
            self.hidden_activation_module = {
                HiddenActivation.RELU: torch.nn.ReLU(True),
                HiddenActivation.SOFTPLUS: torch.nn.Softplus(),
                HiddenActivation.LEAKYRELU: torch.nn.LeakyReLU(
                    negative_slope=0.2, inplace=True
                ),
            }[self.hidden_activation]
        except KeyError as e:
            raise ValueError(
                "`hidden_activation` can only be `RELU`,"
                " `SOFTPLUS`, or `LEAKYRELU`."
            ) from e

        layers = []
        skip_affine_layers = []
        for layeri in range(self.n_layers):
            dimin = self.hidden_dim if layeri > 0 else self.input_dim
            dimout = self.hidden_dim if layeri + 1 < self.n_layers else self.output_dim

            if layeri > 0 and layeri in self.input_skips:
                if self.skip_affine_trans:
                    skip_affine_layers.append(
                        self._make_affine_layer(self.skip_dim, self.hidden_dim)
                    )
                else:
                    dimin = self.hidden_dim + self.skip_dim

            linear = torch.nn.Linear(dimin, dimout)
            if self.use_xavier_init:
                _xavier_init(linear)
            if layeri == self.n_layers - 1 and self.last_layer_bias_init is not None:
                torch.nn.init.constant_(linear.bias, self.last_layer_bias_init)
            layers.append(
                torch.nn.Sequential(linear, self.hidden_activation_module)
                if not layeri + 1 < self.n_layers
                else torch.nn.Sequential(linear, last_activation)
            )
        self.mlp = torch.nn.ModuleList(layers)
        if self.skip_affine_trans:
            self.skip_affines = torch.nn.ModuleList(skip_affine_layers)
        self._input_skips = set(self.input_skips)
        self._skip_affine_trans = self.skip_affine_trans

    def _make_affine_layer(self, input_dim, hidden_dim):
        l1 = torch.nn.Linear(input_dim, hidden_dim * 2)
        l2 = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2)
        if self.use_xavier_init:
            _xavier_init(l1)
            _xavier_init(l2)
        return torch.nn.Sequential(l1, self.hidden_activation_module, l2)

    def _apply_affine_layer(self, layer, x, z):
        mu_log_std = layer(z)
        mu, log_std = mu_log_std.split(mu_log_std.shape[-1] // 2, dim=-1)
        std = torch.nn.functional.softplus(log_std)
        return (x - mu) * std

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        if z is None:
            # if the skip tensor is None, we use `x` instead.
            z = x
        skipi = 0
        # pyre-fixme[6]: For 1st param expected `Iterable[Variable[_T]]` but got
        #  `Union[Tensor, Module]`.
        for li, layer in enumerate(self.mlp):
            # pyre-fixme[58]: `in` is not supported for right operand type
            #  `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`.
            if li in self._input_skips:
                if self._skip_affine_trans:
                    # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C._Te...
                    y = self._apply_affine_layer(self.skip_affines[skipi], y, z)
                else:
                    y = torch.cat((y, z), dim=-1)
                skipi += 1
            y = layer(y)
        return y

@registry.register
class MLPMeanFeatureAggregator(torch.nn.Module, FeatureAggregatorBase):
    """
    Aggregates using a Transformer

    Settings:
        reduction_functions: A list of `ReductionFunction`s` that reduce the
            the stack of source-view-specific features to a single feature.
    """

    exclude_target_view_mask_features: bool = True
    n_hidden: int = 128
    dim_out: int = 128
    n_layers: int = 1
    n_harmonic_functions_ray: int = 3
    checkpointed_mlp: bool = True

    def __post_init__(self):
        super().__init__()
        self._first_sampled = LazyLinearWithXavierInit(self.n_hidden)
        self._first_mean = LazyLinearWithXavierInit(self.n_hidden)
        self._last = torch.nn.Linear(self.n_hidden, self.dim_out)
        self._ray_dir_harmonic_embed = HarmonicEmbedding(
            n_harmonic_functions=self.n_harmonic_functions_ray,
        )

        _xavier_init(self._last)
        self._mlp = MLPWithInputSkips(
            n_layers=self.n_layers,
            input_dim=self.n_hidden,
            output_dim=self.n_hidden,
            skip_dim=self.n_hidden,
            hidden_dim=self.n_hidden,
            input_skips=[],
        )

    def get_aggregated_feature_dim(
        self,
        feats_or_feats_dim: Union[Dict[str, torch.Tensor], int],
    ):
        return self.dim_out

    def forward(
        self,
        feats_sampled: Dict[str, torch.Tensor],
        masks_sampled: torch.Tensor,
        camera: Optional[CamerasBase] = None,
        pts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feats_sampled: A `dict` of sampled feature tensors `{f_i: t_i}`,
                where each `t_i` is a tensor of shape
                `(minibatch, n_source_views, n_samples, dim_i)`.
            masks_sampled: A binary mask represented as a tensor of shape
                `(minibatch, n_source_views, n_samples, 1)` denoting valid
                sampled features.
            camera: A batch of `n_source_views` `CamerasBase` objects corresponding
                to the source view cameras.
            pts: A tensor of shape `(minibatch, n_samples, 3)` denoting the
                3D points whose 2D projections to source views were sampled in
                order to generate `feats_sampled` and `masks_sampled`.

        Returns:
            feats_aggregated: If `concatenate_output==True`, a tensor
                of shape `(minibatch, 1, n_samples, sum(dim_1, ... dim_N))`.
                If `concatenate_output==False`, a dictionary `{f_i: t_i_aggregated}`
                with each `t_i_aggregated` of shape `(minibatch, 1, n_samples, aggr_dim_i)`.
        """
        assert self.concatenate_output
        pts_batch, n_cameras, n_pts = masks_sampled.shape[:3]
        if self.exclude_target_view_mask_features:
            feats_sampled = _mask_target_view_features(feats_sampled)
        sampling_mask = _get_view_sampling_mask(
            n_cameras,
            pts_batch,
            masks_sampled.device,
            self.exclude_target_view,
        )
        aggr_weigths = masks_sampled[..., 0] * sampling_mask[..., None]

        ray_dirs = self._ray_dir_harmonic_embed(
            _get_point_to_source_camera_ray_dirs(camera, pts)
        )

        def _mlp_pass(feats_sampled_, ray_dirs_, aggr_weights_):
            feats_sampled_cat_ = (
                torch.cat(
                    [
                        *tuple(feats_sampled_.values()),
                        ray_dirs_,
                    ],
                    dim=-1,
                )
                * aggr_weights_[..., None]
            )  # b x n_src x n_pts x dim
            mean = _avgmaxstd_reduction_function(
                feats_sampled_cat_,
                aggr_weights_,
                dim=1,
                reduction_functions=[ReductionFunction.AVG],
            )
            mlp_in = self._first_sampled(feats_sampled_cat_) + self._first_mean(mean)
            mlp_out = self._last(self._mlp(mlp_in))
            return (mlp_out * torch.softmax(mlp_out[..., :1], dim=1)).sum(
                dim=1, keepdim=True
            )
        
        if self.checkpointed_mlp:
            feats_out = torch.utils.checkpoint.checkpoint(
                _mlp_pass,
                feats_sampled,
                ray_dirs,
                aggr_weigths,
            )
        else:
            feats_out = _mlp_pass(
                feats_sampled,
                ray_dirs,
                aggr_weigths,
            )

        # mlp_in = self._first_sampled(feats_sampled_cat) + self._first_mean(mean)
        # mlp_out = self._last(self._mlp(mlp_in))
        # feats_out = wmean(
        #     mlp_out,
        #     torch.softmax(mlp_out[..., 0], dim=1),
        #     dim=1,
        # )

        return feats_out


def _get_point_to_source_camera_ray_dirs(camera: CamerasBase, pts: torch.Tensor):
    n_cameras = camera.R.shape[0]
    pts_batch = pts.shape[0]

    camera_rep, pts_rep = cameras_points_cartesian_product(camera, pts)

    # does not produce nans randomly unlike get_camera_center() below
    cam_centers_rep = -torch.bmm(
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(torch.Tensor.__getitem__)[[Named(self,
        #  torch.Tensor), Named(item, typing.Any)], typing.Any], torch.Tensor],
        #  torch.Tensor, torch.nn.modules.module.Module]` is not a function.
        # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch.Tensor.permute)[[N...
        camera_rep.T[:, None],
        camera_rep.R.permute(0, 2, 1),
    ).reshape(-1, *([1] * (pts.ndim - 2)), 3)
    # cam_centers_rep = camera_rep.get_camera_center().reshape(
    #     -1, *([1]*(pts.ndim - 2)), 3
    # )

    ray_dirs = F.normalize(pts_rep - cam_centers_rep, dim=-1)
    # camera_rep = [                 pts_rep = [
    #     camera[0]                      pts[0],
    #     camera[0]                      pts[1],
    #     camera[0]                      ...,
    #     ...                            pts[batch_pts-1],
    #     camera[1]                      pts[0],
    #     camera[1]                      pts[1],
    #     camera[1]                      ...,
    #     ...                            pts[batch_pts-1],
    #     ...                            ...,
    #     camera[n_cameras-1]            pts[0],
    #     camera[n_cameras-1]            pts[1],
    #     camera[n_cameras-1]            ...,
    #     ...                            pts[batch_pts-1],
    # ]                              ]

    ray_dirs_reshape = ray_dirs.view(n_cameras, pts_batch, -1, 3)
    return ray_dirs_reshape.transpose(0, 1)
