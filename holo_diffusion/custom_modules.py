# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from enum import Enum
from pytorch3d.implicitron.models.implicit_function.decoding_functions import (
    _xavier_init,
    DecoderActivation,
)
from pytorch3d.implicitron.tools.config import Configurable
from typing import Optional, Tuple


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
