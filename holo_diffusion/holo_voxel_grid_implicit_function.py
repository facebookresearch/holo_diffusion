# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch

from pytorch3d.renderer import ray_bundle_to_ray_points, HarmonicEmbedding
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures.volumes import VolumeLocator
from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle
from pytorch3d.implicitron.models.implicit_function.base import ImplicitFunctionBase
from pytorch3d.implicitron.models.implicit_function.voxel_grid import (
    VoxelGridBase,
    VoxelGridValuesBase,
    FullResolutionVoxelGrid,
    FullResolutionVoxelGridValues,
)
from pytorch3d.implicitron.tools.config import (
    registry,
    run_auto_creation,
    Configurable,
)
from pytorch3d.implicitron.models.implicit_function.decoding_functions import (
    DecoderActivation,
)
from typing import NamedTuple, Optional, Tuple

from .custom_modules import (
    MLPWithInputSkips,
    HiddenActivation,
)


logger = logging.getLogger(__name__)

COLOUR_DIMS: int = 3


class LocalizedVoxelGrid(NamedTuple):
    locator: VolumeLocator
    grid: VoxelGridBase
    values: VoxelGridValuesBase


class RenderMLP(Configurable, torch.nn.Module):
    input_dims: int = 128
    output_feature_dims: int = COLOUR_DIMS
    output_vp_independent_feature_dims: int = 64
    feat_emb_dims: int = 0
    dir_emb_dims: int = 4
    dnet_num_layers: int = 4
    dnet_hidden_dim: int = 256
    dnet_input_skips: Tuple[int, ...] = (2,)
    rnet_num_layers: int = 1
    rnet_hidden_dim: int = 128
    rnet_input_skips: Tuple[int, ...] = ()
    activation_fn: HiddenActivation = HiddenActivation.LEAKYRELU

    def __post_init__(self):
        # harmonic embedders
        self._feats_encoder = HarmonicEmbedding(self.feat_emb_dims)
        self._dir_encoder = HarmonicEmbedding(self.dir_emb_dims)
        
        if isinstance(self.activation_fn, str):
            # in case the activation_fn is a string, convert it to the corresponding enum
            # this happens while loading the config from yaml
            self.activation_fn = HiddenActivation[self.activation_fn]

        # mlps for density and radiance
        self._density_net = MLPWithInputSkips(
            n_layers=self.dnet_num_layers,
            input_dim=self._feats_encoder.get_output_dim(input_dims=self.input_dims),
            output_dim=self.dnet_hidden_dim + 1,  # 1 for density
            skip_dim=self._feats_encoder.get_output_dim(input_dims=self.input_dims),
            hidden_dim=self.dnet_hidden_dim,
            input_skips=self.dnet_input_skips,
            hidden_activation=self.activation_fn,
            last_activation=DecoderActivation.IDENTITY,
        )
        self._radiance_net = MLPWithInputSkips(
            n_layers=self.rnet_num_layers,
            input_dim=self.dnet_hidden_dim + self._dir_encoder.get_output_dim(),
            output_dim=self.output_feature_dims,
            skip_dim=self.dnet_hidden_dim + self._dir_encoder.get_output_dim(),
            hidden_dim=self.rnet_hidden_dim,
            input_skips=self.rnet_input_skips,
            hidden_activation=self.activation_fn,
            last_activation=DecoderActivation.IDENTITY,
        )

        self._feature_net = None
        if self.output_vp_independent_feature_dims > 0:
            self._feature_net = MLPWithInputSkips(
                n_layers=self.rnet_num_layers,
                input_dim=self.dnet_hidden_dim,
                output_dim=self.output_vp_independent_feature_dims,
                skip_dim=self.dnet_hidden_dim,
                hidden_dim=self.rnet_hidden_dim,
                input_skips=self.rnet_input_skips,
                hidden_activation=self.activation_fn,
                last_activation=DecoderActivation.IDENTITY,
            )

    def forward(
        self, features: torch.Tensor, view_dirs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # density network
        pe_features = self._feats_encoder(features)
        out = self._density_net(pe_features)
        mlp_feats, densities = out[..., :-1], out[..., -1:]
        # densities = torch.nn.functional.softplus(densities)

        # radiance network
        pe_viewdirs = self._dir_encoder(view_dirs)
        radiance_features = self._radiance_net(
            torch.cat([mlp_feats, pe_viewdirs], dim=-1)
        )
        if self.output_feature_dims == COLOUR_DIMS:
            # sigmoid for colour
            radiance_features = torch.sigmoid(radiance_features)

        vp_independent_features = None
        if self._feature_net is not None:
            vp_independent_features = self._feature_net(mlp_feats)

        return densities, radiance_features, vp_independent_features

    def get_normals(self, sampling_fun, pts: torch.Tensor):
        def _fwpass(x):
            f = sampling_fun(x)
            pe_features = self._feats_encoder(f)
            out = self._density_net(pe_features)
            densities = out[..., -1:]
            return densities

        with torch.enable_grad():
            x = pts.clone()
            x.requires_grad = True
            y = _fwpass(x).sum()
            first_derivative = torch.autograd.grad(y, x, create_graph=True)[0]
        normals = torch.nn.functional.normalize(first_derivative, dim=-1)
        return normals


@registry.register
class HoloVoxelGridImplicitFunction(ImplicitFunctionBase, torch.nn.Module):
    resol: int = 32
    volume_extent: float = 8.0
    n_hidden: int = 128
    feature_dim: int = 64
    init_density_bias: float = 1e-4

    # render_mlp:
    render_mlp: RenderMLP

    # visualisation switch
    render_normals: bool = False

    def __post_init__(self):
        run_auto_creation(self)

    def create_render_mlp(self):
        render_mlp_args = getattr(self, "render_mlp_args")
        updated_args = {
            "input_dims": self.n_hidden,
            "output_feature_dims": COLOUR_DIMS,
            "output_vp_independent_feature_dims": self.feature_dim,
        }
        self.render_mlp = RenderMLP(**{**render_mlp_args, **updated_args})

    @staticmethod
    def allows_multiple_passes() -> bool:
        """
        Returns True as this implicit function allows
        multiple passes. Overridden from ImplicitFunctionBase.
        """
        return True

    def forward(
        self,
        *,
        ray_bundle: ImplicitronRayBundle = None,
        fun_viewpool=None,
        camera: Optional[CamerasBase] = None,
        global_code=None,
        run_id=None,
        pass_number=None,
        pts_3d: torch.Tensor = None,
        voxel_grid_features: torch.Tensor = None,
        **kwargs,
    ):
        assert voxel_grid_features is not None, "voxel_grid_features must be provided!"
        assert (
            ray_bundle is not None or pts_3d is not None
        ), "either ray_bundle or pts_3d must be provided!"
        ray_points_world = (
            ray_bundle_to_ray_points(ray_bundle) if pts_3d is None else pts_3d
        )
        spatial_size = ray_points_world.shape[:-1]

        locator = VolumeLocator(
            batch_size=1,  # only batch size of 1 is supported
            grid_sizes=(self.resol, self.resol, self.resol),
            device=ray_points_world.device,
            voxel_size=self.volume_extent / self.resol,
        )
        voxels = LocalizedVoxelGrid(
            locator=locator,
            grid=FullResolutionVoxelGrid(n_features=self.n_hidden),
            values=FullResolutionVoxelGridValues(voxel_grid_features),
        )

        # sample the features on the voxel-grid
        sampled_voxel_feats = voxels.grid.evaluate_world(
            ray_points_world.reshape(1, -1, 3),
            voxels.values,
            voxels.locator,
        )
        sampled_voxel_feats = sampled_voxel_feats.reshape(
            *spatial_size,
            sampled_voxel_feats.shape[-1],
        )

        # decode the features using the render_mlp
        aux_preds = {}
        rays_directions = (
            ray_bundle.directions
            if ray_bundle is not None
            else torch.ones(  # dummy directions
                *spatial_size[:-1],
                3,
                dtype=torch.float32,
                device=ray_points_world.device,
            )
        )
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)
        ray_dirs_input = rays_directions_normed[..., None, :].expand(
            *sampled_voxel_feats.shape[:-2], sampled_voxel_feats.shape[-2], 3
        )
        (
            densities,
            colour_features,
            vp_independent_features,
        ) = self.render_mlp(sampled_voxel_feats, ray_dirs_input)

        if self.render_normals:
            def _sampling_fun(p):
                spatial_size_ = p.shape[:-1]
                _sampled_voxel_feats = voxels.grid.evaluate_world(
                    p.reshape(1, -1, 3),
                    voxels.values,
                    voxels.locator,
                ).reshape(
                    *spatial_size_,
                    voxels.grid.n_features,
                )
                return _sampled_voxel_feats

            normals = self.render_mlp.get_normals(_sampling_fun, ray_points_world)
            aux_preds["normals"] = normals

        # produce the output
        features = colour_features
        if vp_independent_features is not None:
            features = torch.cat([colour_features, vp_independent_features], dim=-1)
        return densities, features, aux_preds
