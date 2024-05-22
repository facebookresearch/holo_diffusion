# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import tempfile
import torch
import warnings

from dataclasses import field
from pytorch3d.renderer import utils as rend_utils
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures.volumes import VolumeLocator
from pytorch3d.implicitron.models.renderer.base import ImplicitronRayBundle
from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.models.base_model import ImplicitronRender
from pytorch3d.implicitron.models.implicit_function.base import ImplicitFunctionBase
from pytorch3d.implicitron.models.renderer.base import (
    EvaluationMode,
    ImplicitFunctionWrapper,
    RenderSamplingMode,
)
from pytorch3d.implicitron.models.utils import preprocess_input
from pytorch3d.implicitron.tools import vis_utils
from pytorch3d.implicitron.tools.config import registry
from pytorch3d.implicitron.tools.video_writer import VideoWriter
from pytorch3d.implicitron.tools.rasterize_mc import rasterize_sparse_ray_bundle
from typing import Any, Dict, List, Optional, Union
from visdom import Visdom

from .utils.diffusion_utils import Unet3DBase, ImplicitronGaussianDiffusion
from .custom_modules import LazyLinearWithXavierInit

# Needed for registry to get populated properly
from .holo_multipass_ea import HoloMultiPassEmissionAbsorptionRenderer
from .holo_voxel_grid_implicit_function import HoloVoxelGridImplicitFunction

logger = logging.getLogger(__name__)


@registry.register
class HoloDiffusionModel(GenericModel):  # pyre-ignore: 13
    # ---- model config
    resol: int = 32  # voxel grid resolution
    volume_extent: float = 8.0  # size of voxel grid in world units
    feature_size: int = 128  # feature size for the voxel_grids
    num_passes: int = 2  # Number of passes done during rendering

    # ---- 3D Unet model
    net_3d_enabled: bool = True
    net_3d: Optional[Unet3DBase]
    net_3d_class_type: str = "SimpleUnet3D"

    # ---- Gaussian Diffusion
    diffusion_enabled: bool = True
    diffusion: ImplicitronGaussianDiffusion
    enable_bootstrap: bool = True
    bootstrap_prob: float = 0.5

    # ---- loss weights
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "loss_rgb_mse": 1.0,
            "loss_prev_stage_rgb_mse": 1.0,
            "loss_prev_stage_prev_stage_rgb_mse": 1.0,
            "loss_prev_stage_prev_stage_prev_stage_rgb_mse": 1.0,
            "loss_prev_stage_prev_stage_prev_stage_prev_stage_rgb_mse": 1.0,
            "loss_mask_bce": 0.0,
            "loss_prev_stage_mask_bce": 0.0,
            "loss_prev_stage_prev_stage_mask_bce": 0.0,
        }
    )

    # ---- variables to be logged (logger automatically ignores if not computed)
    log_vars: List[str] = field(
        default_factory=lambda: [
            "loss_rgb_psnr_fg",
            "loss_rgb_psnr",
            "loss_rgb_mse",
            "loss_rgb_huber",
            "loss_depth_abs",
            "loss_depth_abs_fg",
            "loss_mask_neg_iou",
            "loss_mask_bce",
            "loss_mask_beta_prior",
            "loss_eikonal",
            "loss_density_tv",
            "loss_depth_neg_penalty",
            "loss_autodecoder_norm",
            # metrics that are only logged in 2+stage renderes
            "loss_prev_stage_rgb_mse",
            "loss_prev_stage_prev_stage_rgb_mse",
            "loss_prev_stage_prev_stage_prev_stage_rgb_mse",
            "loss_prev_stage_prev_stage_prev_stage_prev_stage_rgb_mse",
            "loss_prev_stage_prev_stage_prev_stage_prev_stage_prev_stage_rgb_mse",
            "loss_prev_stage_rgb_psnr_fg",
            "loss_prev_stage_rgb_psnr",
            "loss_prev_stage_mask_bce",
            "loss_prev_stage_prev_stage_mask_bce",
            "loss_prev_stage_prev_stage_prev_stage_mask_bce",
            # basic metrics
            "objective",
            "epoch",
            "sec/it",
        ]
    )

    def __post_init__(self):
        super().__post_init__()
        self.pooled_feature_mapper = LazyLinearWithXavierInit(self.feature_size)
        warnings.warn("Setting target view exclusion to False by hard!")
        self.view_pooler.feature_aggregator.exclude_target_view = False
        self.view_pooler.feature_aggregator.exclude_target_view_mask_features = False

    def create_net_3d(self):
        self.net_3d = None
        if self.net_3d_enabled:
            extra_args = dict(
                in_channels=self.feature_size,
                out_channels=self.feature_size,
                image_size=self.resol,
            )
            net_3d_args = getattr(self, "net_3d_" + self.net_3d_class_type + "_args")
            net_3d_args = {**net_3d_args, **extra_args}
            self.net_3d = registry.get(Unet3DBase, self.net_3d_class_type)(
                **net_3d_args
            )

    def create_diffusion(self):
        self.diffusion = None
        if self.diffusion_enabled:
            diffusion_args = getattr(self, "diffusion_args")
            self.diffusion = ImplicitronGaussianDiffusion(**diffusion_args)

    def _construct_implicit_functions(self):
        """
        After run_auto_creation has been called, the arguments
        for each of the possible implicit function methods are
        available. `GenericModel` arguments are first validated
        based on the custom requirements for each specific
        implicit function method. Then the required implicit
        function(s) are initialized.
        """
        if self.implicit_function_class_type != "HoloVoxelGridImplicitFunction":
            raise ValueError(
                f"{str(type(self))} supports only HoloVoxelGridImplicitFunction!"
            )

        extra_args = {}
        extra_args["resol"] = self.resol
        extra_args["volume_extent"] = self.volume_extent
        extra_args["n_hidden"] = self.feature_size
        extra_args["feature_dim"] = 0  # no additional features are rendered
        implicit_function_type = registry.get(
            ImplicitFunctionBase, self.implicit_function_class_type
        )
        config_name = f"implicit_function_{self.implicit_function_class_type}_args"
        config = getattr(self, config_name, None)
        if config is None:
            raise ValueError(f"{config_name} not present")

        # use the same implicit function for all passes so that the RenderMLP is not replicated
        if_ = ImplicitFunctionWrapper(
            implicit_function_type(**{**config, **extra_args})
        )
        implicit_functions_list = [if_ for _ in range(self.num_passes)]

        return torch.nn.ModuleList(implicit_functions_list)

    def sample_random_voxel_features_progressive(self) -> torch.Tensor:
        # creates a generator over the sampling process of the diffusion (denoising)
        # you can use each of the generated sample separately.
        assert self.net_3d_enabled and self.diffusion_enabled

        for sample in self.diffusion.p_sample_loop_progressive(
            model=self.net_3d,
            shape=(1, self.feature_size, self.resol, self.resol, self.resol),
            clip_denoised=True,
            progress=False,
        ):
            # Note that we manually clip each sample to [-1, 1] here.
            # so that the RenderMLP expects the same input range as the final generated samples.
            yield torch.clip(sample["sample"], -1.0, 1.0)

    def sample_random_voxel_features(self) -> torch.Tensor:
        assert self.net_3d_enabled and self.diffusion_enabled

        # generate random voxel features using the diffusion and net_3d:
        logger.info("generating random voxel features through denoising diffusion ...")
        random_voxel_features = self.diffusion.p_sample_loop(
            model=self.net_3d,
            shape=(1, self.feature_size, self.resol, self.resol, self.resol),
            clip_denoised=True,
            progress=True,
        )
        return random_voxel_features

    def forward(
        self,
        *,  # force keyword-only arguments
        image_rgb: Optional[torch.Tensor] = None,
        camera: CamerasBase,
        fg_probability: Optional[torch.Tensor] = None,
        mask_crop: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        sequence_name: Optional[List[str]] = None,
        frame_timestamp: Optional[torch.Tensor] = None,
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        voxel_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            image_rgb: A tensor of shape `(B, 3, H, W)` containing a batch of rgb images;
                the first `min(B, n_train_target_views)` images are considered targets and
                are used to supervise the renders; the rest corresponding to the source
                viewpoints from which features will be extracted.
            camera: An instance of CamerasBase containing a batch of `B` cameras corresponding
                to the viewpoints of target images, from which the rays will be sampled,
                and source images, which will be used for intersecting with target rays.
            fg_probability: A tensor of shape `(B, 1, H, W)` containing a batch of
                foreground masks.
            mask_crop: A binary tensor of shape `(B, 1, H, W)` deonting valid
                regions in the input images (i.e. regions that do not correspond
                to, e.g., zero-padding). When the `RaySampler`'s sampling mode is set to
                "mask_sample", rays  will be sampled in the non zero regions.
            depth_map: A tensor of shape `(B, 1, H, W)` containing a batch of depth maps.
            sequence_name: A list of `B` strings corresponding to the sequence names
                from which images `image_rgb` were extracted. They are used to match
                target frames with relevant source frames.
            frame_timestamp: Optionally a tensor of shape `(B,)` containing a batch
                of frame timestamps.
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering.
            voxel_features: A tensor of shape `(B, C, H, W, D)` containing a batch of
                voxel features. If provided, we use these voxel features to render
                the target_views instead of pooling them from the source_views.
        Returns:
            preds: A dictionary containing all outputs of the forward pass including the
                rendered images, depths, masks, losses and other metrics.
        """

        # masks the images image_rgb with fg_probability
        image_rgb, fg_probability, depth_map = preprocess_input(
            image_rgb,
            fg_probability,
            depth_map,
            self.mask_images,
            self.mask_depths,
            self.mask_threshold,
            self.bg_color,
        )

        # Obtain the batch size and device from the camera as this is the only required input.
        batch_size = camera.R.shape[0]
        device = camera.R.device

        # Determine the number of target views, i.e. cameras we render into.
        n_targets = (
            1
            if evaluation_mode == EvaluationMode.EVALUATION
            else batch_size
            if self.n_train_target_views <= 0
            else min(self.n_train_target_views, batch_size)
        )

        if batch_size <= n_targets:
            logger.info("Batch size <= n_targets!")
            n_targets = 1
            if batch_size == 1:
                logger.warning("Batch size == 1!")

        def safe_slice_(
            tensor: Optional[Union[torch.Tensor, List[str]]],
            is_target: bool,
        ) -> Optional[Union[torch.Tensor, List[str]]]:
            if tensor is None:
                return None
            if is_target:
                sel = list(range(n_targets))
            else:
                # choose the first set of images corresponding to the sequence
                # of the first image
                ok_ = [
                    si for si, s in enumerate(sequence_name) if s == sequence_name[0]
                ]
                sel = ok_[n_targets:]
            if len(sel) == 0:
                logger.info("Nothing to slice!")
                sel = list(range(batch_size))
            try:
                tensor = tensor[sel]
            except TypeError:
                tensor = [tensor[s] for s in sel]
            return tensor

        # A helper function for selecting n_target first elements from the input
        # where the latter can be None.
        def safe_slice_targets(
            tensor: Optional[Union[torch.Tensor, List[str]]],
        ) -> Optional[Union[torch.Tensor, List[str]]]:
            return safe_slice_(tensor, True)

        # A helper function for source views.
        def safe_slice_sources(
            tensor: Optional[Union[torch.Tensor, List[str]]],
        ) -> Optional[Union[torch.Tensor, List[str]]]:
            return safe_slice_(tensor, False)

        # Select the target cameras.
        target_cameras = camera[list(range(n_targets))]

        # Determine the used ray sampling mode.
        sampling_mode = RenderSamplingMode(
            self.sampling_mode_training
            if evaluation_mode == EvaluationMode.TRAINING
            else self.sampling_mode_evaluation
        )

        # custom_args hold additional arguments to the implicit function.
        custom_args = {}
        voxel_batch_size = 1  # only one single voxel grid is supported per GPU
        if image_rgb is not None:
            # ----------------------------------------------------------------------------------------
            #  View pooling Mechanism (i.e. views -> voxel-grid)                                     |
            # ----------------------------------------------------------------------------------------
            # We use the view-pooler to obtain the voxel-grid.
            # fmt: off
            assert self.view_pooler_enabled, "view_pooler must be enabled to use image_rgb"
            assert voxel_features is None, "Cannot provide both image_rgb and voxel_features"
            assert self.image_feature_extractor is not None, "Need an image_feature_extractor"
            assert not self.view_pooler.feature_aggregator.exclude_target_view
            assert not self.view_pooler.feature_aggregator.exclude_target_view_mask_features
            # fmt: on

            # obtain image features using the image feature extractor
            img_feats = self.image_feature_extractor(
                safe_slice_sources(image_rgb) if batch_size > 1 else image_rgb,
                safe_slice_sources(fg_probability)
                if batch_size > 1
                else fg_probability,
            )

            # create a volume locator
            locator = VolumeLocator(
                voxel_batch_size,
                grid_sizes=(self.resol, self.resol, self.resol),
                device=device,
                voxel_size=self.volume_extent / self.resol,
            )
            grid_xyz = locator.get_coord_grid().reshape(voxel_batch_size, -1, 3)

            # 2) pool features from the grid cell points
            voxel_features = self.view_pooler(
                pts=grid_xyz,
                seq_id_pts=sequence_name[:1],
                camera=safe_slice_sources(camera) if batch_size > 1 else camera,
                seq_id_camera=safe_slice_sources(sequence_name)
                if batch_size > 1
                else sequence_name,
                feats=img_feats,
                masks=safe_slice_sources(mask_crop) if batch_size > 1 else mask_crop,
            )
            voxel_features = self.pooled_feature_mapper(voxel_features)
            voxel_features = voxel_features.permute(0, 3, 1, 2).reshape(
                voxel_batch_size, -1, self.resol, self.resol, self.resol
            )
            # this brings the pooled features in the range [-1, 1]
            voxel_features = torch.tanh(voxel_features)
            # ----------------------------------------------------------------------------------------

        if image_rgb is None and voxel_features is None:
            assert evaluation_mode == EvaluationMode.EVALUATION
            # Sample a new voxel grid.
            voxel_features = self.sample_random_voxel_features()

        assert voxel_features.min() >= -1.0 and voxel_features.max() <= 1.0
        # --------------------------------------------------------------------------------------------
        #         Diffusion Mechanism                                                                |
        # --------------------------------------------------------------------------------------------
        if self.net_3d_enabled:
            if self.diffusion_enabled and evaluation_mode == EvaluationMode.TRAINING:
                timesteps, _ = self.diffusion.sample_timesteps(voxel_batch_size, device)
                diffused_voxel_features = self.diffusion.q_sample(
                    voxel_features, timesteps
                )
                # predicted clean ones:
                voxel_features = self.diffusion.p_mean_variance(
                    model=self.net_3d,
                    x=diffused_voxel_features,
                    t=timesteps,
                    clip_denoised=True,
                    model_kwargs={},
                )["pred_xstart"]
                assert voxel_features.min() >= -1.0 and voxel_features.max() <= 1.0

                if evaluation_mode == EvaluationMode.TRAINING and (
                    np.random.uniform() < self.bootstrap_prob
                ):
                    # Perform a denoising diffusion pass on the clean voxel features again
                    timesteps, _ = self.diffusion.sample_timesteps(
                        voxel_batch_size, device
                    )
                    diffused_voxel_features = self.diffusion.q_sample(
                        voxel_features, timesteps
                    )
                    voxel_features = self.diffusion.p_mean_variance(
                        model=self.net_3d,
                        x=diffused_voxel_features,
                        t=timesteps,
                        clip_denoised=True,
                        model_kwargs={},
                    )["pred_xstart"]
                    assert voxel_features.min() >= -1.0 and voxel_features.max() <= 1.0

            else:
                timesteps = torch.zeros(
                    (voxel_batch_size,), dtype=torch.long, device=device
                )
                voxel_features = self.net_3d(voxel_features, timesteps)
                voxel_features = torch.tanh(voxel_features)
                assert voxel_features.min() >= -1.0 and voxel_features.max() <= 1.0
        # --------------------------------------------------------------------------------------------
        assert voxel_features.min() >= -1.0 and voxel_features.max() <= 1.0

        assert voxel_features.shape[1] == self.feature_size, "Wrong voxel feature size!"
        custom_args["voxel_grid_features"] = voxel_features

        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(torch.Tensor.__iter__)[[Named(self,
        #  torch.Tensor)], typing.Iterator[typing.Any]], torch.Tensor], torch.Tensor,
        #  torch.nn.Module]` is not a function.
        for func in self._implicit_functions:
            func.bind_args(**custom_args)

        # Sample rendering rays with the ray sampler.
        # pyre-ignore[29]
        ray_bundle: ImplicitronRayBundle = self.raysampler(
            target_cameras,
            evaluation_mode,
            mask=safe_slice_targets(mask_crop)
            if mask_crop is not None and sampling_mode == RenderSamplingMode.MASK_SAMPLE
            else None,
        )

        # Implicit function evaluation and Rendering
        rendered = self._render(
            ray_bundle=ray_bundle,
            sampling_mode=sampling_mode,
            evaluation_mode=evaluation_mode,
            implicit_functions=self._implicit_functions,
            inputs_to_be_chunked={},
        )

        # Unbind the custom arguments to prevent pytorch from storing
        # large buffers of intermediate results due to points in the
        # bound arguments.
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(torch.Tensor.__iter__)[[Named(self,
        #  torch.Tensor)], typing.Iterator[typing.Any]], torch.Tensor], torch.Tensor,
        #  torch.nn.Module]` is not a function.
        for func in self._implicit_functions:
            func.unbind_args()

        preds: Dict[str, Any] = {}

        # A dict to store losses as well as rendering results.
        preds.update(
            {
                "rendered": rendered,
                "ray_bundle": ray_bundle,
            }
        )

        preds.update(
            self.view_metrics(
                results=preds,
                raymarched=rendered,
                ray_bundle=ray_bundle,
                image_rgb=safe_slice_targets(image_rgb),
                depth_map=safe_slice_targets(depth_map),
                fg_probability=safe_slice_targets(fg_probability),
                mask_crop=safe_slice_targets(mask_crop),
                sampling_mode=sampling_mode,
            )
        )

        preds.update(
            self.regularization_metrics(
                results=preds,
                model=self,
            )
        )

        if sampling_mode == RenderSamplingMode.MASK_SAMPLE:
            if self.output_rasterized_mc:
                # Visualize the monte-carlo pixel renders by splatting onto
                # an image grid.
                (
                    preds["images_render"],
                    preds["depths_render"],
                    preds["masks_render"],
                ) = rasterize_sparse_ray_bundle(
                    ray_bundle,
                    rendered.features,
                    (self.render_image_height, self.render_image_width),
                    rendered.depths,
                    masks=rendered.masks,
                )

        elif sampling_mode == RenderSamplingMode.FULL_GRID:
            preds["images_render"] = rendered.features.permute(0, 3, 1, 2)
            preds["depths_render"] = rendered.depths.permute(0, 3, 1, 2)
            preds["masks_render"] = rendered.masks.permute(0, 3, 1, 2)
            preds["implicitron_render"] = ImplicitronRender(
                image_render=preds["images_render"],
                depth_render=preds["depths_render"],
                mask_render=preds["masks_render"],
            )

        else:
            raise AssertionError(f"Unknown sampling mode: {sampling_mode}")

        # Compute losses
        # finally get the optimization objective using self.loss_weights
        objective = self._get_objective(preds)
        if objective is not None:
            # add objective to predictions
            preds["objective"] = objective

            # added as a hack for accelerate to work:
            for param in self.parameters():
                if param.requires_grad:
                    preds["objective"] += 0 * param.sum()

        return preds

    def visualize(
        self,
        viz: Visdom,
        visdom_env_imgs: str,
        preds: Dict[str, Any],
        prefix: str,
    ) -> None:
        """
        Helper function to visualize the predictions generated
        in the forward pass.

        Args:
            viz: Visdom connection object
            visdom_env_imgs: name of visdom environment for the images.
            preds: predictions dict like returned by forward()
            prefix: prepended to the names of images
        """
        if not viz.check_connection():
            logger.info("no visdom server! -> skipping batch vis")
            return

        idx_image = 0
        title = f"{prefix}_im{idx_image}"

        # viz.close(env=visdom_env_imgs)
        vis_utils.visualize_basics(viz, preds, visdom_env_imgs, title=title)

        # show the prev stage as well
        rendered = preds["rendered"]

        def _show_rendered(r, prefix):
            show_ = {"image": r.features, "depth": r.depths, "mask": r.masks}
            if "x_t" in r.aux:
                show_["x_t"] = r.aux["x_t"]
            if "features" in r.aux:
                show_["feats"] = _feats_to_rgb(r.aux["features"])
            for k, v in show_.items():
                title = "rendered_" + prefix + k
                v = v.permute(0, 3, 1, 2)
                if v.shape[1] == 1:
                    v = v.repeat(1, 3, 1, 1)
                elif v.shape[1] != 3:
                    v = _feats_grid_to_rgb(v)
                viz.images(
                    v.clamp(0.0, 1.0),
                    env=visdom_env_imgs,
                    win=title,
                    opts={"title": title},
                )
            if r.prev_stage is not None:
                _show_rendered(r.prev_stage, "ps_" + prefix)

        if not self.output_rasterized_mc:
            # show these only when not using rasterized mc
            _show_rendered(rendered, "")

        if rendered.aux.get("samples", None) is not None:
            samples = rendered.aux["samples"]
            outfile = tempfile.NamedTemporaryFile(suffix=".mp4")
            vw = VideoWriter(out_path=outfile.name)
            for sample_ in samples:
                sample = sample_["pred_xstart"]
                if sample.shape[1] != 3:
                    sample = _feats_grid_to_rgb(sample)[0]
                else:
                    sample = sample[0]
                frame = sample.clamp(0.0, 1.0)
                vw.write_frame(frame.detach().cpu().numpy())
            vid_fl = vw.get_video()
            viz.video(
                videofile=vid_fl,
                env=visdom_env_imgs,
                win="sample_video",
                opts={"title": "samples"},
            )


def _feats_grid_to_rgb(f: torch.Tensor):
    b, dim, he, wi = f.shape
    f_ = f.reshape(b, dim, -1).permute(0, 2, 1)
    rgb = _feats_to_rgb(f_).permute(0, 2, 1).reshape(b, 3, he, wi)
    return rgb


def _feats_to_rgb(f: torch.Tensor):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        w = torch.randn(f.shape[-1], 3, dtype=f.dtype, device=f.device)
    f_rgb = 0.5 + 0.5 * torch.nn.functional.normalize(
        f.reshape(-1, f.shape[-1]) @ w,
        dim=-1,
    ).reshape(*f.shape[:-1], 3)
    return f_rgb
