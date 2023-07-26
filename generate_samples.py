# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to generate random samples from a trained ``HoloDiffusion'' model. Example call:
    python generate_samples.py \
    exp_dir='./exps/checkpoint_dir' \
    n_eval_cameras=40 render_size="[64,64]" video_size="[256,256]"

Please use the other scripts for generating samples from a trained 
holo_diffusion model.
"""

import gc
import logging
import math
import os
import torch

from omegaconf import OmegaConf
from pytorch3d.implicitron.tools.config import enable_get_default_args, get_default_args
from typing import Optional, Tuple

from experiment import Experiment
from holo_diffusion.utils.checkpoint_utils import load_experiment
from holo_diffusion.utils.render_utils.flyaround import render_flyaround
from visualize_reconstruction import CANONICAL_CO3D_UP_AXIS

logger = logging.getLogger(__name__)

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_samples(
    exp_dir: str = "",
    output_directory: Optional[str] = None,
    render_size: Optional[Tuple[int, int]] = None,
    video_size: Optional[Tuple[int, int]] = (256, 256),
    camera_path: str = "simple_360",
    n_eval_cameras: int = 25 * 3,
    num_samples: int = 2,
    seed: int = 3,
    trajectory_scale: float = 1.3,
    up: Tuple[float, float, float] = CANONICAL_CO3D_UP_AXIS,
    camera_elevation: float = -30.0 * (2 * math.pi / 360),
    progressive_sampling_steps_per_render: int = -1,
    save_voxel_features: bool = True,
) -> None:
    """
    Given an `exp_dir` containing a trained Implicitron model, generates videos consisting
    of renderes of sequences from the dataset used to train and evaluate the trained
    Implicitron model.
    Args:
        exp_dir: Implicitron experiment directory.
        output_directory: If set, defines a custom directory to output visualizations to.
        render_size: The size (HxW) of the generated renders.
        video_size: The size (HxW) of the output video.
        camera_path: The camera path to use for visualization.
                    Can be:["simple_360"| "spiral" | "circular_lsq_fit" |
                    "figure_eight" | "trefoil_knot" | "figure_eight_knot"]
        n_eval_cameras: The number of cameras each fly-around trajectory.
        num_samples: The number of samples to visualize
        seed: The random seed used for generating the fly-around trajectories.
        trajectory_scale: The scale of the fly-around trajectory.
        up: The up vector of the fly-around trajectory.
        camera_elevation: The elevation of the fly-around trajectory.
        progressive_sampling_steps_per_render: Number of steps to denoise in case rendering
                                               animation of denoising sampling. -1 means disbaled.
        save_voxel_features: whether to save the voxel features tensors to disk.
    """
    # In case an output directory is specified use it. If no output_directory
    # is specified create a vis folder inside the experiment directory
    if output_directory is None:
        folder_name = (
            "generated_samples"
            if progressive_sampling_steps_per_render == -1
            else "generated_samples_denoising"
        )
        output_directory = os.path.join(exp_dir, folder_name)
    os.makedirs(output_directory, exist_ok=True)

    # load the experiment:
    _, model, data_source = load_experiment(
        Experiment, exp_dir, None, render_size, seed, device
    )

    assert (
        model.net_3d_enabled and model.diffusion_enabled
    ), "Can generate random samples only from a trained HoloDiffusion model. "
    "Please use the other script for visualizing reconstructions."

    # compute the number of source views
    # this is not really used while generating random samples, but put them here for completeness
    n_source_views = (
        data_source.data_loader_map_provider.batch_size - model.n_train_target_views
    )

    # setup sequence names for the generated samples:
    sequence_names = [f"sample_{i:05d}" for i in range(num_samples)]

    # iterate over the sequences in the dataset
    for sequence_name in sequence_names:
        with torch.no_grad():
            # free all remaining memory
            torch.cuda.empty_cache()
            gc.collect()

            render_flyaround(
                dataset=None,
                sequence_name=sequence_name,
                model=model,
                output_video_path=os.path.join(output_directory, "video"),
                output_video_name=sequence_name,
                n_source_views=n_source_views,
                n_flyaround_poses=n_eval_cameras,
                trajectory_type=camera_path,
                video_resize=video_size,
                device=device,
                up=up,
                trajectory_scale=trajectory_scale,
                camera_elevation=camera_elevation,
                sample_mode=True,
                progressive_sampling_steps_per_render=progressive_sampling_steps_per_render,
                visualize_preds_keys=(
                    "images_render",
                    "masks_render",
                    "depths_render",
                    "noise_render",
                    "images_prev_stage_render",
                    "features_prev_stage_render",
                    "_shaded_depth_render",
                    "_all_source_images",
                ),
                save_voxel_features=save_voxel_features,
            )


enable_get_default_args(generate_samples)


def main() -> None:
    # automatically parses arguments of visualize_reconstruction
    cfg = OmegaConf.create(get_default_args(generate_samples))
    cfg.update(OmegaConf.from_cli())
    with torch.no_grad():
        generate_samples(**cfg)


if __name__ == "__main__":
    main()
