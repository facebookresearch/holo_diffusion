# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to visualize a previously trained ``No-UNET'' model
or a non-diffusion based Unet_enabled model. Example call:
    python visualize_reconstruction.py \
    exp_dir='./exps/checkpoint_dir' visdom_show_preds=True visdom_port=8097 \
    n_eval_cameras=40 render_size="[64,64]" video_size="[256,256]"

Please use the other scripts for generating samples from a trained 
holo_diffusion model.
"""

import gc
import logging
import math
import os
import random
import torch

from omegaconf import OmegaConf
from pytorch3d.implicitron.tools.config import enable_get_default_args, get_default_args
from typing import Optional, Tuple

from experiment import Experiment
from holo_diffusion.utils.checkpoint_utils import load_experiment
from holo_diffusion.utils.render_utils.flyaround import render_flyaround

logger = logging.getLogger(__name__)

CANONICAL_CO3D_UP_AXIS: Tuple[float, float, float] = (-0.0396, -0.8306, -0.5554)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_reconstruction(
    exp_dir: str = "",
    restrict_sequence_name: Optional[str] = None,
    output_directory: Optional[str] = None,
    render_size: Optional[Tuple[int, int]] = None,
    video_size: Optional[Tuple[int, int]] = (256, 256),
    camera_path: str = "simple_360",
    split: str = "train",
    n_source_views: Optional[int] = None,
    n_eval_cameras: int = 25 * 3,
    max_n_sequences: int = 2,
    shuffle_sequences: bool = False,
    shuffle_seed: int = 42,
    seed: int = 3,
    trajectory_scale: float = 1.3,
    up: Tuple[float, float, float] = CANONICAL_CO3D_UP_AXIS,
    camera_elevation: float = -30.0 * (2 * math.pi / 360),
) -> None:
    """
    Given an `exp_dir` containing a trained Implicitron model, generates videos consisting
    of renderes of sequences from the dataset used to train and evaluate the trained
    Implicitron model.
    Args:
        exp_dir: Implicitron experiment directory.
        restrict_sequence_name: If set, defines the list of sequences to visualize.
        output_directory: If set, defines a custom directory to output visualizations to.
        render_size: The size (HxW) of the generated renders.
        video_size: The size (HxW) of the output video.
        camera_path: The camera path to use for visualization.
                    Can be:["simple_360"| "spiral" | "circular_lsq_fit" |
                    "figure_eight" | "trefoil_knot" | "figure_eight_knot"]
        split: The dataset split to use for visualization.
            Can be "train" / "val" / "test".
        n_source_views: The number of source views added to each rendered batch. These
            views are required. By default we use the one in the experiment, but this can override them.
        n_eval_cameras: The number of cameras each fly-around trajectory.
        max_n_sequences: The maximum number of sequences to visualize
        shuffle_sequences: If `True`, shuffles the sequences before visualizing them.
        shuffle_seed: The seed used for shuffling the sequences.
        seed: The seed used for the script.
        trajectory_scale: The scale of the fly-around trajectory.
        up: The up vector of the fly-around trajectory.
        camera_elevation: The elevation of the fly-around trajectory.
    """
    # In case an output directory is specified use it. If no output_directory
    # is specified create a vis folder inside the experiment directory
    if output_directory is None:
        output_directory = os.path.join(exp_dir, "visualize_reconstruction")
    os.makedirs(output_directory, exist_ok=True)

    # load the experiment:
    _, model, data_source = load_experiment(
        Experiment, exp_dir, restrict_sequence_name, render_size, seed, device
    )
    model.eval()

    if model.net_3d_enabled:
        assert (
            not model.diffusion_enabled
        ), "This script is only meant for visualizing reconstructions. "
        "Please use the other script for generating samples from a holodiffusion model."

    dataset = data_source.dataset_map_provider.get_dataset_map()[split]

    # compute the number of source views
    if n_source_views is None:
        n_source_views = (
            data_source.data_loader_map_provider.batch_size - model.n_train_target_views
        )

    if dataset is None:
        raise ValueError(f"{split} dataset not provided")
    sequence_names = list(dataset.sequence_names())
    if shuffle_sequences:
        random.Random(shuffle_seed).shuffle(sequence_names)
    sequence_names = sequence_names[:max_n_sequences]

    # iterate over the sequences in the dataset
    for sequence_name in sequence_names:
        with torch.no_grad():
            # free all remaining memory
            torch.cuda.empty_cache()
            gc.collect()

            render_flyaround(
                dataset=dataset,
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
            )


enable_get_default_args(visualize_reconstruction)


def main() -> None:
    # automatically parses arguments of visualize_reconstruction
    cfg = OmegaConf.create(get_default_args(visualize_reconstruction))
    cfg.update(OmegaConf.from_cli())
    with torch.no_grad():
        visualize_reconstruction(**cfg)


if __name__ == "__main__":
    main()
