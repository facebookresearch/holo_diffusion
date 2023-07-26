# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


""" 
    Modified from pytorch3d.implicitron.models.visualization.render_flyaround 
    to include a custom trajectories while visualization.
"""

import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch
import torch.nn.functional as Fu
from pytorch3d.implicitron.dataset.dataset_base import DatasetBase, FrameData
from pytorch3d.implicitron.dataset.utils import is_train_frame
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.models.base_model import EvaluationMode
from pytorch3d.implicitron.tools.eval_video_trajectory import (
    generate_eval_video_cameras,
    _fit_plane,
)
from pytorch3d.implicitron.tools.video_writer import VideoWriter
from pytorch3d.implicitron.tools.vis_utils import (
    get_visdom_connection,
    make_depth_image,
)
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras
from tqdm import tqdm

if TYPE_CHECKING:
    from visdom import Visdom

logger = logging.getLogger(__name__)


def render_flyaround(
    dataset: DatasetBase,
    sequence_name: str,
    model: torch.nn.Module,
    output_video_path: str,
    output_video_name: Optional[str] = None,
    n_flyaround_poses: int = 40,
    fps: int = 20,
    trajectory_type: str = "circular_lsq_fit",
    max_angle: float = 2 * math.pi,
    trajectory_scale: float = 1.1,
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, -1.0, 0.0),
    camera_elevation: float = -30.0 * (2 * math.pi / 360),
    camera_focal_length: float = 3.2,
    hemispherical_radius: float = 10,
    traj_offset: float = 0.0,
    n_source_views: int = 9,
    visdom_show_preds: bool = False,
    visdom_environment: str = "render_flyaround",
    visdom_server: str = "http://127.0.0.1",
    visdom_port: int = 8097,
    num_workers: int = 10,
    device: Union[str, torch.device] = "cuda",
    seed: Optional[int] = None,
    video_resize: Optional[Tuple[int, int]] = None,
    output_video_frames_dir: Optional[str] = None,
    sample_mode: bool = False,
    progressive_sampling_steps_per_render: int = -1,
    visualize_preds_keys: Sequence[str] = (
        "images_render",
        "masks_render",
        "depths_render",
        "_all_source_images",
    ),
    save_voxel_features: bool = False,
) -> None:
    """
    Uses `model` to generate a video consisting of renders of a scene imaged from
    a camera flying around the scene. The scene is specified with the `dataset` object and
    `sequence_name` which denotes the name of the scene whose frames are in `dataset`.
    Args:
        dataset: The dataset object containing frames from a sequence in `sequence_name`.
        sequence_name: Name of a sequence from `dataset`.
        model: The model whose predictions are going to be visualized.
        output_video_path: The path to the video output by this script.
        n_flyaround_poses: The number of camera poses of the flyaround trajectory.
        fps: Framerate of the output video.
        trajectory_type: The type of the camera trajectory. Can be one of:
            simple_360: Camera centers follow a simple 360 deg trajectory.
            spiral: Camera centers follow a spiral trajectory.
            circular_lsq_fit: Camera centers follow a trajectory obtained
                by fitting a 3D circle to train_cameras centers.
                All cameras are looking towards scene_center.
            figure_eight: Figure-of-8 trajectory around the center of the
                central camera of the training dataset.
            trefoil_knot: Same as 'figure_eight', but the trajectory has a shape
                of a trefoil knot (https://en.wikipedia.org/wiki/Trefoil_knot).
            figure_eight_knot: Same as 'figure_eight', but the trajectory has a shape
                of a figure-eight knot
                (https://en.wikipedia.org/wiki/Figure-eight_knot_(mathematics)).
        trajectory_type: The type of the camera trajectory. Can be one of:
            circular_lsq_fit: Camera centers follow a trajectory obtained
                by fitting a 3D circle to train_cameras centers.
                All cameras are looking towards scene_center.
            figure_eight: Figure-of-8 trajectory around the center of the
                central camera of the training dataset.
            trefoil_knot: Same as 'figure_eight', but the trajectory has a shape
                of a trefoil knot (https://en.wikipedia.org/wiki/Trefoil_knot).
            figure_eight_knot: Same as 'figure_eight', but the trajectory has a shape
                of a figure-eight knot
                (https://en.wikipedia.org/wiki/Figure-eight_knot_(mathematics)).
        max_angle: Defines the total length of the generated camera trajectory.
            All possible trajectories (set with the `trajectory_type` argument) are
            periodic with the period of `time==2pi`.
            E.g. setting `trajectory_type=circular_lsq_fit` and `time=4pi` will generate
            a trajectory of camera poses rotating the total of 720 deg around the object.
        trajectory_scale: The extent of the trajectory.
        scene_center: The center of the scene in world coordinates which all
            the cameras from the generated trajectory look at.
        up: The "up" vector of the scene (=the normal of the scene floor).
            Active for the `trajectory_type="circular"`.
        traj_offset: 3D offset vector added to each point of the trajectory.
        n_source_views: The number of source views sampled from the known views of the
            training sequence added to each evaluation batch.
        visdom_show_preds: If `True`, exports the visualizations to visdom.
        visdom_environment: The name of the visdom environment.
        visdom_server: The address of the visdom server.
        visdom_port: The visdom port.
        num_workers: The number of workers used to load the training data.
        seed: The random seed used for reproducible sampling of the source views.
        video_resize: Optionally, defines the size of the output video.
        output_video_frames_dir: If specified, the frames of the output video are going
            to be permanently stored in this directory.
        visualize_preds_keys: The names of the model predictions to visualize.
    """

    if seed is None:
        seed = hash(sequence_name)

    if visdom_show_preds:
        viz = get_visdom_connection(server=visdom_server, port=visdom_port)
    else:
        viz = None

    # setup the train_data and batch based on the mode (reconstruction or sample)
    if sample_mode:
        train_data = None
        batch = _get_dummy_test_batch_for_sampling(n_source_views + 1, device=device)
    else:  # reconstruction mode
        logger.info(f"Loading all data of sequence '{sequence_name}'.")
        seq_idx = list(dataset.sequence_indices_in_order(sequence_name))
        train_data = _load_whole_dataset(dataset, seq_idx, num_workers=num_workers)
        assert all(train_data.sequence_name[0] == sn for sn in train_data.sequence_name)
        # pyre-ignore[6]
        if train_data.frame_type is not None:
            sequence_set_name = (
                "train" if is_train_frame(train_data.frame_type)[0] else "test"
            )
            logger.info(f"Sequence set = {sequence_set_name}.")
        # sample the source views reproducibly
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            source_views_i = torch.randperm(len(seq_idx))[:n_source_views]

        # add the first dummy view that will get replaced with the target camera
        source_views_i = Fu.pad(source_views_i, [1, 0])
        source_views = [seq_idx[i] for i in source_views_i.tolist()]
        batch = _load_whole_dataset(dataset, source_views, num_workers=num_workers)
        assert all(batch.sequence_name[0] == sn for sn in batch.sequence_name)

    # setup the test-rendering cameras:
    if trajectory_type.lower() == "simple_360":
        test_cameras = get_simple_360_camera_trajectory(
            max_angle,
            n_flyaround_poses,
            camera_elevation,
            hemispherical_radius,
            up,
            camera_focal_length,
        )
    elif trajectory_type.lower() == "spiral":
        test_cameras = _get_spiral_camera_trajectory(
            max_angle,
            n_flyaround_poses,
            camera_elevation,
            hemispherical_radius,
            up,
            camera_focal_length,
        )
    elif trajectory_type.lower() in (
        "circular_lsq_fit",
        "figure_eight",
        "trefoil_knot",
        "figure_eight_knot",
    ):
        train_cameras = train_data.camera
        time = torch.linspace(0, max_angle, n_flyaround_poses + 1)[:n_flyaround_poses]
        test_cameras = generate_eval_video_cameras(
            train_cameras,
            time=time,
            n_eval_cams=n_flyaround_poses,
            trajectory_type=trajectory_type,
            trajectory_scale=trajectory_scale,
            scene_center=scene_center,
            up=up,
            focal_length=None,
            principal_point=torch.zeros(n_flyaround_poses, 2),
            traj_offset_canonical=(0.0, 0.0, traj_offset),
        )
    else:
        raise ValueError(
            f"Unknown `trajectory_type` {trajectory_type.lower()} requested "
        )

    voxel_features = None
    if sample_mode and progressive_sampling_steps_per_render <= 0:
        # obtain the voxel_features by generating using the diffusion model
        voxel_features = model.sample_random_voxel_features()

    progressive_voxel_feature_generator = None
    if progressive_sampling_steps_per_render > 0:
        progressive_voxel_feature_generator = model.sample_random_voxel_features_progressive()

    # the rendering loop:
    preds_total = []
    for n in tqdm(range(n_flyaround_poses), total=n_flyaround_poses):
        # set the first batch camera to the target camera
        for k in ("R", "T", "focal_length", "principal_point"):
            getattr(batch.camera, k)[0] = getattr(test_cameras[n], k)

        # Move to cuda
        net_input = batch.to(device)
        
        # render the predictions
        with torch.no_grad():
            if progressive_voxel_feature_generator is not None:
                for _ in range(progressive_sampling_steps_per_render):
                    try:
                        voxel_features = next(progressive_voxel_feature_generator)
                    except StopIteration:
                        break

            preds = model(
                **{
                    **net_input,
                    "evaluation_mode": EvaluationMode.EVALUATION,
                    "voxel_features": voxel_features,
                }
            )

            # make sure we dont overwrite something
            assert all(k not in preds for k in net_input.keys())
            preds.update(net_input)  # merge everything into one big dict

            # Render the predictions to images
            rendered_pred = _images_from_preds(preds, extract_keys=visualize_preds_keys)
            preds_total.append(rendered_pred)

            # show the preds every 5% of the export iterations
            if visdom_show_preds and (
                n % max(n_flyaround_poses // 20, 1) == 0 or n == n_flyaround_poses - 1
            ):
                assert viz is not None
                _show_predictions(
                    preds_total,
                    sequence_name=batch.sequence_name[0],
                    viz=viz,
                    viz_env=visdom_environment,
                    predicted_keys=visualize_preds_keys,
                )

    if output_video_name is None:
        output_video_name = batch.sequence_name[0]

    logger.info(f"Exporting videos for sequence {sequence_name} ...")
    _generate_prediction_videos(
        preds_total,
        sequence_name=output_video_name,
        viz=viz,
        viz_env=visdom_environment,
        fps=fps,
        video_path=output_video_path,
        resize=video_resize,
        video_frames_dir=output_video_frames_dir,
        predicted_keys=visualize_preds_keys,
    )

    if voxel_features is not None and save_voxel_features:
        logger.info(f"Saving voxel features for sequence {sequence_name} ...")
        output_directory = "/".join(output_video_path.split("/")[:-1])
        torch.save(
            voxel_features,
            os.path.join(output_directory, f"{sequence_name}_voxel_features.pth")
        )


def get_simple_360_camera_trajectory(
    max_angle: float,
    n_flyaround_poses: int,
    camera_elevation: float,
    hemispherical_radius: float,
    up: Tuple[float, float, float],
    camera_focal_length: float,
    canonical_up: Tuple[float, float, float] = (0.0, -1.0, 0.0),
):
    max_angle_deg = 360 * max_angle / (math.pi * 2)
    camera_elevation_deg = 360 * camera_elevation / (math.pi * 2)
    azimuths = torch.linspace(0, max_angle_deg, n_flyaround_poses + 1)[
        :n_flyaround_poses
    ]
    elevations = np.full_like(azimuths, fill_value=camera_elevation_deg)
    hemispherical_radii = np.full_like(azimuths, fill_value=hemispherical_radius)
    rots, trans = [], []
    for azimuth, elevation, radius in zip(azimuths, elevations, hemispherical_radii):
        R, T = look_at_view_transform(
            dist=radius,
            elev=elevation,
            azim=azimuth,
            up=(canonical_up,),
        )
        rots.append(R)
        trans.append(T)
    rots, trans = torch.cat(rots, dim=0), torch.cat(trans, dim=0)
    focal_length = torch.ones(n_flyaround_poses, 1) * camera_focal_length
    principal_point = torch.zeros(n_flyaround_poses, 2)

    # rotates the up vector of the look_at cameras to the desired up direction
    from pytorch3d.transforms import so3_exp_map

    rot_axis_angle = torch.cross(
        torch.FloatTensor(canonical_up),
        torch.FloatTensor(up),
    )
    R_plane = so3_exp_map(rot_axis_angle[None])[0]

    rots_ = torch.bmm(R_plane[None].expand_as(rots).to(rots), rots)
    trans_ = trans

    test_cameras_rot = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=rots_,
        T=trans_,
    )

    return test_cameras_rot


def _get_spiral_camera_trajectory(
    max_angle: float,
    n_flyaround_poses: int,
    camera_elevation: float,
    hemispherical_radius: float,
    up: Tuple[float, float, float],
    camera_focal_length: float,
    canonical_up: Tuple[float, float, float] = (0.0, -1.0, 0.0),
):
    raise NotImplementedError("finish this")


def _get_dummy_test_batch_for_sampling(
    batch_size: int, device: torch.device = torch.device("cpu")
) -> FrameData:
    return FrameData(
        frame_number=None,
        sequence_category=None,
        image_rgb=None,
        camera=PerspectiveCameras(
            R=torch.eye(3, device=device)[None].repeat(batch_size, 1, 1),
            T=torch.zeros(batch_size, 3, device=device),
            principal_point=torch.ones(batch_size, 2, device=device),
            focal_length=torch.zeros(batch_size, 2, device=device),
            device=device,
        ),
        fg_probability=None,
        mask_crop=None,
        depth_map=None,
        sequence_name=None,
        frame_timestamp=None,
    )


def _load_whole_dataset(
    dataset: torch.utils.data.Dataset, idx: Sequence[int], num_workers: int = 10
) -> FrameData:
    load_all_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, idx),
        batch_size=len(idx),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=FrameData.collate,
    )
    return next(iter(load_all_dataloader))


def _make_shaded_from_normals(
    n: torch.Tensor,
    mask: torch.Tensor,
    diffuse_strength: float = 0.3,
    specular_strength: float = 0.1,
    ambient_strength: float = 0.3,
    specular_hardness: float = 10.0,
):
    """
    Note that this calculates a simple shading model with a point light
    coinciding with the camera center location. This simplifies the equation below.
    The specular component is probably wrong ...
    """
    shading = n[:, -1:].clamp(0.0)  # take the last dim as shading
    shading_ambient = (
        shading * diffuse_strength
        + specular_strength * shading**specular_hardness
        + ambient_strength
    ) / (diffuse_strength + specular_strength + ambient_strength)
    return (shading_ambient * mask + (1 - mask)).clamp(0.0, 1.0)


def _images_from_preds(
    preds: Dict[str, Any],
    extract_keys: List[str] = (
        "image_rgb",
        "images_render",
        "fg_probability",
        "masks_render",
        "depths_render",
        "depth_map",
        "_all_source_images",
    ),
) -> Dict[str, torch.Tensor]:
    imout = {}
    for k in extract_keys:
        if k == "_all_source_images" and "image_rgb" in preds and preds["image_rgb"] is not None:
            src_ims = preds["image_rgb"][1:].cpu().detach().clone()
            v = _stack_images(src_ims, None)[None]
        elif k == "_shaded_depth_render" and (
            "normals_render" in preds or "depths_render" in preds
        ):
            if "normals_render" in preds:
                normal_render = preds["normals_render"].detach().clone()
                mask_render = preds["masks_render"][:1].detach().clone()
                v = _make_shaded_from_normals(normal_render, mask_render)

            else:
                from .shaded_depth_render import depth_to_shaded

                depth_render = preds["depths_render"][:1].detach().clone()
                mask_render = preds["masks_render"][:1].detach().clone()
                camera = preds["camera"].clone()[[0]]
                shaded_depth, shaded_depth_mask = depth_to_shaded(
                    depth_render,
                    mask_render,
                    camera,
                    method="mesh",  # 'pointcloud' | 'mesh'
                    ambient=0.05,
                    ambient_color=0.05,
                    K=20,
                    mask_thr=0.5,
                    depth_thr=1e-2,
                    material="medium",
                    bg_color=[1.0, 1.0, 1.0],
                    smoothing_kernel_size=0.005,
                    scene_center=(0.0, 0.0, 0.0),
                    mask_smooth_factor=0.002,
                )
                # white bg
                v = shaded_depth
        else:
            if k not in preds or preds[k] is None:
                logger.debug(f"cant show {k}")
                continue
            v = preds[k].cpu().detach().clone()
        if k.startswith("depth"):
            mask_resize = Fu.interpolate(
                preds["masks_render"],
                size=preds[k].shape[2:],
                mode="nearest",
            )
            v = make_depth_image(preds[k], mask_resize)
            v = v * mask_resize + (1 - mask_resize)
        if v.shape[1] == 1:
            v = v.repeat(1, 3, 1, 1)
        imout[k] = v.detach().cpu()

    return imout


def _stack_images(ims: torch.Tensor, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    ba = ims.shape[0]
    H = int(np.ceil(np.sqrt(ba)))
    W = H
    n_add = H * W - ba
    if n_add > 0:
        ims = torch.cat((ims, torch.zeros_like(ims[:1]).repeat(n_add, 1, 1, 1)))

    ims = ims.view(H, W, *ims.shape[1:])
    cated = torch.cat([torch.cat(list(row), dim=2) for row in ims], dim=1)
    if size is not None:
        cated = Fu.interpolate(cated[None], size=size, mode="bilinear")[0]
    return cated.clamp(0.0, 1.0)


def _show_predictions(
    preds: List[Dict[str, Any]],
    sequence_name: str,
    viz: "Visdom",
    viz_env: str = "visualizer",
    predicted_keys: Sequence[str] = (
        "images_render",
        "masks_render",
        "depths_render",
        "_all_source_images",
    ),
    n_samples=10,
    one_image_width=200,
) -> None:
    """Given a list of predictions visualize them into a single image using visdom."""
    assert isinstance(preds, list)

    pred_all = []
    # Randomly choose a subset of the rendered images, sort by ordr in the sequence
    n_samples = min(n_samples, len(preds))
    pred_idx = sorted(random.sample(list(range(len(preds))), n_samples))
    for predi in pred_idx:
        # Make the concatentation for the same camera vertically
        pred_all.append(
            torch.cat(
                [
                    torch.nn.functional.interpolate(
                        preds[predi][k].cpu(),
                        scale_factor=one_image_width / preds[predi][k].shape[3],
                        mode="bilinear",
                    ).clamp(0.0, 1.0)
                    for k in predicted_keys
                    if k in preds[predi]
                ],
                dim=2,
            )
        )
    # Concatenate the images horizontally
    pred_all_cat = torch.cat(pred_all, dim=3)[0]
    viz.image(
        pred_all_cat,
        win="show_predictions",
        env=viz_env,
        opts={"title": f"pred_{sequence_name}"},
    )


def _generate_prediction_videos(
    preds: List[Dict[str, Any]],
    sequence_name: str,
    viz: Optional["Visdom"] = None,
    viz_env: str = "visualizer",
    predicted_keys: Sequence[str] = (
        "images_render",
        "masks_render",
        "depths_render",
        "_all_source_images",
    ),
    fps: int = 20,
    video_path: str = "/tmp/video",
    video_frames_dir: Optional[str] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> None:
    """Given a list of predictions create and visualize rotating videos of the
    objects using visdom.
    """

    # make sure the target video directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # init a video writer for each predicted key
    vws = {}
    for k in predicted_keys:
        cache_dir = (
            None
            if video_frames_dir is None
            else os.path.join(video_frames_dir, f"{sequence_name}_{k}")
        )
        vws[k] = VideoWriter(
            fps=fps,
            out_path=f"{video_path}_{sequence_name}_{k}.mp4",
            cache_dir=cache_dir,
        )

    for rendered_pred in tqdm(preds):
        for k in predicted_keys:
            if k not in rendered_pred:
                continue
            vws[k].write_frame(
                rendered_pred[k][0].clip(0.0, 1.0).detach().cpu().numpy(),
                resize=resize,
            )

    for k in predicted_keys:
        if k not in preds[0]:
            continue
        vws[k].get_video()
        logger.info(f"Generated {vws[k].out_path}.")
        if viz is not None:
            viz.video(
                videofile=vws[k].out_path,
                env=viz_env,
                win=k,  # we reuse the same window otherwise visdom dies
                opts={"title": sequence_name + " " + k},
            )
