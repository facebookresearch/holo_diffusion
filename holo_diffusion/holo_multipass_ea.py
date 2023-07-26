# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from pytorch3d.implicitron.tools.config import registry
from pytorch3d.implicitron.models.renderer.base import EvaluationMode
from pytorch3d.implicitron.models.renderer.multipass_ea import (
    MultiPassEmissionAbsorptionRenderer,
)


@registry.register
class HoloMultiPassEmissionAbsorptionRenderer(  # pyre-ignore: 13
    MultiPassEmissionAbsorptionRenderer
):
    """
    MultiPassEmissionAbsorptionRenderer which also passes to the implicit function
    the number of the rendering pass.
    MultiPassEmissionAbsorptionRenderer doc:
    Implements the multi-pass rendering function, in particular,
    with emission-absorption ray marching used in NeRF [1]. First, it evaluates
    opacity-based ray-point weights and then optionally (in case more implicit
    functions are given) resamples points using importance sampling and evaluates
    new weights.
    During each ray marching pass, features, depth map, and masks
    are integrated: Let o_i be the opacity estimated by the implicit function,
    and d_i be the offset between points `i` and `i+1` along the respective ray.
    Ray marching is performed using the following equations:
    ```
    ray_opacity_n = cap_fn(sum_i=1^n cap_fn(d_i * o_i)),
    weight_n = weight_fn(cap_fn(d_i * o_i), 1 - ray_opacity_{n-1}),
    ```
    and the final rendered quantities are computed by a dot-product of ray values
    with the weights, e.g. `features = sum_n(weight_n * ray_features_n)`.
    By default, for the EA raymarcher from [1] (
        activated with `self.raymarcher_class_type="EmissionAbsorptionRaymarcher"`
    ):
        ```
        cap_fn(x) = 1 - exp(-x),
        weight_fn(x) = w * x.
        ```
    Note that the latter can altered by changing `self.raymarcher_class_type`,
    e.g. to "CumsumRaymarcher" which implements the cumulative-sum raymarcher
    from NeuralVolumes [2].
    Settings:
        n_pts_per_ray_fine_training: The number of points sampled per ray for the
            fine rendering pass during training.
        n_pts_per_ray_fine_evaluation: The number of points sampled per ray for the
            fine rendering pass during evaluation.
        stratified_sampling_coarse_training: Enable/disable stratified sampling in the
            refiner during training. Only matters if there are multiple implicit
            functions (i.e. in GenericModel if num_passes>1).
        stratified_sampling_coarse_evaluation: Enable/disable stratified sampling in
            the refiner during evaluation. Only matters if there are multiple implicit
            functions (i.e. in GenericModel if num_passes>1).
        append_coarse_samples_to_fine: Add the fine ray points to the coarse points
            after sampling.
        density_noise_std_train: Standard deviation of the noise added to the
            opacity field.
        return_weights: Enables returning the rendering weights of the EA raymarcher.
            Setting to `True` can lead to a prohibitivelly large memory consumption.
        raymarcher_class_type: The type of self.raymarcher corresponding to
            a child of `RaymarcherBase` in the registry.
        raymarcher: The raymarcher object used to convert per-point features
            and opacities to a feature render.
    References:
        [1] Mildenhall, Ben, et al. "Nerf: Representing Scenes as Neural Radiance
            Fields for View Synthesis." ECCV 2020.
        [2] Lombardi, Stephen, et al. "Neural Volumes: Learning Dynamic Renderable
            Volumes from Images." SIGGRAPH 2019.
    """

    # set this default to 1.0 by hard!
    density_noise_std_train: float = 1.0

    def _run_raymarcher(
        self,
        ray_bundle,
        implicit_functions,
        prev_stage,
        evaluation_mode,
        pass_number=0,
    ):
        density_noise_std = (
            self.density_noise_std_train
            if evaluation_mode == EvaluationMode.TRAINING
            else 0.0
        )

        if_output = implicit_functions[0](
            ray_bundle=ray_bundle, pass_number=pass_number
        )
        output = self.raymarcher(
            *if_output,
            ray_lengths=ray_bundle.lengths,
            density_noise_std=density_noise_std,
        )
        output.prev_stage = prev_stage

        weights = output.weights

        if "normals" in output.aux:
            # assign normals to output by rendering them
            output.normals = (output.aux.pop("normals") * weights[..., None]).sum(
                dim=-2
            )

        if not self.return_weights:
            output.weights = None

        # we may need to make a recursive call
        if len(implicit_functions) > 1:
            fine_ray_bundle = self._refiners[evaluation_mode](ray_bundle, weights)
            output = self._run_raymarcher(
                fine_ray_bundle,
                implicit_functions[1:],
                output,
                evaluation_mode,
                pass_number=pass_number + 1,
            )

        return output
