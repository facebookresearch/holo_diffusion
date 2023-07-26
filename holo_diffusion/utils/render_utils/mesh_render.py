# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as Fu

from pytorch3d import renderer
from pytorch3d.structures import Meshes
from typing import Tuple


def _signed_clamp(x, eps):
    sign = x.sign() + (x == 0.0).type_as(x)
    x_clamp = sign * torch.clamp(x.abs(), eps)
    return x_clamp


def mesh_render(
    cameras,
    mesh,
    render_size: Tuple[int, int],
    blur_sigma=1e-4,
    topk_sigma=1e-4,
    topk=50,
    feature_render=False,
    shininess=128,
    orthographic=False,
    lights=None,
    ambient_color=((1.0, 1.0, 1.0),),
    diffuse_color=((1.0, 1.0, 1.0),),
    specular_color=((0.0, 0.0, 0.0),),
    background_color=[0.0, 0.0, 0.0],
    visualize=False,
    min_depth=1e-3,
    scene_center=[0.0, 0.0, 0.0],
):
    raster_settings_soft = renderer.RasterizationSettings(
        # image_size=int(max(render_size)),
        image_size=int(max(render_size)),
        blur_radius=math.log(1.0 / 1e-4 - 1.0) * blur_sigma,
        faces_per_pixel=topk,
        perspective_correct=False,
    )

    # get the feature dim
    assert mesh.textures is not None, "has to be textured"

    # Transform to camera view and clamp the depth so we dont
    # divide by small numbers.
    verts_t = cameras.get_world_to_view_transform().transform_points(
        mesh.verts_padded(), eps=min_depth  # the eps here makes sure we dont div0
    )
    # make sure no point has abs(depth) <= min_depth
    verts_t = torch.cat(
        (verts_t[..., :-1], _signed_clamp(verts_t[..., -1:], min_depth)), dim=-1
    )
    mesh = Meshes(verts=verts_t, faces=mesh.faces_padded(), textures=mesh.textures)

    # convert the camera to a trivial one
    cameras = cameras.clone()
    cameras.R = torch.eye(3, device=cameras.device)[None].repeat(verts_t.shape[0], 1, 1)
    cameras.T *= 0.0

    if lights is None:
        lloc = [scene_center]
        lights = renderer.PointLights(device=mesh.device, location=lloc)

    if feature_render:
        blend_params_feature = renderer.BlendParams(
            sigma=topk_sigma,
            gamma=1e-4,
            background_color=[0.0] * featdim,
        )
        featdim = mesh.textures.verts_features_packed().shape[-1]
        shader = FeatureShader(
            device=cameras.device,
            cameras=cameras,
            blend_params=blend_params_feature,
        )

    else:
        blend_params = renderer.BlendParams(
            sigma=topk_sigma,
            gamma=1e-4,
            background_color=background_color,
        )
        # Differentiable soft renderer using per vertex RGB colors for texture
        ambient_materials = renderer.Materials(
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            shininess=shininess,
            device=mesh.device,
        )
        shader = renderer.SoftGouraudShader(
            device=cameras.device,
            cameras=cameras,
            lights=lights,
            materials=ambient_materials,
            blend_params=blend_params,
        )

    rasterizer = renderer.MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_soft,
    )

    fragments = rasterizer(mesh, cameras=cameras, lights=lights)
    blend_params_depth = renderer.BlendParams(
        sigma=topk_sigma,
        gamma=1e-4,
        background_color=[0.0],
    )
    depth = softmax_depth_blend(
        fragments,
        blend_params_depth,
        znear=getattr(cameras, "znear", 0.01),
        zfar=getattr(cameras, "zfar", 1000.0),
    )
    images = shader(fragments, mesh, cameras=cameras, lights=lights)
    rendered_blob = torch.cat((images, depth), dim=-1).permute(0, 3, 1, 2)

    rendered_blob = Fu.interpolate(rendered_blob, size=render_size, mode="bilinear")
    data_rendered, render_mask, depth_rendered = rendered_blob.split(
        [rendered_blob.shape[1] - 2, 1, 1], dim=1
    )

    if visualize:
        from tools.vis_utils import get_visdom_connection
        from tools.functions import select_cameras
        from pytorch3d.vis.plotly_vis import plot_scene

        viz = get_visdom_connection()
        scene_dict = {
            "1": {
                "meshes": mesh,
                "cameras": cameras,
            }
        }
        scene = plot_scene(
            scene_dict,
            pointcloud_max_points=5000,
            pointcloud_marker_size=1.5,
            camera_scale=1.0,
        )
        viz.plotlyplot(scene, env="mesh_dbg", win="scenes")
        import pdb

        pdb.set_trace()

    return data_rendered, render_mask, depth_rendered


from pytorch3d.ops import interpolate_face_attributes


class FeatureShader(torch.nn.Module):
    """
    Renders a feature map without any complicated shading.
    """

    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = (
            blend_params if blend_params is not None else renderer.BlendParams()
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_feature_blend(
            texels, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images


def softmax_depth_blend(
    fragments,
    blend_params,
    znear: float = 1.0,
    zfar: float = 100,
):
    depth = softmax_feature_blend(
        fragments.zbuf[..., None], fragments, blend_params, znear=znear, zfar=zfar
    )
    return depth[..., :1]


def softmax_feature_blend(
    colors, fragments, blend_params, znear: float = 1.0, zfar: float = 100
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.
    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction
    Returns:
        RGBA pixel_colors: (N, H, W, 4)
    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """

    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    dim = colors.shape[-1]
    pixel_colors = torch.ones(
        (N, H, W, dim + 1), dtype=colors.dtype, device=colors.device
    )
    background = blend_params.background_color
    if not torch.is_tensor(background):
        background = torch.tensor(background, dtype=torch.float32, device=device)
    else:
        background = background.to(device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[20]: Argument `max` expected.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background

    pixel_colors[..., :dim] = (weighted_colors + weighted_background) / denom
    pixel_colors[..., dim] = 1.0 - alpha

    return pixel_colors
