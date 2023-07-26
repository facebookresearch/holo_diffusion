# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as Fu

import pytorch3d as pt3d

from .mesh_render import mesh_render


def _smooth_depth(g, m, k=1.0 / 50.0):
    if isinstance(k, float):
        k = int(np.ceil(k * np.sqrt(g.shape[2] ** 2 + g.shape[3] ** 2)))
    if m is None:
        m = torch.ones_like(g[:, :1])
    gm = torch.cat((g, m.float()), dim=1)
    gma = torch.nn.functional.avg_pool2d(gm, 2 * k + 1, padding=k, stride=1)
    g, m = gma.split([gma.shape[1] - 1, 1], dim=1)
    return g / m.clamp(1e-4)


def _depth_lap_filter(d, m, sigma=1.0):
    ba = d.shape[0]
    d_smooth = _smooth_depth(d, None, 0.03)
    df = (d - d_smooth).abs()
    # cut_off the outliers
    m_bin = (m > 0.0).float()
    mu_df = pt3d.ops.wmean(df.view(ba, -1, 1), m_bin.view(ba, -1)).view(-1)
    std_df = (
        pt3d.ops.wmean((df.view(ba, -1, 1) - mu_df) ** 2, m_bin.view(ba, -1))
        .view(-1)
        .clamp(1e-3)
        .sqrt()
    )
    m = (
        ((df - mu_df.view(ba, 1, 1, 1)) ** 2).sqrt()
        < (std_df.view(ba, 1, 1, 1) * sigma)
    ).float() * m_bin
    return m


def grid_pcl_to_shaded_mesh(
    cameras,
    pcl_grid,
    masks_ok,
    K=10,
    material="high_contrast",  # high_contrast | medium
    bg_color=[0.0, 0.0, 0.0],
    max_render_size=256,
    scene_center=[0.0, 0.0, 0.0],
    mask_smooth_factor=0.001,
):
    ba, h, w, _ = pcl_grid.shape

    assert ba == 1, "only works for one sample now!"

    v, f, tex = [], [], []
    for pcl_grid_, mask in zip(pcl_grid, masks_ok):
        # smooth the grid a bir first
        mesh_verts, mesh_faces, _ = get_grid_mesh(
            pcl_grid_.permute(2, 0, 1), mask[0].float()
        )

        if len(mesh_faces) < 10:
            print("Not enough faces!")
            return (
                torch.zeros(ba, 3, h, w).type_as(pcl_grid),
                torch.zeros(ba, 1, h, w).type_as(pcl_grid),
            )

        v_ = mesh_verts.permute(1, 2, 0).view(-1, 3)
        v.append(v_)
        f.append(mesh_faces.long())
        tex.append(torch.ones_like(v[-1]))

    meshes = pt3d.structures.Meshes(
        verts=v, faces=f, textures=pt3d.renderer.TexturesVertex(tex)
    )

    if material == "high_contrast":
        colors = dict(
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((2.0, 2.0, 2.0),),
            specular_color=((1.0, 1.0, 0.9),),
            shininess=256,
        )
    elif material == "medium":
        colors = dict(
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((1.0, 1.0, 1.0),),
            specular_color=((1.0, 1.0, 0.9),),
            shininess=128,
        )
    else:
        raise ValueError(material)

    if max(h, w) > max_render_size:
        rh, rw = [round(s * max_render_size / max(h, w)) for s in [h, w]]
        print(f"Resizing mesh render to {rh}x{rw}.")
    else:
        rh, rw = h, w

    shaded, render_mask, depth_rendered = mesh_render(
        cameras,
        meshes,
        [rh, rw],
        blur_sigma=1e-4,
        topk_sigma=1e-4,
        topk=K,
        feature_render=False,
        orthographic=False,
        lights=None,
        background_color=bg_color,
        visualize=False,
        min_depth=1e-3,
        scene_center=scene_center,
        **colors,
    )

    if rh != h or rw != w:
        shaded, render_mask, depth_rendered = [
            Fu.interpolate(v, size=[h, w], mode=mode)
            for v, mode in zip(
                (shaded, render_mask, depth_rendered),
                ("bilinear", "nearest", "nearest"),
            )
        ]

    render_mask = _depth_lap_filter(depth_rendered, render_mask, sigma=1.0)

    # smooth the render mask to remove artifacts
    render_mask = _smooth_depth((render_mask > 0.0).float(), None, mask_smooth_factor)

    # cut off the far pixels
    return shaded, render_mask


def depth_to_shaded(
    depths,
    masks,
    cameras_pt3,
    method="mesh",  # 'pointcloud' | 'mesh'
    ambient=0.05,
    ambient_color=0.05,
    K=20,
    mask_thr=0.5,
    depth_thr=1e-2,
    material="medium",
    bg_color=[0.0, 0.0, 0.0],
    smoothing_kernel_size=0.005,
    scene_center=(0.0, 0.0, 0.0),
    mask_smooth_factor=0.005,
):
    ba, _, H, W = depths.shape

    masks_ok = (masks > mask_thr) * (depths > depth_thr)

    if masks_ok.sum() < 3:
        shaded = (depths * 0).repeat(1, 3, 1, 1)
        shaded_mask = torch.zeros_like(depths)
        return shaded, shaded_mask

    depths = _smooth_depth(depths, masks_ok, k=smoothing_kernel_size)

    cameras_pt3_trivial = cameras_pt3.clone()
    cameras_pt3_trivial.R = torch.eye(3)[None].to(cameras_pt3.R).repeat(ba, 1, 1)
    cameras_pt3_trivial.T = cameras_pt3_trivial.T * 0.0

    # unproject the depth
    ray_bundle = pt3d.renderer.NDCGridRaysampler(
        image_width=W,
        image_height=H,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )(cameras_pt3_trivial)._replace(lengths=depths[:, 0][..., None])

    pcl_grid = pt3d.renderer.ray_bundle_to_ray_points(ray_bundle).view(ba, H, W, 3)

    if method == "pointcloud":
        # with the light coinciding with the camera center, the shading is
        # just the 3rd coord of the normal
        shaded = grid_pcl_to_shaded(pcl_grid, masks_ok, K=K)
        shaded = ambient * ambient_color + (1 - ambient) * shaded
        shaded = shaded.clamp(0.0, 1.0)
        shaded_mask = torch.ones_like(shaded[:, :1])
    elif method == "mesh":
        shaded, shaded_mask = grid_pcl_to_shaded_mesh(
            cameras_pt3_trivial,
            pcl_grid,
            masks_ok,
            K=K,
            material=material,
            bg_color=bg_color,
            scene_center=scene_center,
            mask_smooth_factor=mask_smooth_factor,
        )
    else:
        raise ValueError(method)

    return shaded, shaded_mask


def grid_pcl_to_shaded(pcl_grid, masks_ok, K=20, max_size=200):
    orig_size = None
    if max(pcl_grid.shape[2:]) > max_size:
        orig_size = list(pcl_grid.shape[1:3])
        scl = max_size / max(pcl_grid.shape[2:])
        pcl_grid = torch.nn.functional.interpolate(
            pcl_grid.permute(0, 3, 1, 2), scale_factor=scl, mode="bilinear"
        ).permute(0, 2, 3, 1)
        masks_ok = torch.nn.functional.interpolate(
            masks_ok.float(),
            size=list(pcl_grid.shape[1:3]),
            mode="nearest",
        )

    masks_ok = masks_ok > 0.5

    ba, H, W, _ = pcl_grid.shape

    ba = pcl_grid.shape[0]

    pcl = pt3d.structures.Pointclouds(
        [p.view(-1, 3)[m.view(-1) > 0.5] for p, m in zip(pcl_grid, masks_ok)]
    )

    normals = pt3d.ops.estimate_pointcloud_normals(
        pcl,
        neighborhood_size=K,
        disambiguate_directions=True,
    )

    # all have to point to the camera
    normals = normals * normals[..., 2:].sign()

    shaded = torch.zeros(ba, H * W, device=pcl_grid.device, dtype=torch.float32)
    shaded[masks_ok.view(ba, -1)] = normals[..., -1][:, None]

    shaded = shaded.view(ba, 1, H, W).repeat(1, 3, 1, 1)

    if orig_size is not None:
        shaded = torch.nn.functional.interpolate(
            shaded, size=orig_size, mode="bilinear"
        )

    return shaded


def get_grid_mesh(verts, mask):
    _, he, wi = verts.shape
    idx = torch.arange(he * wi).reshape(he, wi)
    faces_quad = torch.nn.functional.unfold(idx[None, None].float(), 2).long()[0]
    masks_quad = torch.nn.functional.unfold((mask[None, None] > 0.5).float(), 2).long()[
        0
    ]
    ok = masks_quad.sum(0) == 4

    faces_quad = faces_quad[:, ok.cpu()]

    # quads to tris
    # faces_tri = torch.cat((faces_quad[:3], faces_quad[1:]), dim=1)

    tri1 = faces_quad[:3].T
    tri2 = faces_quad[1:].T[:, [0, 2, 1]]

    tri1 = tri1[:, [0, 2, 1]]
    tri2 = tri2[:, [0, 2, 1]]

    faces_tri = torch.cat((tri1, tri2), dim=1).reshape(-1, 3).t()
    faces_tri = faces_tri.to(verts.device)

    # faces_tri = torch.cat((faces_quad[:3], faces_quad[1:]), dim=0).reshape(3, -1)
    # return verts.view(3, -1).permute(1,0).float(), faces_tri.t().long(), mask.view(
    return verts.view(3, he, wi).float(), faces_tri.t().long(), mask.view(he, wi)
