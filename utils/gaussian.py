import torch
from plyfile import PlyData, PlyElement
import os
import numpy as np

# import open3d
import math


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1] * _features_dc.shape[2]):
        l.append("f_dc_{}".format(i))
    for i in range(_features_rest.shape[1] * _features_rest.shape[2]):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(_scaling.shape[1]):
        l.append("scale_{}".format(i))
    for i in range(_rotation.shape[1]):
        l.append("rot_{}".format(i))
    return l


def write_gaussian_feature_to_ply(gaussian_feature, save_path):
    # gaussian Nx59 feature 3+1+3+4+48
    _xyz = gaussian_feature[:, :3]
    _opcaity = gaussian_feature[:, 3:4]
    _opcaity = inverse_sigmoid(_opcaity)
    _scaling = gaussian_feature[:, 4:7]
    _scaling = torch.log(_scaling)  # back to log space
    _rotation = gaussian_feature[:, 7:11]
    _features_dc = gaussian_feature[:, 11:14].reshape(-1, 3, 1)
    _features_rest = torch.zeros(_features_dc.shape[0], 3, 15)

    _features_dc = (
        torch.tensor(_features_dc, dtype=torch.float, device="cpu")
        .transpose(1, 2)
        .contiguous()
        .requires_grad_(False)
    )

    _features_rest = (
        torch.tensor(_features_rest, dtype=torch.float, device="cpu")
        .transpose(1, 2)
        .contiguous()
        .requires_grad_(False)
    )

    save_ply_tensor(
        _xyz, _features_dc, _features_rest, _opcaity, _scaling, _rotation, save_path
    )


def unnormalize_gaussians(
    original_gaussians, vis_gaussians, full_rebuild_gaussian, scale_c, scale_m, config
):
    if "opacity" in config.dataset.train.others.norm_attribute:
        original_gaussians[..., 3] = (1 + (original_gaussians[..., 3])) / 2
        vis_gaussians[..., 3] = (1 + vis_gaussians[..., 3]) / 2
        full_rebuild_gaussian[..., 3] = (
            1 + full_rebuild_gaussian[..., 3].clip(-1, 1) + 1e-9
        ) / 2

    if "scale" in config.dataset.train.others.norm_attribute:
        original_gaussians[..., 4:7] = original_gaussians[..., 4:7] * scale_m.unsqueeze(
            -1
        ).unsqueeze(-1).to(original_gaussians.device) + scale_c.unsqueeze(1).repeat(
            1, original_gaussians.shape[1], 1
        ).to(
            original_gaussians.device
        )
        vis_gaussians[..., 4:7] = vis_gaussians[..., 4:7] * scale_m.unsqueeze(
            -1
        ).unsqueeze(-1).to(vis_gaussians.device) + scale_c.unsqueeze(1).repeat(
            1, vis_gaussians.shape[1], 1
        ).to(
            vis_gaussians.device
        )
        full_rebuild_gaussian[..., 4:7] = full_rebuild_gaussian[..., 4:7].clip(
            -1, 1
        ) * scale_m.unsqueeze(-1).unsqueeze(-1).to(
            full_rebuild_gaussian.device
        ) + scale_c.unsqueeze(
            1
        ).repeat(
            1, full_rebuild_gaussian.shape[1], 1
        ).to(
            full_rebuild_gaussian.device
        )

    if "sh" in config.dataset.train.others.norm_attribute:
        original_gaussians[..., 11:14] = (
            original_gaussians[..., 11:14] * math.sqrt(3) / (2 * 0.28209479177387814)
        )
        vis_gaussians[..., 11:14] = (
            vis_gaussians[..., 11:14] * math.sqrt(3) / (2 * 0.28209479177387814)
        )
        full_rebuild_gaussian[..., 11:14] = (
            full_rebuild_gaussian[..., 11:14] * math.sqrt(3) / (2 * 0.28209479177387814)
        )

    return original_gaussians, vis_gaussians, full_rebuild_gaussian


def save_ply_tensor(
    _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation, path
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = _xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = (
        _features_dc.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    f_rest = (
        _features_rest.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    opacities = _opacity.detach().cpu().numpy()
    scale = _scaling.detach().cpu().numpy()
    rotation = _rotation.detach().cpu().numpy()

    attributes = construct_list_of_attributes(
        _features_dc, _features_rest, _scaling, _rotation
    )
    dtype_full = [(attribute, "f4") for attribute in attributes]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5
