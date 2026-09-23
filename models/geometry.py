"""SO(3) and native-20 forward kinematics. Units are metres and radians."""
from __future__ import annotations

import torch
from torch import Tensor

PARENTS = (-1, 0, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18)
TIPS = (3, 7, 11, 15, 19)
NON_THUMB = tuple(range(4, 20))


def skew(v: Tensor) -> Tensor:
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def so3_exp(v: Tensor) -> Tensor:
    """Rodrigues map with finite derivatives at zero (no sqrt-at-zero branch)."""
    theta2 = v.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1e-12).sqrt()
    a = torch.where(theta2 < 1e-6, 1 - theta2 / 6 + theta2.square() / 120,
                    torch.sin(theta) / theta)
    b = torch.where(theta2 < 1e-6, .5 - theta2 / 24 + theta2.square() / 720,
                    (1 - torch.cos(theta)) / theta2.clamp_min(1e-12))
    k = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def orthonormalize(r: Tensor) -> Tensor:
    """Stable column re-orthogonalization to limit long-stream float32 drift."""
    u = torch.nn.functional.normalize(r[..., 0], dim=-1)
    v = r[..., 1] - (r[..., 1] * u).sum(-1, keepdim=True) * u
    v = torch.nn.functional.normalize(v, dim=-1)
    w = torch.cross(u, v, dim=-1)
    return torch.stack((u, v, w), -1)


def rotation_distance(a: Tensor, b: Tensor) -> Tensor:
    """Geodesic angle, with a small numerical floor at exact identity."""
    r = a.transpose(-1, -2) @ b
    cosine = (r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))


def fk(local_rotations: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor]:
    """Standard joint-frame FK: p_j=p_parent+R_parent offset_j; R_j=R_parent R_local_j.

    local_rotations: [B,20,3,3]; offsets: [B,20,3] or [20,3].
    Returns root-relative positions (in reference axes) and global joint frames.
    Terminal joint rotations do not affect positions and have no positional GT.
    """
    if local_rotations.shape[-3:] != (20, 3, 3) or offsets.shape[-2:] != (20, 3):
        raise ValueError('Expected the audited native-20 topology.')
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(local_rotations.shape[0], -1, -1)
    positions = [torch.zeros_like(offsets[:, 0])]
    frames = [local_rotations[:, 0]]
    for j, parent in enumerate(PARENTS[1:], 1):
        positions.append(positions[parent] + (frames[parent] @ offsets[:, j, :, None]).squeeze(-1))
        frames.append(frames[parent] @ local_rotations[:, j])
    return torch.stack(positions, 1), torch.stack(frames, 1)


def joint_variance(positions: Tensor, frames: Tensor, tangent_variance: Tensor) -> Tensor:
    """First-order diagonal Cartesian variance from independent joint tangent blocks.

    This deliberately omits cross-joint/velocity/calibration covariance. It is not
    an augmented EKF posterior. Variance must be empirically calibrated.
    """
    values = []
    for j in range(20):
        var = torch.zeros_like(positions[:, j])
        ancestor = PARENTS[j]
        while ancestor >= 0:
            arm = positions[:, j] - positions[:, ancestor]
            axes = frames[:, ancestor].transpose(-1, -2)
            jac = torch.cross(axes, arm[:, None].expand_as(axes), dim=-1)
            var = var + (jac.square() * tangent_variance[:, ancestor, :, None]).sum(-2)
            ancestor = PARENTS[ancestor]
        values.append(var + 1e-6)  # 1 mm numerical/label floor in Cartesian space
    return torch.stack(values, 1)


def attached_markers(positions: Tensor, frames: Tensor, bones: Tensor, offsets: Tensor) -> Tensor:
    if bones.ndim == 1:
        bones = bones.unsqueeze(0).expand(positions.shape[0], -1)
    b = torch.arange(positions.shape[0], device=positions.device)[:, None]
    return positions[b, bones] + (frames[b, bones] @ offsets[..., None]).squeeze(-1)


def project(points: Tensor, intrinsics: Tensor) -> tuple[Tensor, Tensor]:
    """Undistorted full-image pixels. Never compare with distorted/crop pixels."""
    homogeneous = points @ intrinsics.transpose(-1, -2)
    positive = points[..., 2] > 1e-3
    return homogeneous[..., :2] / homogeneous[..., 2:].clamp_min(1e-3), positive
