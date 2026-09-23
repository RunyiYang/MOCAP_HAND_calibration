"""Masked supervision. Missing labels contribute zero, never fabricated targets."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F
from models.geometry import TIPS, attached_markers, project, rotation_distance

DEFAULT_WEIGHTS = dict(joints=1., root=1., markers=1., rotation=.1, velocity=.02,
                       reprojection=.01, observation=.01, calibration=.001, nll=0., rollout=.2)


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(value)
    # where prevents 0*nan from silently contaminating a masked loss.
    return torch.where(weight > 0, value * weight, torch.zeros_like(value)).sum() / weight.sum().clamp_min(1)


def robust(a: Tensor, b: Tensor, scale: float = .01) -> Tensor:
    return F.smooth_l1_loss(a / scale, b / scale, reduction='none')


def supervised_loss(outputs: dict, batch: dict, weights: dict | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    weights = DEFAULT_WEIGHTS | (weights or {})
    unknown = set(weights) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f'Unsupported losses: {unknown}. Raw-IMU/depth-surface losses are not implemented.')
    target, inp = batch['targets'], batch['inputs']
    pred = outputs['joints']
    if not target:
        raise ValueError('Training requires a separate targets object')
    joint_weight = target['joint_valid'].float().clone()
    joint_weight[..., list(TIPS)] *= 2
    terms = {'joints': masked_mean(robust(pred, target['joints']), joint_weight)}
    terms['root'] = masked_mean(robust(outputs['root'], target['root']), target['root_valid'])
    terms['rotation'] = pred.sum() * 0
    if target['rotation_valid'].any():
        terms['rotation'] = masked_mean(rotation_distance(outputs['rotations'], target['rotations']) / .08726646,
                                        target['rotation_valid'])
    terms['markers'] = pred.sum() * 0
    if target['marker_valid'].any():
        b, t = pred.shape[:2]
        bones = batch['marker_bones']
        off = batch['marker_offsets'].expand(b, -1, -1)[:, None].expand(-1, t, -1, -1)
        markers = attached_markers(pred.reshape(b*t, 20, 3), outputs['frames'].reshape(b*t, 20, 3, 3),
                                   bones, off.reshape(b*t, -1, 3)).reshape(b, t, -1, 3)
        terms['markers'] = masked_mean(robust(markers, target['markers']), target['marker_valid'])
    terms['velocity'] = pred.sum() * 0
    if pred.shape[1] > 1:
        dt = inp['dt'][:, 1:, None, None]
        vp = (pred[:, 1:] - pred[:, :-1]) / dt
        vt = (target['joints'][:, 1:] - target['joints'][:, :-1]) / dt
        valid = target['joint_valid'][:, 1:] & target['joint_valid'][:, :-1]
        terms['velocity'] = masked_mean(robust(vp, vt, .1), valid)
    terms['reprojection'] = pred.sum() * 0
    if target['uv_valid'].any():
        uv, positive = project(outputs['world_joints'], target['intrinsics'])
        # Invalid negative-depth predictions are penalized, not simply dropped.
        terms['reprojection'] = masked_mean(robust(uv, target['uv'], 5), target['uv_valid'])
        terms['reprojection'] = terms['reprojection'] + masked_mean(
            F.relu(.001 - outputs['world_joints'][..., 2]) / .01, target['uv_valid'])
    terms['observation'] = pred.sum() * 0
    terms['calibration'] = pred.sum() * 0
    if 'glove_reconstruction' in outputs:
        terms['observation'] = masked_mean(robust(outputs['glove_reconstruction'], inp['glove']), inp['glove_valid'])
    if 'bias_increment' in outputs:
        terms['calibration'] = (outputs['bias_increment'] / inp['dt'][..., None, None] / .01).square().mean()
    variance = (outputs['joint_variance'] + target['label_variance']).clamp_min(1e-6)
    # Divide by position scale squared so log variance is dimensionless.
    terms['nll'] = masked_mean(.5 * ((pred-target['joints']).square() / variance + (variance/.0001).log()),
                               target['joint_valid'])
    total = sum(weights.get(k, 0) * value for k, value in terms.items())
    return total, terms


def forecast_loss(outputs: dict, target: dict) -> Tensor:
    return masked_mean(robust(outputs['joints'], target['joints']), target['joint_valid'])
