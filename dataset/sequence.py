"""Strict NPZ sequence contract. Observations and supervision are separate objects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = 'handcalib.sequence_manifest.v1'
INPUT_NAMES = ('dt', 'glove', 'glove_valid', 'glove_age', 'rgb_features', 'rgb',
               'rgb_valid', 'geometry', 'depth', 'depth_mask', 'depth_valid')
TARGET_NAMES = ('joints', 'joint_valid', 'root', 'root_valid', 'rotations', 'rotation_valid',
                'markers', 'marker_valid', 'uv', 'uv_valid', 'intrinsics', 'label_variance')


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: str | Path, *, allow_synthetic: bool = False) -> dict[str, Any]:
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA or not data.get('sequences'):
        raise ValueError('Missing or unsupported sequence manifest')
    group_key = data.get('split_group', 'session_id')
    if group_key not in {'session_id', 'subject_id'}:
        raise ValueError('split_group must be session_id or subject_id')
    seen_ids, group_splits, source_splits, hashes = set(), {}, {}, {}
    for rec in data['sequences']:
        for key in ('id', 'path', 'sha256', 'split', 'subject_id', 'session_id', 'source_group',
                    'units', 'frame', 'calibration_source', 'timing_verified', 'geometry_verified',
                    'input_uses_mocap', 'crop_source', 'synthetic'):
            if key not in rec:
                raise ValueError(f'Missing manifest field {key}')
        if rec['split'] not in {'train', 'val', 'test'} or rec['id'] in seen_ids:
            raise ValueError('Invalid split or duplicate sequence id')
        seen_ids.add(rec['id'])
        if rec['synthetic'] and not allow_synthetic:
            raise ValueError('Synthetic fixtures require --allow-synthetic; not benchmark data')
        if rec['units'] != 'm' or rec['frame'] != 'camera_axes_root_relative':
            raise ValueError('Expected metres and fixed camera axes; root-normalized without per-frame rotation')
        if rec['input_uses_mocap'] is not False or rec['crop_source'] not in {'sensor_only', 'synthetic'}:
            raise ValueError('MoCap-conditioned inputs or crops are forbidden')
        if rec['timing_verified'] is not True or rec['geometry_verified'] is not True:
            raise ValueError('Complete clock and geometry audit before training')
        if rec['calibration_source'] not in {'independent_calibration', 'train_calibration', 'synthetic'}:
            raise ValueError('Do not fit calibration or bone lengths on evaluation motion')
        if not rec['synthetic'] and ('synthetic' in (rec['crop_source'], rec['calibration_source'])):
            raise ValueError('Synthetic defaults cannot label real recordings')
        for key, index in ((group_key, group_splits), ('source_group', source_splits), ('sha256', hashes)):
            value = rec[key]
            if value in index and index[value] != rec['split']:
                raise ValueError(f'Split leakage: {key}={value}')
            index[value] = rec['split']
        file = (path.parent / rec['path']).resolve()
        if not file.is_relative_to(path.parent):
            raise ValueError('Sequence files must be inside the manifest directory')
        if not file.is_file() or sha256(file) != rec['sha256']:
            raise ValueError(f'Missing or modified sequence file: {rec["id"]}')
        rec['_path'] = str(file)
    data['_sha256'] = sha256(path)
    return data


def _array(z: Any, key: str, default: np.ndarray, shape: tuple | None = None) -> np.ndarray:
    x = np.asarray(z[key] if key in z else default)
    if x.dtype.kind not in 'bifu' or (shape is not None and x.shape != shape):
        raise ValueError(f'Bad dtype/shape for {key}: {x.dtype}, {x.shape}')
    if not np.isfinite(x).all():
        raise ValueError(f'Non-finite {key}; store zero + false mask for missing values')
    return x


def load_sequence(rec: dict, *, include_targets: bool = True) -> dict[str, Any]:
    """Load one recording. No interpolation, fitting or target-derived normalization."""
    with np.load(rec['_path'], allow_pickle=False) as z:
        for key in ('time', 'glove', 'glove_valid', 'glove_age', 'offsets'):
            if key not in z:
                raise ValueError(f'Missing required NPZ field {key}')
        times = _array(z, 'time', np.empty(0)).astype(np.float64)
        if times.ndim != 1 or len(times) < 2 or (np.diff(times) <= 0).any():
            raise ValueError('time must be strictly increasing, at least two frames')
        t = len(times)
        dt = np.diff(times, prepend=times[0] - (times[1] - times[0]))
        if (dt > 1).any():
            raise ValueError('Split timestamp resets or gaps >1 second into new sequences')
        offsets = _array(z, 'offsets', np.zeros((20, 3)), (20, 3)).astype(np.float32)
        lengths = np.linalg.norm(offsets[1:], axis=-1)
        if np.linalg.norm(offsets[0]) > 1e-7 or (lengths < .002).any() or (lengths > .2).any():
            raise ValueError('Invalid fixed skeleton offsets; check topology, provenance and units')
        g = _array(z, 'glove', np.zeros((t, 20, 3)), (t, 20, 3)).astype(np.float32)
        gv = _array(z, 'glove_valid', np.zeros((t, 20)), (t, 20)).astype(bool)
        ga = _array(z, 'glove_age', np.zeros(t), (t,)).astype(np.float32)
        if (ga < 0).any() or (ga > rec.get('max_observation_age_s', .05))[gv.any(-1)].any():
            raise ValueError('Future or stale glove observations marked valid')
        if (np.linalg.norm(g[:, 0], axis=-1)[gv[:, 0]] > 1e-5).any():
            raise ValueError('glove must be root-relative; do not inject MoCap root')
        if np.abs(g[gv]).max(initial=0) > 1:
            raise ValueError('Hand positions exceed one metre; check units/frame')
        rv = _array(z, 'rgb_valid', np.zeros(t), (t,)).astype(bool)
        dv = _array(z, 'depth_valid', np.zeros(t), (t,)).astype(bool)
        rf = _array(z, 'rgb_features', np.zeros((t, rec.get('rgb_feature_dim', 256))),
                    (t, rec.get('rgb_feature_dim', 256))).astype(np.float32)
        rgb = _array(z, 'rgb', np.zeros((t, 3, 16, 16))).astype(np.float32)
        depth = _array(z, 'depth', np.zeros((t, 1, 16, 16))).astype(np.float32)
        dm = _array(z, 'depth_mask', np.zeros_like(depth)).astype(bool)
        if rgb.ndim != 4 or rgb.shape[:2] != (t, 3) or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('rgb must be float [T,3,H,W] in [0,1]')
        if depth.ndim != 4 or depth.shape[:2] != (t, 1) or dm.shape != depth.shape:
            raise ValueError('depth/depth_mask must have matching [T,1,H,W] shapes')
        if rv.any() and not ('rgb' in z or 'rgb_features' in z):
            raise ValueError('RGB is valid but no images/features were supplied')
        if dv.any() and (not rec.get('metric_depth_validated', False) or 'depth' not in z):
            raise ValueError('Metric depth has not been validated')
        for name, mask in (('rgb', rv), ('depth', dv)):
            age = _array(z, name + '_age', np.zeros(t), (t,))
            if mask.any() and (name + '_age' not in z or (age[mask] < 0).any() or
                               (age[mask] > rec.get('max_observation_age_s', .05)).any()):
                raise ValueError(f'Future/stale or missing {name} acquisition age')
        geometry = _array(z, 'geometry', np.zeros((t, 18)), (t, 18)).astype(np.float32)
        if rv.any() and 'geometry' not in z:
            raise ValueError('RGB requires normalized camera/crop geometry [T,18]')
        inputs = dict(dt=dt.astype(np.float32), glove=g, glove_valid=gv, glove_age=ga,
                      rgb_features=rf, rgb=rgb, rgb_valid=rv, geometry=geometry,
                      depth=depth, depth_mask=dm, depth_valid=dv)
        # Supervision is never copied into inputs.
        targets = {}
        bones = _array(z, 'marker_bones', np.empty(0, np.int64)).astype(np.int64)
        moff = _array(z, 'marker_offsets', np.empty((0, 3)), (len(bones), 3)).astype(np.float32)
        if bones.ndim != 1 or ((bones < 0) | (bones >= 20)).any():
            raise ValueError('Invalid marker attachment bone ids')
        if include_targets:
            shapes = {'joints': (t, 20, 3), 'joint_valid': (t, 20), 'root': (t, 3),
                      'root_valid': (t,), 'rotations': (t, 20, 3, 3), 'rotation_valid': (t, 20),
                      'markers': (t, len(bones), 3), 'marker_valid': (t, len(bones)),
                      'uv': (t, 20, 2), 'uv_valid': (t, 20), 'intrinsics': (t, 3, 3),
                      'label_variance': (t, 20, 3)}
            for key, shape in shapes.items():
                value = _array(z, 'target_' + key, np.zeros(shape), shape)
                targets[key] = value.astype(bool if 'valid' in key else np.float32)
            targets['joint_valid'][:, 0] = False  # trivial normalized wrist is not an accuracy sample
            if targets['joint_valid'].any() and rec.get('label_type') not in {'anatomical_joints', 'synthetic'}:
                raise ValueError('Surface markers cannot be labeled as anatomical joint GT')
            if targets['root_valid'].any() and not rec.get('independent_root_gt', False):
                raise ValueError('Root loss requires independently measured anatomical root GT')
            if targets['rotation_valid'].any() and not rec.get('rotation_gt_validated', False):
                raise ValueError('Rotation loss requires validated native-axis rotational GT')
            if targets['rotation_valid'].any():
                rotations = targets['rotations'][targets['rotation_valid']]
                if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
                    raise ValueError('Rotational targets must be proper SO(3) matrices')
            if targets['marker_valid'].any() and not rec.get('marker_offsets_calibrated', False):
                raise ValueError('Marker supervision requires calibrated, frozen attachment offsets')
            if targets['uv_valid'].any() and not rec.get('undistorted_uv_validated', False):
                raise ValueError('2D targets must be undistorted full-image pixels')
            if (targets['label_variance'] < 0).any():
                raise ValueError('Negative label covariance')
        tensor = lambda x: torch.from_numpy(np.ascontiguousarray(x))
        return {'id': rec['id'], 'time': times, 'inputs': {k: tensor(v) for k, v in inputs.items()},
                'targets': {k: tensor(v) for k, v in targets.items()}, 'offsets': tensor(offsets),
                'marker_bones': tensor(bones), 'marker_offsets': tensor(moff)}


def chunk(sequence: dict, start: int, end: int, device: str | torch.device) -> dict:
    return {'inputs': {k: v[start:end].unsqueeze(0).to(device) for k, v in sequence['inputs'].items()},
            'targets': {k: v[start:end].unsqueeze(0).to(device) for k, v in sequence['targets'].items()},
            'offsets': sequence['offsets'].unsqueeze(0).to(device),
            'marker_bones': sequence['marker_bones'].to(device),
            'marker_offsets': sequence['marker_offsets'].unsqueeze(0).to(device)}
