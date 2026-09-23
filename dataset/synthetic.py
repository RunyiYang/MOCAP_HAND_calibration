"""Synthetic CONTRACT fixtures only. Their scores are not research results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from models.geometry import fk, so3_exp
from .sequence import SCHEMA, sha256


def synthetic_offsets() -> np.ndarray:
    """Illustrative skeleton, forbidden as a default for real participant data."""
    x = np.zeros((20, 3), np.float32)
    x[1:4] = [(-.03, .012, 0), (-.026, .01, 0), (-.023, .006, 0)]
    for base, px, length in ((4, -.025, .035), (8, -.008, .04), (12, .011, .037), (16, .028, .03)):
        x[base] = (px, .045, 0)
        x[base+1:base+4, 1] = (length, length * .7, length * .55)
    return x


def generate(out: str | Path, frames: int = 96, seed: int = 7) -> Path:
    out = Path(out)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Choose a fresh fixture directory; do not overwrite a dataset')
    if frames < 8:
        raise ValueError('Need at least eight frames')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(60, 256)).astype(np.float32) / np.sqrt(60)
    records = []
    for i, split in enumerate(('train', 'train', 'val', 'test')):
        offsets = synthetic_offsets() * (1 + i * .025)
        time = np.arange(frames, dtype=np.float64) / 30
        angles = np.zeros((frames, 20, 3), np.float32)
        for j in (1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18):
            angles[:, j, 0] = .3 + .25 * np.sin(time * (1 + (j % 4) * .1) + i * .2)
        with torch.no_grad():
            positions, _ = fk(so3_exp(torch.from_numpy(angles)), torch.from_numpy(offsets))
        target = positions.numpy()
        glove = target.copy()
        glove[..., 2] += (.008 + .001 * time[:, None]).astype(np.float32)
        glove += rng.normal(0, .001, glove.shape).astype(np.float32)
        glove -= glove[:, :1].copy()
        # Explicitly synthetic features correlated with true pose for wiring tests.
        features = target.reshape(frames, -1) / .1 @ projection
        mask = np.zeros((frames, 20), bool)
        mask[:, 4:] = True
        rgb_valid = np.ones(frames, bool)
        rgb_valid[frames//2:frames//2 + 3] = False
        file = out / f'synthetic_{i}.npz'
        np.savez_compressed(file, time=time, offsets=offsets, glove=glove,
                            glove_valid=np.ones((frames, 20), bool), glove_age=np.zeros(frames),
                            rgb_features=features, rgb_valid=rgb_valid, rgb_age=np.zeros(frames),
                            geometry=np.zeros((frames, 18), np.float32), target_joints=target,
                            target_joint_valid=mask)
        records.append(dict(id=f'synthetic_{i}', path=file.name, sha256=sha256(file), split=split,
                            subject_id=f'subject_{i}', session_id=f'session_{i}', source_group=f'capture_{i}',
                            units='m', frame='camera_axes_root_relative', calibration_source='synthetic',
                            timing_verified=True, geometry_verified=True, input_uses_mocap=False,
                            crop_source='synthetic', synthetic=True, label_type='synthetic',
                            rgb_feature_dim=256, metric_depth_validated=False))
    manifest = out / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA, split_group='subject_id', sequences=records), indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, default=96)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    print(generate(a.out, a.frames, a.seed))


if __name__ == '__main__':
    main()
