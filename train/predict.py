"""Inference export. Does not load or require MoCap labels."""
import argparse
from pathlib import Path

import numpy as np
import torch
from dataset.sequence import read_manifest, load_sequence, chunk
from .common import load_checkpoint


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--device', default='cpu')
    p.add_argument('--allow-synthetic', action='store_true')
    a = p.parse_args()
    manifest = read_manifest(a.manifest, allow_synthetic=a.allow_synthetic)
    model, _ = load_checkpoint(a.checkpoint, a.device)
    model.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for rec in manifest['sequences']:
        if rec['split'] != a.split:
            continue
        seq = load_sequence(rec, include_targets=False)
        state, predictions = None, []
        for start in range(0, len(seq['time']), 64):
            b = chunk(seq, start, min(start+64, len(seq['time'])), a.device)
            pred, state = model(b['inputs'], b['offsets'], state)
            predictions.append({k: pred[k][0].cpu().numpy() for k in
                                ('joints', 'root', 'rotations', 'joint_variance', 'calibration_bias')})
        # Do not use participant-controlled IDs as filesystem paths.
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in rec['id'])
        destination = out / f'{safe}_{rec["sha256"][:8]}.npz'
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite {destination}')
        np.savez_compressed(destination, time=seq['time'], **{k: np.concatenate([v[k] for v in predictions])
                                                             for k in predictions[0]})
        print(destination)


if __name__ == '__main__':
    main()
