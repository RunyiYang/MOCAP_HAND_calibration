"""Whole-sequence evaluation, vendor baseline, camera outages and free forecasts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from dataset.sequence import read_manifest, load_sequence, chunk
from models.geometry import TIPS, attached_markers
from .common import load_checkpoint


def summary(arrays: list[np.ndarray]) -> dict:
    values = np.concatenate([x.reshape(-1) for x in arrays if x.size]) if any(x.size for x in arrays) else np.empty(0)
    if not len(values):
        return {'count': 0, 'mean_mm': None, 'median_mm': None, 'p95_mm': None}
    if not np.isfinite(values).all():
        raise FloatingPointError('Non-finite predictions count as a failure, not missing data')
    return {'count': len(values), 'mean_mm': float(values.mean() * 1000),
            'median_mm': float(np.median(values) * 1000), 'p95_mm': float(np.quantile(values, .95) * 1000)}


def camera_outage(sequence: dict, start: float, duration: float) -> dict:
    """Copies masks only. Glove packets remain available; targets never change."""
    result = dict(sequence)
    result['inputs'] = dict(sequence['inputs'])
    selection = torch.from_numpy((sequence['time'] - sequence['time'][0] >= start) &
                                 (sequence['time'] - sequence['time'][0] < start + duration))
    for key in ('rgb_valid', 'depth_valid'):
        value = sequence['inputs'][key].clone()
        value[selection] = False
        result['inputs'][key] = value
    return result


@torch.no_grad()
def evaluate(model, records: list[dict], device='cpu', chunk_length=64,
             outage: tuple[float, float] | None = None, forecasts: tuple[int, ...] = ()) -> dict:
    if model is not None:
        model.eval()
    errors, tips, roots, markers, coverage, sequence_reports = [], [], [], [], [], []
    future = {str(h): [] for h in forecasts}
    output_frames, expected_frames = 0, 0
    begin = time.perf_counter()
    for rec in records:
        seq = load_sequence(rec)
        if outage:
            seq = camera_outage(seq, *outage)
        n = len(seq['time'])
        state, predicted, variances, pred_roots = None, [], [], []
        for start in range(0, n, chunk_length):
            end = min(n, start + chunk_length)
            batch = chunk(seq, start, end, device)
            if model is None:
                # Static vendor baseline. Missing rows are retained; coverage reported below.
                out = {'joints': batch['inputs']['glove'], 'root': torch.zeros(1, end-start, 3, device=device)}
            else:
                out, state = model(batch['inputs'], batch['offsets'], state)
                variances.append(out['joint_variance'].cpu()[0])
                if batch['targets']['marker_valid'].any():
                    count = end - start
                    mp = attached_markers(out['joints'].reshape(count, 20, 3), out['frames'].reshape(count, 20, 3, 3), batch['marker_bones'], batch['marker_offsets'].expand(count, -1, -1))
                    me = (mp - batch['targets']['markers'][0]).norm(dim=-1)
                    markers.append(me[batch['targets']['marker_valid'][0]].cpu().numpy())
            predicted.append(out['joints'].cpu()[0])
            pred_roots.append(out['root'].cpu()[0])
            if model is not None and forecasts and end < n:
                for horizon in forecasts:
                    future_time = seq['time'][end-1] + horizon / 1000
                    target_index = int(np.searchsorted(seq['time'], future_time - 1e-9))
                    if target_index >= n:
                        continue
                    fb = chunk(seq, end, target_index+1, device)
                    fo, _ = model.rollout(state, batch['offsets'], fb['inputs']['dt'])
                    mask = fb['targets']['joint_valid'][:, -1]
                    e = (fo['joints'][:, -1] - fb['targets']['joints'][:, -1]).norm(dim=-1)
                    future[str(horizon)].append(e[mask].cpu().numpy())
        pred = torch.cat(predicted)
        valid = seq['targets']['joint_valid']
        expected_frames += n
        if model is None:
            available = seq['inputs']['glove_valid']
            # Invalid vendor points are failures, not silently selected away.
            output_frames += int((available | ~valid).all(-1).sum())
        else:
            output_frames += n
        if not torch.isfinite(pred).all():
            raise FloatingPointError(f'Non-finite output on {rec["id"]}')
        e = (pred - seq['targets']['joints']).norm(dim=-1)
        selected = e[valid].numpy()
        errors.append(selected)
        tip_mask = valid[:, list(TIPS)]
        tips.append(e[:, list(TIPS)][tip_mask].numpy())
        sequence_reports.append({'id': rec['id'], **summary([selected])})
        root_valid = seq['targets']['root_valid']
        if model is not None and root_valid.any():
            roots.append((torch.cat(pred_roots) - seq['targets']['root']).norm(dim=-1)[root_valid].numpy())
        if variances:
            var = torch.cat(variances) + seq['targets']['label_variance']
            coordinate_mask = valid[..., None].expand_as(pred)
            within = (pred - seq['targets']['joints']).abs() <= 1.96 * var.clamp_min(1e-6).sqrt()
            coverage.append(within[coordinate_mask].float().numpy())
    return {'metric': 'fixed_camera_axes_root_translation_normalized_EPE',
            'joints': summary(errors), 'fingertips': summary(tips), 'independent_root': summary(roots), 'surface_markers': summary(markers),
            'frames': expected_frames, 'output_frame_coverage': output_frames / max(1, expected_frames),
            'coordinate_95pct_coverage': float(np.concatenate(coverage).mean()) if any(x.size for x in coverage) else None,
            'sequences': sequence_reports, 'forecasts_ms': {k: summary(v) for k, v in future.items()},
            'forecast_protocol': 'all future observations withheld; first reference timestamp at/after requested horizon',
            'elapsed_evaluation_seconds': time.perf_counter() - begin,
            'latency_note': 'Evaluation runtime includes I/O; this is NOT acquisition-to-output latency',
            'synthetic': any(r['synthetic'] for r in records),
            'uncertainty_note': 'First-order diagonal approximation; coverage is evaluated, not guaranteed'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--vendor', action='store_true')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--device', default='cpu')
    p.add_argument('--chunk-length', type=int, default=64)
    p.add_argument('--out', required=True)
    p.add_argument('--allow-synthetic', action='store_true')
    p.add_argument('--camera-outage', nargs=2, type=float, metavar=('START_S', 'DURATION_S'))
    p.add_argument('--forecast-ms', type=int, nargs='*', default=[])
    args = p.parse_args()
    if args.chunk_length < 1 or any(h <= 0 for h in args.forecast_ms):
        p.error('Chunk length and forecast horizons must be positive')
    manifest = read_manifest(args.manifest, allow_synthetic=args.allow_synthetic)
    records = [r for r in manifest['sequences'] if r['split'] == args.split]
    if not records:
        p.error('Requested split has no sequences')
    model = None
    if args.checkpoint:
        model, ckpt = load_checkpoint(args.checkpoint, args.device)
        if ckpt['manifest_sha256'] != manifest['_sha256']:
            p.error('Manifest differs from checkpoint. Use a separately reviewed transfer-evaluation protocol.')
    report = evaluate(model, records, args.device, args.chunk_length,
                      tuple(args.camera_outage) if args.camera_outage else None, tuple(args.forecast_ms))
    report['manifest_sha256'] = manifest['_sha256']
    report['split'] = args.split
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
