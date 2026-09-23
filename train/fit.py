"""Single-process, single-stream TBPTT trainer with persistent states and resumable checkpoints.

Run from the repository root. No scheduler submission, distributed launcher,
external logging service, auto-download or background process is used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from dataset.sequence import read_manifest, load_sequence, chunk
from models.handcalib import StreamState
from .common import read_config, new_model, atomic_save, git_revision, load_checkpoint
from .evaluate import evaluate
from .losses import supervised_loss, forecast_loss


def run(args) -> dict:
    cfg = read_config(args.config)
    tr = cfg['training']
    torch.set_num_threads(tr.get('cpu_threads', 2))
    torch.manual_seed(tr.get('seed', 42))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for a smoke test')
    manifest = read_manifest(args.manifest, allow_synthetic=args.allow_synthetic)
    records = [r for r in manifest['sequences'] if r['split'] == 'train']
    validation = [r for r in manifest['sequences'] if r['split'] == 'val']
    if not records or not validation:
        raise ValueError('Both train and val splits are required')
    # Validate all contract fields, including test data, without using test labels for fitting.
    visual_events = 0
    for rec in manifest['sequences']:
        seq_check = load_sequence(rec)
        if rec['split'] == 'train':
            visual_events += int(seq_check['inputs']['rgb_valid'].sum())
        if rec['split'] == 'train' and not any(seq_check['targets'][k].any() for k in
                                               ('joint_valid', 'marker_valid', 'root_valid', 'uv_valid')):
            raise ValueError(f'No usable training supervision in {rec["id"]}')
        if cfg['model'].get('use_depth', False) and not rec.get('metric_depth_validated', False):
            raise ValueError('Depth model requested before metric-depth validation')
        if cfg['model'].get('visual_mode', 'cached') == 'image' and seq_check['inputs']['rgb_valid'].any():
            # Image mode must not silently train on a cached-only sequence's zero images.
            with np.load(rec['_path'], allow_pickle=False) as z:
                if 'rgb' not in z:
                    raise ValueError('image mode requires actual RGB arrays, not feature-cache-only records')
    del seq_check
    if cfg['model'].get('visual_mode', 'cached') != 'none' and visual_events == 0:
        raise ValueError('No valid training RGB events. Fix clocks/crops or explicitly select glove_only.json')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Output already has a checkpoint; choose another path or --resume')
    model = new_model(cfg, device)
    if getattr(args, 'init_weights', None):
        _, initial = load_checkpoint(args.init_weights, device)
        if initial['manifest_sha256'] != manifest['_sha256']:
            raise ValueError('Warm-start requires the same audited manifest')
        model.load_state_dict(initial['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg['optimizer'])
    cursor = {'epoch': 0, 'order': [], 'sequence': 0, 'offset': 0}
    step, best, state = 0, float('inf'), None
    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device)
        if ckpt['config'] != cfg or ckpt['manifest_sha256'] != manifest['_sha256']:
            raise ValueError('Resume requires the exact configuration and dataset manifest')
        optimizer = torch.optim.AdamW(model.parameters(), **cfg['optimizer'])
        optimizer.load_state_dict(ckpt['optimizer'])
        cursor, step, best = ckpt['cursor'], ckpt['step'], ckpt['best_val_mm']
        state = StreamState.unpack(ckpt['stream_state'], device) if ckpt['stream_state'] else None
        torch.set_rng_state(ckpt['torch_rng'])
        if device.type == 'cuda' and ckpt.get('cuda_rng'):
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    # Store source split identities and hashes, not raw participant data.
    provenance = {'manifest_sha256': manifest['_sha256'], 'git_revision': git_revision(),
                  'torch_version': str(torch.__version__), 'synthetic': any(r['synthetic'] for r in records),
                  'sequences': [{k: r[k] for k in ('id', 'split', 'subject_id', 'session_id', 'source_group', 'sha256')}
                                for r in manifest['sequences']],
                  'parameters': sum(p.numel() for p in model.parameters()), 'model_specification': model.specification()}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    epochs = args.epochs if args.epochs is not None else tr.get('epochs', 20)
    max_steps = args.max_steps if args.max_steps is not None else tr.get('max_steps', 5000)
    width = tr.get('chunk_length', 64)
    if min(epochs, max_steps, width, tr.get('checkpoint_every', 100)) < 1:
        raise ValueError('epochs, max_steps, chunk_length and checkpoint_every must be positive')

    def save(name='last.pt'):
        payload = dict(schema='handcalib.checkpoint.v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
                       config=cfg, manifest_sha256=manifest['_sha256'], provenance=provenance, cursor=dict(cursor),
                       step=step, best_val_mm=best, stream_state=state.pack() if state else None,
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        atomic_save(payload, out / name)

    last_metrics = None
    while cursor['epoch'] < epochs and step < max_steps:
        if not cursor['order']:
            cursor['order'] = torch.randperm(len(records)).tolist()
        while cursor['sequence'] < len(records) and step < max_steps:
            rec = records[cursor['order'][cursor['sequence']]]
            seq = load_sequence(rec)
            n = len(seq['time'])
            while cursor['offset'] < n and step < max_steps:
                start, end = cursor['offset'], min(n, cursor['offset'] + width)
                batch = chunk(seq, start, end, device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                predictions, new_state = model(batch['inputs'], batch['offsets'], state)
                weights = dict(cfg.get('losses', {}))
                if step < tr.get('nll_warmup_steps', 500):
                    weights['nll'] = 0.
                loss, terms = supervised_loss(predictions, batch, weights)
                rollout = int(tr.get('rollout_steps', 8))
                roll = loss.detach() * 0
                if rollout > 0 and end < n and weights.get('rollout', .2) > 0:
                    future = chunk(seq, end, min(n, end+rollout), device)
                    prediction, _ = model.rollout(new_state, batch['offsets'], future['inputs']['dt'])
                    roll = forecast_loss(prediction, future['targets'])
                    loss = loss + weights.get('rollout', .2) * roll
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at step {step}; no checkpoint overwritten')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get('grad_clip', 1.), error_if_nonfinite=True)
                optimizer.step()
                state = new_state.detach()  # TBPTT detaches gradients, not the physical/calibration state.
                step += 1
                cursor['offset'] = end
                row = {'step': step, 'epoch': cursor['epoch'], 'sequence': rec['id'], 'loss': float(loss.detach()),
                       'rollout': float(roll.detach()), 'gradient_norm': float(grad),
                       **{k: float(v.detach()) for k, v in terms.items()}}
                with (out / 'train.jsonl').open('a') as f:
                    f.write(json.dumps(row, allow_nan=False) + '\n')
                if step % tr.get('log_every', 10) == 0 or step == 1:
                    print(json.dumps(row), flush=True)
                if step % tr.get('checkpoint_every', 100) == 0:
                    save()
            if cursor['offset'] == n:
                state = None
                cursor['offset'] = 0
                cursor['sequence'] += 1
        if cursor['sequence'] == len(records):
            cursor = {'epoch': cursor['epoch'] + 1, 'order': [], 'sequence': 0, 'offset': 0}
        # Validation is state-reset per sequence, never teacher forced or test-selected.
        last_metrics = evaluate(model, validation, device, width)
        score = last_metrics['joints']['mean_mm']
        if score is None:
            score = last_metrics['surface_markers']['mean_mm']
        if score is None:
            score = last_metrics['independent_root']['mean_mm']
        (out / 'validation.json').write_text(json.dumps(last_metrics, indent=2, allow_nan=False))
        if score is not None and score < best:
            best = score
            save('best.pt')
        save()
    if last_metrics is None:
        save()
    return {'steps': step, 'parameters': provenance['parameters'], 'checkpoint': str(out / 'last.pt'),
            'best_validation_joint_mm': best if best != float('inf') else None,
            'synthetic': provenance['synthetic']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='train/configs/solved_pose.json')
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int)
    p.add_argument('--max-steps', type=int)
    restart = p.add_mutually_exclusive_group()
    restart.add_argument('--resume')
    restart.add_argument('--init-weights', help='Warm-start weights only; reset optimizer/state for a new training stage')
    p.add_argument('--allow-synthetic', action='store_true')
    args = p.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
