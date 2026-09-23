from argparse import Namespace
from pathlib import Path

import torch
from train.fit import run
from train.common import load_checkpoint
from train.evaluate import evaluate


def test_checkpoint_resume_matches_continuous(fixture_data, tmp_path):
    manifest_path, manifest, _ = fixture_data
    config = str(Path(__file__).resolve().parents[1] / 'configs' / 'smoke.json')
    def args(out, steps, resume=None):
        return Namespace(config=config, manifest=str(manifest_path), out=str(out), device='cpu',
                         epochs=1, max_steps=steps, resume=resume, allow_synthetic=True)
    full = tmp_path / 'full'
    interrupted = tmp_path / 'interrupted'
    run(args(full, 4))
    run(args(interrupted, 2))
    run(args(interrupted, 4, str(interrupted / 'last.pt')))
    m1, c1 = load_checkpoint(full / 'last.pt')
    m2, c2 = load_checkpoint(interrupted / 'last.pt')
    assert c1['step'] == c2['step'] == 4
    assert c1['cursor'] == c2['cursor']
    assert all(torch.equal(v, m2.state_dict()[k]) for k, v in m1.state_dict().items())
    records = [r for r in manifest['sequences'] if r['split'] == 'test']
    result = evaluate(m2, records, chunk_length=4, forecasts=(100,))
    assert result['joints']['count'] > 0
    assert result['forecasts_ms']['100']['count'] > 0
    assert result['synthetic'] is True


def test_tiny_synthetic_sequence_can_fit(fixture_data):
    from models.handcalib import HandCalib, ModelConfig
    from dataset.sequence import chunk
    from train.losses import masked_mean, robust
    torch.manual_seed(101)
    _, _, seq = fixture_data
    batch = chunk(seq, 0, 4, 'cpu')
    model = HandCalib(ModelConfig(hidden=32, slow_hidden=16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    values = []
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        output, _ = model(batch['inputs'], batch['offsets'])
        loss = masked_mean(robust(output['joints'], batch['targets']['joints']), batch['targets']['joint_valid'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        values.append(float(loss.detach()))
    assert values[-1] < values[0] * .7, values
