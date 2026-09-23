import inspect

import numpy as np
import torch
from dataset.sequence import chunk, load_sequence
from models.handcalib import ModelConfig, HandCalib
from models.geometry import PARENTS
from train.losses import supervised_loss
from train.evaluate import camera_outage


def test_prefix_invariance(network, fixture_data):
    _, _, seq = fixture_data
    b = chunk(seq, 0, 16, 'cpu')
    with torch.no_grad():
        expected, _ = network(b['inputs'], b['offsets'])
        changed = {k: v.clone() for k, v in b['inputs'].items()}
        changed['glove'][:, 8:] += .07
        changed['rgb_features'][:, 8:] += 4
        actual, _ = network(changed, b['offsets'])
    assert torch.equal(expected['joints'][:, :8], actual['joints'][:, :8])
    assert not torch.allclose(expected['joints'][:, 8:], actual['joints'][:, 8:])


def test_chunked_equals_streaming(network, fixture_data):
    _, _, seq = fixture_data
    b = chunk(seq, 0, 16, 'cpu')
    with torch.no_grad():
        full, _ = network(b['inputs'], b['offsets'])
        first, state = network(chunk(seq, 0, 7, 'cpu')['inputs'], b['offsets'])
        second, _ = network(chunk(seq, 7, 16, 'cpu')['inputs'], b['offsets'], state.detach())
    assert torch.equal(full['joints'], torch.cat((first['joints'], second['joints']), 1))


def test_target_removal_does_not_change_inputs(fixture_data):
    _, manifest, original = fixture_data
    clean = load_sequence(manifest['sequences'][0], include_targets=False)
    assert clean['targets'] == {}
    for key in original['inputs']:
        assert torch.equal(original['inputs'][key], clean['inputs'][key])


def test_forecast_no_measurement_api_and_growing_variance(network, fixture_data, monkeypatch):
    _, _, seq = fixture_data
    b = chunk(seq, 0, 4, 'cpu')
    _, state = network(b['inputs'], b['offsets'])
    assert list(inspect.signature(network.rollout).parameters) == ['state', 'offsets', 'future_dt']
    def forbidden(*a, **kw):
        raise AssertionError('Forecast called observation update')
    monkeypatch.setattr(network, 'step', forbidden)
    out, final = network.rollout(state, b['offsets'], torch.full((1, 8), 1/30))
    assert (final.variance >= state.variance).all()
    assert torch.equal(final.bias, state.bias)
    assert torch.equal(final.slow, state.slow)
    assert torch.isfinite(out['joints']).all()


def test_vision_outage_holds_slow_state(network, fixture_data):
    _, _, seq = fixture_data
    b = chunk(seq, 0, 4, 'cpu')
    _, state = network(b['inputs'], b['offsets'])
    missing = camera_outage(seq, 0, 100)
    c = chunk(missing, 4, 8, 'cpu')
    _, final = network(c['inputs'], c['offsets'], state)
    assert torch.equal(final.slow, state.slow)
    assert torch.equal(final.bias, state.bias)
    assert torch.equal(missing['inputs']['glove'], seq['inputs']['glove'])


def test_all_missing_equals_prediction(network, fixture_data):
    _, _, seq = fixture_data
    b = chunk(seq, 0, 1, 'cpu')
    inp = {k: v[:, 0] for k, v in b['inputs'].items()}
    for k in ('rgb_valid', 'depth_valid', 'glove_valid'):
        inp[k] = torch.zeros_like(inp[k])
    state = network.initial_state(1, 'cpu')
    expected = network.predict(state, inp['dt'])
    _, actual = network.step(inp, b['offsets'], state)
    assert torch.allclose(actual.rotations, expected.rotations, atol=1e-7)
    assert torch.equal(actual.variance, expected.variance)


def test_trainable_finite_gradients_and_fixed_bones(network, fixture_data):
    _, _, seq = fixture_data
    b = chunk(seq, 0, 8, 'cpu')
    out, _ = network(b['inputs'], b['offsets'])
    loss, _ = supervised_loss(out, b)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    for j in range(1, 20):
        actual = (out['joints'][:, :, j] - out['joints'][:, :, PARENTS[j]]).norm(dim=-1)
        assert torch.allclose(actual, b['offsets'][:, j].norm(dim=-1)[:, None].expand_as(actual), atol=1e-6)


def test_image_depth_branches_backward(fixture_data):
    _, _, seq = fixture_data
    b = chunk(seq, 0, 2, 'cpu')
    b['inputs']['rgb'] = torch.rand(1, 2, 3, 32, 32)
    b['inputs']['depth'] = torch.rand(1, 2, 1, 32, 32)
    b['inputs']['depth_mask'] = torch.ones(1, 2, 1, 32, 32, dtype=torch.bool)
    b['inputs']['depth_valid'][:] = True
    model = HandCalib(ModelConfig(visual_mode='image', use_depth=True, hidden=32, slow_hidden=16))
    with torch.no_grad():
        model.delta_head.weight.normal_(0, .005)
    out, _ = model(b['inputs'], b['offsets'])
    loss, _ = supervised_loss(out, b)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
