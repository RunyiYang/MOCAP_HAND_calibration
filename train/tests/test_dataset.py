import json

import numpy as np
import pytest
from dataset.alignment import asof_indices, old_bvh21_targets
from dataset.sequence import read_manifest, load_sequence, sha256


def test_no_future_out_of_order_and_no_double_count():
    acq = np.array([0., .01, .02])
    arrival = np.array([.05, .015, .025])
    idx, ages, fresh = asof_indices(acq, arrival, np.array([0., .015, .025, .04, .05]), .1)
    assert idx.tolist() == [-1, 1, 2, 2, 2]
    assert fresh.tolist() == [False, True, True, False, False]
    assert (ages >= 0).all()


def test_stale_packet_rejected():
    idx, _, valid = asof_indices(np.array([0.]), np.array([.1]), np.array([.1, .2]), .05)
    assert (idx == -1).all() and not valid.any()


def test_old_mapping_masks_thumb():
    p = np.random.randn(3, 21, 3)
    target, valid = old_bvh21_targets(p, np.ones((3, 21), bool))
    assert not valid[:, :4].any()
    assert np.allclose(target[:, 4:], p[:, 5:] - p[:, :1])


def test_synthetic_explicit_opt_in(fixture_data):
    path, _, _ = fixture_data
    with pytest.raises(ValueError, match='Synthetic'):
        read_manifest(path)


@pytest.mark.parametrize('field', ['subject_id', 'source_group', 'sha256'])
def test_split_leakage_rejected(fixture_data, field):
    path, _, _ = fixture_data
    data = json.loads(path.read_text())
    data['sequences'][2][field] = data['sequences'][0][field]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_manifest(path, allow_synthetic=True)


def test_mocap_input_flag_rejected(fixture_data):
    path, _, _ = fixture_data
    data = json.loads(path.read_text())
    data['sequences'][0]['input_uses_mocap'] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='MoCap'):
        read_manifest(path, allow_synthetic=True)


def test_changed_file_hash_rejected(fixture_data):
    path, manifest, _ = fixture_data
    with open(manifest['sequences'][0]['_path'], 'ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='modified'):
        read_manifest(path, allow_synthetic=True)


def test_marker_not_anatomical_gt(fixture_data):
    _, manifest, _ = fixture_data
    rec = dict(manifest['sequences'][0], label_type='surface_markers')
    with pytest.raises(ValueError, match='anatomical'):
        load_sequence(rec)


def test_labels_physically_removed(fixture_data, tmp_path):
    _, manifest, original = fixture_data
    rec = dict(manifest['sequences'][0])
    with np.load(rec['_path'], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if not k.startswith('target_')}
    path = tmp_path / 'unlabeled.npz'
    np.savez_compressed(path, **arrays)
    rec['_path'] = str(path)
    clean = load_sequence(rec, include_targets=False)
    for key in original['inputs']:
        assert np.array_equal(original['inputs'][key].numpy(), clean['inputs'][key].numpy())
    assert clean['targets'] == {}


def test_async_packer_contract(fixture_data, tmp_path):
    from dataset.pack import pack
    _, manifest, _ = fixture_data
    sequences = []
    for i, rec in enumerate(manifest['sequences']):
        with np.load(rec['_path'], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        times = arrays['time']
        arrays.update(glove_acquisition=times, glove_arrival=times,
                      rgb_acquisition=times, rgb_arrival=times)
        bundle = tmp_path / f'bundle{i}.npz'
        np.savez_compressed(bundle, **arrays)
        metadata = {k: v for k, v in rec.items() if k not in {'_path', 'path', 'sha256'}}
        metadata.update(bundle=bundle.name, source_kind='raw_audited_export')
        sequences.append(metadata)
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'schema': 'handcalib.raw_export_spec.v1', 'split_group': 'subject_id', 'sequences': sequences}))
    packed = pack(spec, tmp_path / 'packed')
    new = read_manifest(packed, allow_synthetic=True)
    assert len(new['sequences']) == 4
    seq = load_sequence(new['sequences'][0])
    assert seq['inputs']['glove_valid'].all()
    assert seq['targets']['joint_valid'][:, 4:].all()
