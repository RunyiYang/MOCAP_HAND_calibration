"""Pack audited asynchronous NPZ export bundles into the training contract.

This does not parse proprietary acquisition files. The coding agent must export
unconditioned sensor arrays and audited BVH/marker labels from local raw records.
See dataset/README.md. No fit, nearest-future sampling, or MoCap wrist injection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from .alignment import asof_indices, old_bvh21_targets
from .sequence import SCHEMA, sha256, read_manifest, load_sequence


def pack(spec_path: str | Path, destination: str | Path) -> Path:
    spec_path, destination = Path(spec_path).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new output directory; no dataset overwrite is allowed')
    spec = json.loads(spec_path.read_text())
    if spec.get('schema') != 'handcalib.raw_export_spec.v1':
        raise ValueError('Unsupported export spec')
    destination.mkdir(parents=True)
    records = []
    for number, raw in enumerate(spec['sequences']):
        rec = dict(raw)
        source = (spec_path.parent / rec.pop('bundle')).resolve()
        if rec.pop('source_kind', None) != 'raw_audited_export':
            raise ValueError('Use raw audited exports, not visualization/teacher-fit artifacts')
        if not rec.get('synthetic', False):
            if not rec.get('audit_report') or not rec.get('source_assets'):
                raise ValueError('Real data require a written audit and source-file hashes')
            cache = rec.get('feature_extractor', {})
            if cache.get('uses_future') is not False or cache.get('uses_mocap') is not False:
                raise ValueError('Document a causal sensor-only visual feature/crop pipeline')
        with np.load(source, allow_pickle=False) as z:
            query = np.asarray(z['time'], np.float64)
            offsets = np.asarray(z['offsets'], np.float32)
            arrays = {'time': query, 'offsets': offsets}
            for modality in ('glove', 'rgb', 'depth'):
                clock = modality + '_acquisition'
                if clock not in z:
                    if modality == 'glove':
                        raise ValueError('Raw export requires glove acquisition and arrival clocks')
                    continue
                index, age, fresh = asof_indices(z[clock], z[modality + '_arrival'], query,
                                                rec.get('max_observation_age_s', .05))
                available = index >= 0
                keys = {'glove': ('glove', 'glove_valid'),
                        'rgb': ('rgb', 'rgb_features', 'geometry', 'rgb_valid'),
                        'depth': ('depth', 'depth_mask', 'depth_valid')}[modality]
                for key in keys:
                    if key not in z:
                        continue
                    values = np.asarray(z[key])
                    if len(values) != len(z[clock]):
                        raise ValueError(f'{key} does not match its acquisition clock')
                    sample = np.zeros((len(query), *values.shape[1:]), values.dtype)
                    sample[available] = values[index[available]]
                    arrays[key] = sample
                arrays[modality + '_age'] = age
                if modality == 'glove':
                    if 'glove' not in arrays:
                        raise ValueError('Missing glove points')
                    points = arrays['glove'].astype(np.float32)
                    mask = arrays.get('glove_valid', np.isfinite(points).all(-1)).astype(bool)
                    mask &= np.isfinite(points).all(-1)
                    mask &= mask[:, :1]
                    mask &= fresh[:, None]
                    points = points - points[:, :1].copy()  # sensor wrist only, never reference wrist
                    arrays['glove'] = np.where(mask[..., None], points, 0)
                    arrays['glove_valid'] = mask
                else:
                    arrays[modality + '_valid'] = fresh & arrays.get(modality + '_valid', np.ones(len(query), bool)).astype(bool)
            # Targets have already been interpolated at query times by the audited exporter.
            # Their offline interpolation is permissible for labels, never observations.
            for key in z.files:
                if key.startswith('target_') or key in ('marker_bones', 'marker_offsets'):
                    arrays[key] = z[key]
            if 'bvh_joints_m' in z:
                if rec.get('reference_source') != 'Skeleton_BVH_FK':
                    raise ValueError('Old anatomical references must be Skeleton BVH-FK')
                arrays['target_joints'], arrays['target_joint_valid'] = old_bvh21_targets(z['bvh_joints_m'], z['bvh_valid'])
            file = destination / f'sequence_{number:04d}.npz'
            np.savez_compressed(file, **arrays)
        rec.update(path=file.name, sha256=sha256(file), export_bundle_sha256=sha256(source))
        records.append(rec)
    manifest = destination / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA, split_group=spec['split_group'], sequences=records), indent=2))
    verified = read_manifest(manifest, allow_synthetic=all(r.get('synthetic', False) for r in records))
    for rec in verified['sequences']:
        load_sequence(rec)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(pack(a.spec, a.out))


if __name__ == '__main__':
    main()
