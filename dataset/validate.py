"""Validate every sequence before training. Exits nonzero on the first violation."""
import argparse
import json
from .sequence import read_manifest, load_sequence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--allow-synthetic', action='store_true')
    args = p.parse_args()
    manifest = read_manifest(args.manifest, allow_synthetic=args.allow_synthetic)
    report = []
    for rec in manifest['sequences']:
        seq = load_sequence(rec)
        report.append({'id': rec['id'], 'split': rec['split'], 'frames': len(seq['time']),
                       'valid_joint_labels': int(seq['targets']['joint_valid'].sum()),
                       'valid_marker_labels': int(seq['targets']['marker_valid'].sum()),
                       'rgb_events': int(seq['inputs']['rgb_valid'].sum())})
    print(json.dumps({'status': 'pass', 'manifest_sha256': manifest['_sha256'],
                      'sequences': report}, indent=2))


if __name__ == '__main__':
    main()
