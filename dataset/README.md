# Training data contract

Run everything from the repository root. Raw acquisition files remain local. This module does **not** train from the reviewed overlay videos or same-sequence teacher-fit outputs.

## First use

```bash
python -m dataset.synthetic --out dataset/local/smoke --frames 96
python -m dataset.validate --manifest dataset/local/smoke/manifest.json --allow-synthetic
```

The synthetic feature vectors deliberately encode synthetic target motion for wiring tests. Their scores cannot be presented as calibration accuracy or generalization results. Real-data commands must omit `--allow-synthetic`.

## Canonical NPZ: one hand in one recording

All arrays have numeric/bool dtypes, no pickle/object arrays. Units are metres, seconds and radians. Missing numeric values are zero with false masks, not NaN. Time is a corrected common clock. Each sequence has at least two frames with dt in `(0,1]` seconds; split clock resets/long discontinuities into new sequences.

| Field | Shape | Semantics |
|---|---|---|
| `time` | T | Strictly increasing output-event times, float64 |
| `offsets` | 20,3 | Fixed local joint offsets; offset[0]=0; from independent/train calibration |
| `glove` | T,20,3 | Solver points in fixed camera axes, sensor-wrist subtracted; no GT rotation or translation |
| `glove_valid` | T,20 | Fresh, available, quality-valid observations; false when merely reusing a held packet |
| `glove_age` | T | Output time minus selected packet acquisition, nonnegative |
| `rgb_features` | T,F | Optional causal feature cache, default F=256 |
| `rgb` | T,3,H,W | Optional RGB float `[0,1]`, for image mode |
| `rgb_valid`, `rgb_age` | T | Fresh RGB event and acquisition age |
| `geometry` | T,18 | Flattened normalized intrinsics (9) + crop-to-full normalized homography (9) |
| `depth`, `depth_mask` | T,1,H,W | Optional original metric depth and validity; preview MP4 is forbidden |
| `depth_valid`, `depth_age` | T | Fresh validated depth event and age |

Normalize the two geometry matrices using original image/crop sizes according to a documented extractor convention. Keep that convention fixed across splits; no whole-sequence feature normalization. Stored feature vectors alone do not prove causality: the extractor/crop/normalization must be audited.

### Supervision (never passed to model inputs)

| Field | Shape | Enabled only when |
|---|---|---|
| `target_joints`, `target_joint_valid` | T,20,3 / T,20 | Anatomical joint labels with validated mapping |
| `target_root`, `target_root_valid` | T,3 / T | Independent anatomical wrist position in camera frame |
| `target_rotations`, `target_rotation_valid` | T,20,3,3 / T,20 | Proper local SO(3) labels in the same joint frames |
| `target_markers`, `target_marker_valid` | T,M,3 / T,M | Root-relative marker labels with a verified common native-wrist reference |
| `marker_bones`, `marker_offsets` | M / M,3 | Frozen, surveyed/fitted-on-calibration attachment model |
| `target_uv`, `target_uv_valid` | T,20,2 / T,20 | Undistorted full-image pixels, not crop/distorted coordinates |
| `target_intrinsics` | T,3,3 | Unnormalized intrinsics corresponding to target_uv |
| `target_label_variance` | T,20,3 | Optional nonnegative Cartesian label variance in m² |

Unsupported labels are omitted; masks default false. The trivial normalized wrist is excluded from joint metrics. Leaf joint rotations do not become valid GT because a skeleton has 20 nodes.

`load_sequence(..., include_targets=False)` does not load supervision arrays. `chunk` returns separate `inputs`, `targets`, and fixed calibration structures. Training masks never become model features.

## Manifest

`manifest.json` has `schema: handcalib.sequence_manifest.v1`, `split_group: subject_id` (or `session_id`) and `sequences`. Each record requires:

`id`, relative `path`, file `sha256`, `split` (train/val/test), `subject_id`, `session_id`, `source_group`, `units: m`, `frame: camera_axes_root_relative`, `calibration_source`, `timing_verified: true`, `geometry_verified: true`, `input_uses_mocap: false`, `crop_source: sensor_only`, and `synthetic: false`.

Also declare `label_type: anatomical_joints` or `surface_markers`, `rgb_feature_dim`, `metric_depth_validated`, `independent_root_gt`, `rotation_gt_validated`, `marker_offsets_calibrated`, `undistorted_uv_validated` as applicable. `max_observation_age_s` defaults to 0.05. These flags attest to a separately documented audit; they do not establish calibration quality by themselves.

Split identities and source hashes cannot appear in different splits. Keep left/right hands and Take006a/Take006b from one capture under the same `source_group`. File hashes are verified before training. Calibration-source values for real data are `independent_calibration` or `train_calibration`; never an evaluation-take fit.

## Packing asynchronous streams

```bash
python -m dataset.pack --spec /local/audited_exports/spec.json --out dataset/local/real_v1
python -m dataset.validate --manifest dataset/local/real_v1/manifest.json
```

The spec schema is `handcalib.raw_export_spec.v1` with `split_group` and `sequences`. Each record contains the manifest metadata above, plus `bundle`, `source_kind: raw_audited_export`, a written `audit_report`, `source_assets` with hashes, and `feature_extractor` declaring `uses_future: false` and `uses_mocap: false`. The packer generates `path` and `sha256`.

The local NPZ bundle contains `time` query times, `offsets`, `glove_acquisition`, `glove_arrival`, `glove` and optional per-packet `glove_valid`. RGB uses `rgb_acquisition`, `rgb_arrival`, `rgb_features` or `rgb`, per-packet `geometry` and optional `rgb_valid`. Depth is analogous. Each stream's arrays share its own packet count, not the query count. All clocks must already be mapped to a common reference.

Supervision arrays use query length T and the `target_*` names above. An exporter may instead supply `bvh_joints_m [T,21,3]`, `bvh_valid [T,21]`, and `reference_source: Skeleton_BVH_FK`; the packer maps BVH 5:21 to glove 4:20 and excludes thumb. Offline interpolation of **labels** is allowed, subject to reference validity; it never enters the observation resampler.

`asof_indices` selects the newest acquisition that has arrived, supports out-of-order arrival, rejects excessive age, and marks repeated held packets non-fresh. It never selects a future nearest neighbour. This implementation rejects delayed observations rather than rewinding and repropagating state. Do not increase age limits or rewrite arrival timestamps merely to inflate RGB coverage.

## Existing repository integration: required agent work

The local proprietary/raw archive -> audited NPZ export adapter is not implemented or tested here because those files are absent from this environment. Use the actual raw parsers after inspecting their current source:

- `gt_calib_viz.py`: clock/solver contracts. Some comments still describe older CMA XYZ; do not use those as positional truth.
- `src/gt_calib_delivery/imu_mocap_comparison.py`: current BVH-FK reference and mapping. Its nearest-row sampler is a **display** sampler, not this causal input sampler.
- `src/gt_calib_delivery/new_capture.py`: CMM tracks and marker semantics, not complete anatomical GT.
- `docs/calibration/IMU_SOLVED_VS_MOCAP_V1.md` and `docs/dataset/NEW_CAPTURE_2026-08-31.md`: current interpretation boundaries.

Never export `guided_ik`, `posterior_75/98`, `rbf_self_fit`, per-frame Kabsch-aligned poses, MoCap-wrist-conditioned inputs, operator Y=-44 mm correction, or lossy Depth.mp4 as independent training observations/labels. Take_007 marker attachments/root correspondence are still a data gate; a virtual wrist is not surveyed anatomical wrist GT.
