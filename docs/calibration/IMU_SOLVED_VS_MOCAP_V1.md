# IMU-solved pose vs MOCAP comparison V1

Date: 2026-09-01

## Purpose

This supplement answers two separate questions without mixing them:

1. Keep the delivered glove-solver pose visible on every RGB frame.
2. Measure its 3D difference from the synchronized MOCAP reference only on the
   original strict timing-valid subset.

It does not replace or add a tenth item to the canonical nine-video delivery.
The standalone folder is `imu_mocap_comparison_delivery/`; the local review URL
is `http://127.0.0.1:8811/downloads/imu-mocap/`.

## What the old “validity gate” did

The old renderer accepted a video frame only when the surrounding glove-solver
timestamps met a 25 ms bracket-span limit. Take_007 additionally required the
nearest solver row to be within 20 ms and its camera anchor to be valid. A
rejected frame was changed to NaN and the renderer drew no hand. That policy
created the visible on/off flicker; it was a display decision, not proof that
the saved solver pose itself vanished.

V1 separates display from evaluation:

- Display: choose the nearest observed solver row; no interpolation, no
  smoothing, no extrapolated pose, and never hide it because of the gate.
- Quality annotation: show `solver_sample_age_ms`; timing-rejected frames are
  marked `STALE DISPLAY / NOT SCORED`.
- Scientific metrics: retain the original 25 ms old-take mask or the Take_007
  camera-valid plus nearest-age-at-most-20-ms mask. Stale display rows never
  enter strict metrics.

## What “IMU pose” means here

The videos show the delivered 20-joint glove solver keypoints derived from the
IMU sensors. They do not show the raw per-sensor quaternion/accelerometer/gyro
stream. The old 01/02/03 folders do not contain that raw stream; therefore the
honest label is `IMU-SOLVED`, not `RAW IMU`.

## Old Take 01/02/03 protocol

- MOCAP reference positions: latest requested `Skeleton_0/1.bvh` forward
  kinematics, 21 joints per hand. `Human.cma` supplies only its checked 120 Hz
  timestamps and frame ordinals.
- Solver pose: nearest observed `world_*` 20-joint row, so the solver's own
  wrist orientation is preserved.
- Coordinate registration: one Take01 non-thumb-MCP-only fixed rotation and
  uniform scale fitted directly against the Take01 Skeleton BVH-FK positions,
  then frozen for Take02/03. The tracked profile is
  `calibration_profiles/imu_solver_vs_bvh_take01_registration.v1.json`; its
  source-asset hashes and fitted values are recomputed and checked at build
  time, so the earlier Human.cma-position fit cannot be reused silently.
- Per-frame conditioning: copy only synchronized MOCAP wrist translation,
  because the glove keypoints have zero global translation. MOCAP wrist
  orientation is not copied.
- Metric: wrist-root-normalized 3D EPE over 16 unambiguous index/middle/ring/
  pinky MCP, PIP, DIP, and tip correspondences.
- Excluded: thumb, because glove has three thumb joints while BVH has four;
  absolute wrist translation and absolute 6DoF, because translation is copied.

Strict 25 ms results against BVH-FK, in millimetres:

| Take | Strict frames | Left 16J median / P95 | Right 16J median / P95 | Left / right tip median |
|---|---:|---:|---:|---:|
| 01 | 1275 / 1747 | 32.62 / 121.19 | 30.74 / 125.35 | 54.09 / 51.43 |
| 02 | 1324 / 1805 | 31.97 / 137.67 | 40.12 / 153.73 | 52.06 / 62.35 |
| 03 | 1218 / 1799 | 43.48 / 191.90 | 66.32 / 200.65 | 65.95 / 101.92 |

The median remains moderate while the P95 is very large, particularly on
Take03. The video now exposes those solver-orientation/articulation differences
instead of reducing them by copying MOCAP wrist orientation.

## Take_007 protocol

Take_007 has 11 CMM surface reflectors per hand, not a 21-joint anatomical
skeleton. Marker #1..#10 have photograph-defined tip/base correspondences;
marker #11 constructs the virtual wrist 20 mm toward the forearm.

To avoid the former per-frame CMM rotation conditioning, V1 fits one fixed
rotation and scale on RGB frames `[0, 396)` using only the four non-thumb base
markers. It freezes that transform and evaluates `[396, 1981)`. Each frame
still copies the CMM virtual-wrist translation because glove global translation
is absent. No per-frame CMM rotation is used.

Strict evaluation results over 1088 frames:

| Side | All 10 surface pairs median / P95 | Five tips median / P95 | Five bases median / P95 |
|---|---:|---:|---:|
| Left | 39.03 / 103.78 mm | 50.83 / 117.24 mm | 31.13 / 64.45 mm |
| Right | 41.11 / 110.38 mm | 52.66 / 123.35 mm | 31.49 / 67.54 mm |

These are surface-correspondence residuals after a temporally frozen coordinate
registration. Reflectors are mounted on the glove surface rather than at
anatomical joint centres, so the numbers are not independent anatomical GT or
absolute glove-wrist 6DoF accuracy.

## Display translation and artifact contracts

The supplied manual profile applies `[0, -44, 0] mm` to both the solver-anchored
layer and its MOCAP/CMM reference in each video's own MOCAP world. This is an
operator display correction: it cancels from root-normalized EPE and must not
be interpreted as metric improvement or a cross-session physical extrinsic.
An exact copy is packaged at `calibration/applied_manual_profile.json`; the
manifest records both its in-package hash and its project-source provenance.

The standalone contracts are:

- manifest: `gt_calib.imu_solved_pose_mocap_comparison_delivery.v1`
- old-take metrics: `gt_calib.imu_solved_pose_vs_skeleton_bvh.v1`
- Take_007 metrics: `gt_calib.imu_solved_pose_vs_cmm_markers.v1`
- validation: `gt_calib.imu_mocap_comparison_validation.v1`

The reviewed package contains 23 regular files / 22 checksum entries /
68,661,131 bytes. Its four
MP4 files are H.264/yuv420p/fast-start and passed full decode with frame counts
1747 / 1805 / 1799 / 1981. `validation.json` reports `status=pass` and
`failures=[]`. The current manifest SHA-256 is
`55478a785a8742337cfd5f26768d1c84bfa5751151f86e67fbc0a0697b7dd984`.

For old takes, each per-frame row maps glove indices 4..19 to BVH indices 5..20
in index/middle/ring/pinky MCP/PIP/DIP/tip order. `strict_frames / total_frames`
uses the original 25 ms interpolation-bracket/common-interval mask, but the
error itself is computed on the same nearest observed solver row shown in the
video. For Take_007, frames `[0,396)` fit the fixed base-only registration and
frames `[396,1981)` are the same-recording temporal evaluation split; 1088 of
those 1585 evaluation frames pass the strict camera/20 ms solver-age mask.

## Reproduction and artifacts

```bash
uv run gt-calib-delivery build-imu-comparison \
  --destination rebuilt_imu_mocap_comparison_delivery \
  --manual-profile calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json

uv run gt-calib-delivery validate-imu-comparison \
  --destination rebuilt_imu_mocap_comparison_delivery --full-decode

uv run python web/build_site.py --assemble
uv run gt-calib-delivery serve --host 0.0.0.0 --port 8811
```

Each comparison has an H.264/yuv420p/fast-start MP4, poster, full metrics JSON,
and per-frame CSV containing solver row, sample age, strict-valid flag, and
frame error. `manifest.json`, `validation.json`, and `SHA256SUMS.txt` bind the
complete folder.
