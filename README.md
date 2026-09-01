# MOCAP hand calibration delivery

This UV project turns the delivered RGB, MOCAP, solved-glove, and camera/world
calibration records into one reviewed nine-video package and a browser review
page. Generated media live in `final_9_video_delivery/`; the static page is
`web/public/final-nine/`.

## Final inventory

| Videos | Source | Visualization |
|---:|---|---|
| 01 / 02 | old Take 01 | Skeleton_0/1 BVH-FK 2x21 joints / solved 20-joint pose |
| 03 / 04 | old Take 02 | Skeleton_0/1 BVH-FK 2x21 joints / solved 20-joint pose |
| 05 / 06 | old Take 03 | Skeleton_0/1 BVH-FK 2x21 joints / solved 20-joint pose |
| 07 / 08 | new Take_007 | 20 labeled CMM surface markers / aligned solved 20-joint pose |
| 09 | no-glove 155410 | CS-400 world axes and camera calibration |

Because BVH ordinal 0 is a vendor seed, the formal old-take intervals are RGB
61..1807 (1747 frames), 0..1804 (1805), and 4..1802 (1799). The paired solved
videos use the same intervals.

Every MP4 is H.264/yuv420p, 960x540, 30 FPS, fast-start encoded, and indexed by
`final_9_video_delivery/manifest.json`. `SHA256SUMS.txt` covers every regular
delivery artifact except the checksum file itself. `validation.json` records a
full nine-video decode pass.

## Reproduce and review

System prerequisites: Python 3.11, UV, `ffmpeg`, and `ffprobe`.

```bash
uv sync --frozen --group dev
uv run gt-calib-delivery inspect
uv run gt-calib-delivery build \
  --destination rebuilt_9_video_delivery \
  --manual-profile calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json
uv run gt-calib-delivery validate \
  --destination rebuilt_9_video_delivery --full-decode
cd web && npm run build && cd ..
uv run gt-calib-delivery serve --host 127.0.0.1 --port 8811
```

Open `http://127.0.0.1:8811/final-nine/`. The read-only local server supports
HTTP byte ranges, so all nine MP4 timelines can seek correctly. The web build
validates and atomically mirrors the checked-in final delivery; the lower-level
`publish-web` command remains available for an initially absent destination and
refuses an invalid delivery, symlinks, build scratch, or overwrite. If an
approved visual-QA repair replaces a poster or metrics file,
refresh only those auxiliary hashes before validation:

```bash
uv run gt-calib-delivery normalize-provenance
uv run gt-calib-delivery refresh-aux-hashes
```

The supplemental IMU-solved pose vs MOCAP review is intentionally outside the
canonical nine-video manifest. It keeps the nearest observed solver pose visible
without interpolation or smoothing and retains the old timing mask only for
strict metrics:

```bash
uv run gt-calib-delivery build-imu-comparison \
  --destination rebuilt_imu_mocap_comparison_delivery
uv run gt-calib-delivery validate-imu-comparison \
  --destination rebuilt_imu_mocap_comparison_delivery --full-decode
```

Open `http://127.0.0.1:8811/downloads/imu-mocap/`, or download the standalone
`imu_mocap_comparison_delivery/` folder. See
`docs/calibration/IMU_SOLVED_VS_MOCAP_V1.md` for joint mappings and results.

The same page contains the final-nine manual XYZ workbench. It lets the operator
select all nine videos, records a per-video XYZ residual, and exports
`gt_calib.final_nine_manual_xyz.v1`. Videos 07/08 additionally support live
clean-RGB reprojection and optional left/right residuals; videos 01-06 record
the override for offline rebuilding. Video 09 remains visible but is
fail-closed as excluded because it is no-glove calibration evidence.

The reviewed default is `Y=-44 mm` for videos 01-08. The value is applied
separately in each video's own MOCAP world and does not mean that the 2026-08-29
and 2026-08-31 sessions share one corrected extrinsic. Rebuild the complete
package with the checked-in profile:

```bash
uv run gt-calib-delivery build \
  --destination rebuilt_9_video_delivery \
  --manual-profile calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json

uv run gt-calib-delivery validate \
  --destination rebuilt_9_video_delivery --full-decode
```

The manual offset is an operator display correction recorded as provenance. It
does not rewrite either camera/world calibration, is not a cross-session shared
extrinsic, and must not be reported as independent GT or accuracy evidence. See
`docs/calibration/FINAL_NINE_MANUAL_XYZ_V1.md` for the exact schema and scope.

## Source-data boundary

Large/raw acquisition data are intentionally workspace-local and are not part
of the Git code distribution:

- `同步整理_20260829_三段/`
- `thor_new4_20260831_processed/`
- `movementcap_20260831_worldcalib_tabletop_final.tar.gz`

The extracted calibration package and final reviewed media remain versioned.
See `docs/dataset/NEW_CAPTURE_2026-08-31.md` for the exact recording selection,
clock reconstruction, coordinate convention, marker IDs, and requested missing
data.

## Interpretation boundary

- Old solved-pose videos use glove root-local articulation conditioned on the
  synchronized MOCAP wrist SE(3); they do not test independent glove wrist 6DoF.
- Final videos 01/03/05 now obtain all 21 positions per hand from
  `Skeleton_0/1` BVH forward kinematics. The previous final render incorrectly
  used `Human.cma` joint positions; `Human.cma` is retained only as the verified
  120 Hz timestamp/frame-counter axis for these three MOCAP-only videos.
- The placement photo and spreadsheet identify 11 reflector tracks per hand in
  Take_007. Video 07 draws marker #1..#10: five fingertip/base pairs per hand,
  or 20 measured points total. This is a landmark count, not 20 rotational DoF.
- Marker #11 is the black dorsum-module reflector. It is hidden as a joint and
  only constructs a virtual wrist: `#11 + 20 mm * normalize(#11 -
  mean(#4,#6,#8,#10))`. The 20 mm shift is hand-local, not a world-axis offset.
- Take_007 video 08 fixes translation at that virtual wrist, estimates a
  per-frame Kabsch rotation from all ten photographed correspondences, and uses
  one frozen scale per side. Its residual is same-take fit consistency between
  glove-surface reflectors and anatomical solved keypoints, not independent GT.
- Video 09 is explicitly bound to no-glove recording `155410`, including its
  RGB and its own intrinsics, plus the delivered CS-400 world calibration.
  Lossy `Depth.mp4` is not used as metric depth evidence, and no manual XYZ
  translation is permitted on this video.
- The `Y=-44 mm` setting is a display-only translation independently evaluated
  in each applicable video's MOCAP world. It is not a physical transform shared
  across sessions and does not make the overlays independent hand-pose GT.
- The videos demonstrate synchronization, projection, and conditioned
  composition. They are not an independent dynamic hand-pose GT accuracy claim.

## Code map

- `src/gt_calib_delivery/`: package CLI, delivery builder, new-capture adapter.
- `src/gt_calib_delivery/imu_mocap_comparison.py`: always-visible nearest-row
  IMU-solved pose vs MOCAP supplemental renderer, metrics, and validator.
- `gt_calib_viz.py`: old-take MOCAP / solved-pose preparation and renderer.
- `local_review_server.py`: path-safe, read-only HTTP Range server.
- `web/public/final-nine/`: final review UI and manual XYZ workbench; reads the
  delivery manifest and `calibration-workbench/take007_alignment.json`.
- `calibration_profiles/operator_y_minus_44_all_hand_overlays.v1.json`: reviewed
  final-nine operator profile for videos 01-08; video 09 is explicitly excluded.
- `tests/`: rendering, dataset, delivery, webpage, and HTTP contracts.
- `docs/`: daily handoff, calibration protocols, evidence, and data request.
