# MOCAP hand calibration delivery

This UV project turns the delivered RGB, MOCAP, solved-glove, and camera/world
calibration records into one reviewed nine-video package and a browser review
page. Generated media live in `final_9_video_delivery/`; the static page is
`web/public/final-nine/`.

## Final inventory

| Videos | Source | Visualization |
|---:|---|---|
| 01 / 02 | old Take 01 | 2x21 MOCAP skeleton / solved 20-joint pose |
| 03 / 04 | old Take 02 | 2x21 MOCAP skeleton / solved 20-joint pose |
| 05 / 06 | old Take 03 | 2x21 MOCAP skeleton / solved 20-joint pose |
| 07 / 08 | new Take_007 | 20 labeled CMM surface markers / aligned solved 20-joint pose |
| 09 | no-glove 155410 | CS-400 world axes and camera calibration |

Every MP4 is H.264/yuv420p, 960x540, 30 FPS, fast-start encoded, and indexed by
`final_9_video_delivery/manifest.json`. `SHA256SUMS.txt` covers every regular
delivery artifact. `validation.json` records a full nine-video decode pass.

## Reproduce and review

System prerequisites: Python 3.11, UV, `ffmpeg`, and `ffprobe`.

```bash
uv sync --frozen --group dev
uv run gt-calib-delivery inspect
uv run gt-calib-delivery build --destination rebuilt_9_video_delivery
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

The same page contains the Take_007 manual XYZ workbench. It projects the CMM
and solved-pose layers onto clean RGB, accepts a global world-space XYZ offset
and optional left/right residual offsets in millimetres, and exports
`take007_manual_xyz_calibration.json`. Apply that operator-selected profile to
videos 07/08 with:

```bash
uv run gt-calib-delivery render-new \
  --destination outputs/take007_manual_review \
  --manual-profile /path/to/take007_manual_xyz_calibration.json

uv run gt-calib-delivery build \
  --destination rebuilt_9_video_delivery \
  --manual-profile /path/to/take007_manual_xyz_calibration.json
```

The manual offset is a display calibration and is recorded as provenance; it
must not be reported as independent accuracy evidence.

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
  Lossy `Depth.mp4` is not used as metric depth evidence.
- The videos demonstrate synchronization, projection, and conditioned
  composition. They are not an independent dynamic hand-pose GT accuracy claim.

## Code map

- `src/gt_calib_delivery/`: package CLI, delivery builder, new-capture adapter.
- `gt_calib_viz.py`: old-take MOCAP / solved-pose preparation and renderer.
- `local_review_server.py`: path-safe, read-only HTTP Range server.
- `web/public/final-nine/`: final review UI and manual XYZ workbench; reads the
  delivery manifest and `calibration-workbench/take007_alignment.json`.
- `tests/`: rendering, dataset, delivery, webpage, and HTTP contracts.
- `docs/`: daily handoff, calibration protocols, evidence, and data request.
