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
| 07 / 08 | new Take_006 | 22 raw MOCAP markers / solved 20-joint pose |
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
uv run gt-calib-delivery validate --full-decode
uv run gt-calib-delivery publish-web
uv run gt-calib-delivery serve --host 127.0.0.1 --port 8811
```

Open `http://127.0.0.1:8811/final-nine/`. The read-only local server supports
HTTP byte ranges, so all nine MP4 timelines can seek correctly. Publication is
atomic and refuses an invalid delivery, symlinks, build scratch, or an existing
destination. If an approved visual-QA repair replaces a poster or metrics file,
refresh only those auxiliary hashes before validation:

```bash
uv run gt-calib-delivery normalize-provenance
uv run gt-calib-delivery refresh-aux-hashes
```

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
- Take_006 CMM contains anonymous marker trajectories without anatomical joint
  names or topology. Video 07 therefore draws raw points, not a fabricated
  21-joint skeleton.
- Take_006 video 08 estimates root pose from a wrist marker and four palm
  markers; fingertip articulation never enters the fit.
- Video 09 validates RGB PnP camera/world reprojection. Lossy `Depth.mp4` is
  not used as metric depth evidence.
- The videos demonstrate synchronization, projection, and conditioned
  composition. They are not an independent dynamic hand-pose GT accuracy claim.

## Code map

- `src/gt_calib_delivery/`: package CLI, delivery builder, new-capture adapter.
- `gt_calib_viz.py`: old-take MOCAP / solved-pose preparation and renderer.
- `local_review_server.py`: path-safe, read-only HTTP Range server.
- `web/public/final-nine/`: final review UI; reads the delivery manifest.
- `tests/`: rendering, dataset, delivery, webpage, and HTTP contracts.
- `docs/`: daily handoff, calibration protocols, evidence, and data request.
