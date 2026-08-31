# 2026-08-31 CS-400 camera/world calibration

Status: complete (multi-frame visual marker labeling + metric CS-400 RGB PnP).

- Reference: `camera_glove_recording_20260831_155410` frame 30
- World origin: black vertex-hole axis projected onto the tabletop
- Tabletop world Z: 0 mm
- Marker/black-hole plane world Z: +45 mm
- RGB marker reprojection RMS: 0.294 px
- Fixed-camera validation recordings: 3
- Maximum static-scene probe displacement: 1.986 px
- Approximate preview-depth quantization: 20.257 mm/index
- Approximate point count: 110319

Primary result: `camera_to_world.json`.

The transform is metric and does not depend on Depth.mp4.  PLY files are visual-review artifacts only because the delivered depth video is an 8-bit lossy color preview rather than original 16-bit depth.
