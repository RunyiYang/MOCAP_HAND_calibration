# IMU-solved pose vs MOCAP supplemental comparison

This folder is separate from the canonical nine-video delivery. It contains
four same-frame overlays: three old takes compare the unmodified glove solver
world pose against the latest Skeleton_0/1 BVH-FK MOCAP reference; Take_007
compares it against the ten photographed CMM surface markers per hand.

The display uses the nearest observed solver row on every RGB frame. It does
not interpolate, smooth, or hide stale rows. Amber means the row failed the
original 20/25 ms scientific timing gate and is excluded from strict metrics.

Important interpretation boundaries:

- “IMU-solved” means the delivered 20-joint glove solver output derived from
  IMUs. It is not the raw sensor quaternion stream.
- Old takes preserve solver wrist orientation and copy only MOCAP wrist
  translation because the glove output has no global translation.
- Their one fixed rotation and scale is fitted directly on Take01 Skeleton BVH
  non-thumb MCP positions, then frozen for Take02/03. The builder recomputes
  and verifies the tracked profile and its source hashes before rendering.
- Primary old-take metrics cover the unambiguous 16 non-thumb joints. Thumb is
  excluded because the glove has three thumb joints and BVH has four.
- Take_007 uses one fixed base-only coordinate registration fitted on the first
  20% and frozen on the remaining 80%; it never copies per-frame CMM rotation.
- Absolute wrist translation and independent 6DoF accuracy are unavailable.
  CMM markers are glove-surface points, not anatomical joint centers.
- The `[0, -44, 0] mm` operator profile moves both compared layers for display
  and cancels from root-normalized error; it is not a metric improvement.

Open `index.html` through the project review server, or use `manifest.json`,
the per-frame CSV files, and `SHA256SUMS.txt` for reproducible review.
