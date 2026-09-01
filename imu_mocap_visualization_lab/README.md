# IMU -> MOCAP visualization lab

This independent review folder contains 28 videos: four action-review segments
from `thor_new4_20260831_processed`, multiplied by seven offline visualization
methods. Every method draws a pose on every RGB frame. There is no validity-gate
disappearance, stale-color flash, joint-error connector, or per-frame error text.

The four published segments are Take_005, the disjoint first and second halves
of Take_006, and Take_007. They come from the three recordings that contain the
complete RGB + solved 20-joint hand + CMM marker stack. The fourth recording,
`camera_glove_recording_20260831_155410`, has 98 RGB/depth calibration frames but
zero glove packets, no solved hand pose, and no CMM action target; it is recorded
in `manifest.json` as calibration evidence and is not mislabeled as a hand demo.

## Seven methods

1. `s2_continuous`: IMU solver bones resampled at RGB time by S2 SLERP.
2. `s2_gaussian`: IMU-articulation symmetric S2 Gaussian smoothing.
3. `kalman_rts`: IMU-articulation forward Kalman plus backward RTS smoothing.
4. `posterior_75`: MOCAP direction plus 25% smoothed IMU residual.
5. `posterior_98`: MOCAP direction plus 2% smoothed IMU residual.
6. `rbf_self_fit`: full-sequence IMU-pose/velocity to MOCAP RBF neural fit.
   All rendered frames are training frames and there is deliberately no test
   split, following the operator request.
7. `guided_ik`: CMM endpoint-constrained fixed-phalanx constant-curvature IK,
   with one temporally continuous bend plane and bounded joint flexion. This is
   the recommended visual upper bound.

## Interpretation boundary

The last four methods consume synchronized MOCAP targets. The RBF model is
trained and rendered on the same sequence. Those outputs demonstrate how well
the recording can be made to look after teacher conditioning; they are not an
independent IMU accuracy estimate, a held-out evaluation, or evidence of
generalization. All four segments use CMM glove-surface reflectors, not
anatomical joint centers. The Take_005/006/007 marker IDs come from the supplied
#1..#11 ID table and placement photographs. Only the Take_007 left #1
`11781 -> 12503` alias is inferred from trajectory continuity; the exact IDs and
palm-fit residuals are retained in every metrics file.

Open `index.html` through the local review server. `manifest.json`, all 28
method metrics, the 28 synchronized 3D motion assets, `validation.json`, and
`SHA256SUMS.txt` provide the reproducible audit trail. The interactive viewer
uses the selected MP4 as its 30 FPS clock and supports orbit, pan, zoom, timeline
seek, and view reset without an external runtime or CDN. The earlier unmodified comparison remains in
`../imu_mocap_comparison_delivery/` and is not overwritten.
