# Implementation status

Prepared 23 September 2026. This is executable first-stage code, not trained research results. The learning files are separate from the pre-existing visualization toolkit.

| Component | Status | Scope |
|---|---|---|
| Native-20 FK and SO(3) updates | Implemented, CPU-tested | Fixed calibration offsets; no anatomical hinge/twist limits yet |
| Glove graph encoder | Implemented | Two message-passing layers, 128 channels; solved keypoints only |
| Visual encoder | Implemented interface + scratch CNN | Cached pretrained features require the agent's audited extractor; no pretrained weights bundled |
| Depth feature encoder | Implemented, branch-tested | Original metric depth required; no rendered-depth loss |
| Fast motion state | Implemented | Two GRUCells; historical state only |
| Slow calibration state | Implemented | Gated, persistent XYZ observation bias; not a physical inertial bias |
| Predict/correct update | Implemented approximation | Learned pseudo-measurement, diagonal uncertainty; not full augmented EKF |
| Future rollout | Implemented, no-future-input API tested | Shared deterministic transition; no future sensor arrays accepted |
| Joint/root/marker/rotation/2D/velocity losses | Implemented | Independently valid masks and frames required |
| Observation/calibration/NLL/rollout losses | Implemented | NLL defaults off; calibration interpretation remains limited |
| Sequence manifests, hashes, split gates | Implemented | Flags require a real external audit; cannot prove source semantics automatically |
| Causal resampling and old BVH mapping | Implemented, tested | Acquisition AND arrival clocks; no future nearest neighbour; excludes thumb |
| Generic NPZ packer | Implemented | Consumes audited raw-export bundles |
| Proprietary local raw-record adapter | TODO: first agent data task | Raw files were not available here; no invented parser or GT |
| Persistent single-stream TBPTT trainer | Implemented, tested | One recording per batch; optimizer/checkpoint/state resume |
| Evaluation and label-free prediction export | Implemented | Fixed-axis EPE, separate marker/root metrics, covariance coverage, outages and forecasts |
| Vendor baseline / no-slow / no-dynamics / fixed-noise | Implemented | Config-controlled ablations; no claim of matched full literature reproduction |
| Plain GRU / AVI-HT reimplementation | TODO | Required stronger causal baselines before research claims |
| Delayed-vision rewind/repropagation | TODO | Current packer rejects overly stale frames; it does not implement out-of-sequence updates |
| Sensor-to-bone raw-IMU physics | TODO | Need raw gyro/accel/placement/noise/calibration data |
| Full state/calibration covariance | TODO | Require observation Jacobians, cross-covariance and observability checks |
| Stochastic RSSM / KL objectives | TODO | Add only after deterministic prediction baseline works |
| Contact / surface rendering / depth residual | TODO | Need valid mesh, visibility and object/contact observations |
| Anatomical rig constraints / thumb mapping | TODO | Verify joint axes and topology, do not invent missing anatomical GT |
| DDP / multi-stream optimization / AMP | TODO | Profile single-stream implementation first; geometry/filter math remains float32 |
| Real-data generalization and CUDA performance | Not tested here | Agent must run on local audited recordings and target hardware |

## Source-preservation boundary

No changes to existing delivery source, raw data, media, manual profiles, `README.md`, `pyproject.toml` or `uv.lock` are required. New code is run as `python -m ...` from the checkout. Local artifacts and datasets have nested `.gitignore` protection. Do not add private captures or generated checkpoints to the repository.

## Validation record

See `todos/reports/CPU_VALIDATION.md` and `todos/reports/validation.json` for executed tests, environment and limitations. These are software-correctness tests on synthetic data, not estimates of hand-tracking performance.
