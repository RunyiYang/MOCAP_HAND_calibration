# Data and experiment contract

Proposal, 23 September 2026. No results below are measured. This protocol supplements [the model plan](HAND_WORLD_MODEL_PLAN.md).

## 1. Required sequence record

Each sequence must declare `subject_id`, `session_id`, `donning_id`, `source_take`, `source_hashes`, handedness, hardware/sensor layout, firmware/solver version, and calibration split membership. Store acquisition and arrival times independently for every stream. Store physical units and every transform as `target_from_source`, with axis and handedness definitions.

Separate fields: RGB; original metric depth and invalid mask; intrinsics/distortion; color-depth registration; device-to-reference clock model and uncertainty; solved glove keypoints/rotations with sample age; optional raw gyro/acceleration/orientation/temperature; MoCap native skeleton; measured marker tracks; visibility/residual/re-ID/gap-fill flags; and per-label covariance or confidence when available. Every output label must declare whether it is observed, interpolated, fitted, or virtual.

`raw_imu_available`, `metric_depth_validated`, `anatomical_joint_gt_available`, `rotation_gt_available`, and `independent_root_gt_available` are explicit booleans. A false value disables dependent losses and metrics. Do not substitute zero tensors and count them as valid measurements. Keep raw acquisition data local unless a separate data-release approval exists.

## 2. Coordinate, timing and supervision audit

Use Skeleton_0/1 BVH-FK for old positional references and the checked Human.cma time/frame axis, as specified in the current repository protocol. Old glove points are in meters while MoCap exports may be millimeters. Do not reuse an old eight-parameter rational distortion model as a new five-parameter camera model. Preserve each capture's camera calibration provenance.

An affine device-clock map has `reference_time = alpha * device_time + delta`; resets may require piecewise segments. Calibrate with hardware events where available and report residual uncertainty. The acquisition-to-host latency is not the sensor's physical timestamp. Do not use nominal MP4 FPS as an absolute clock. For an online model, an event is usable only once it has arrived: a nearest sample from the future is forbidden even if an offline viewer displays it.

For Take_007, fit/measure marker attachments independently. Its ten displayed surface correspondences per hand do not define all internal rotations. The virtual wrist is not anatomical wrist GT. Per-frame Kabsch/Procrustes against test MoCap and the operator Y=-44 mm display correction must never improve a deployment score.

## 3. Named task definitions

| Task | Inputs after initialization | Prediction target | Interpretation |
|---|---|---|---|
| FUSION | Available RGB, validated D, and glove events | Current pose/state | Main deployable estimator, no MoCap inputs |
| CAMERA-OUTAGE | Glove events continue; RGB/D hidden for a defined interval | Current pose through outage and recovery | Tests missing-vision filtering, not autonomous prediction |
| GLOVE-ONLY | Glove after a documented visual calibration prefix | Articulation/relative pose and uncertainty | Absolute long-term global tracking not guaranteed |
| FREE-FORECAST | No observations after cutoff t | State at t+100/250/500/1000 ms | Genuine predictive dynamics benchmark |
| GT-ROOT-DIAGNOSTIC | Explicitly supplied MoCap root | Local articulation only | Oracle-conditioned diagnostic; never the main result |
| OFFLINE-ORACLE | Future observations and/or GT-assisted fit | Offline reconstruction | Upper-bound/debugging comparator, labeled noncausal |

All methods within a comparison receive identical initialization, time masks, available input streams, context, and latency allowance. Do not compare a bidirectional smoother to a causal estimator without stating the extra information.

## 4. Splits and calibration access

Split subjects and sessions before extracting windows. Keep all crops/windows from one recording together. In particular Take006a/Take006b are one source capture for splitting. Hold out re-donnings and, when available, glove hardware/firmware variants. Report in-subject/new-session and cross-subject tests separately.

Define a disjoint calibration prefix/recording for each permitted adaptation protocol. For deployment tests, adaptation may use only the announced sensor observations, not test MoCap. A train-time MoCap calibration followed by a frozen transform is a different protocol and must be named. Fit normalization, sensor noise distributions, marker offsets, and learned priors on train/calibration data only. Freeze model selection before final test evaluation.

Current short/tiny collections are for smoke tests and per-session feasibility, not cross-population conclusions. Optical marker labels, geometry fits and virtual joints carry different uncertainty. Measure a label floor from repeatability/independent held-out calibration before targeting sub-label-noise gains.

## 5. Minimum baseline and ablation run order

| Run | Model | Required condition |
|---|---|---|
| 01 | Vendor solved pose with only permitted fixed frame conversion | Do not inject MoCap root |
| 02 | Fixed explicit session calibration | Calibration observations and fit split recorded |
| 03 | RGB / RGB-D hand estimator | Same crop source and global-root convention |
| 04 | Causal error-state EKF or matched causal optimizer | Learned or fixed noise choices documented |
| 05 | Small causal GRU residual corrector | Same input/history budget as proposed model |
| 06 | Strong vision/glove attention fusion inspired by AVI-HT | Mark adaptation versus faithful reproduction |
| 07 | Proposed fast + slow structured model | Same input modalities as 05/06 |
| 08 | 07 without slow calibration state | Tests persistent calibration contribution |
| 09 | 07 without learned transition | Tests dynamics contribution |
| 10 | 07 with fixed noise, without learned reliability | Tests confidence adaptation |
| 11 | 07 without depth / without RGB / without glove | Modality contribution and observability |
| 12 | 07 with white noise only versus measured drift augmentation | Realistic corruption relevance |
| 13 | Solved-pose versus raw-IMU adapter | Only after raw streams are collected |
| 14 | Deterministic versus stochastic latent dynamics | Only after free-forecast baseline works |

The existing `rbf_self_fit` trains on the same sequence it displays; `posterior_75`, `posterior_98` and `guided_ik` consume same-sequence MoCap; `s2_gaussian` and `kalman_rts` use future context. Keep these as explicitly labeled offline/GT-conditioned comparisons. Even the existing IMU-articulation variants retain MoCap-derived spatial calibration/root anchoring.

Do not run this whole grid before the pilot. Gate each stage on a working baseline and valid data. Run three seeds only for selected decisive comparisons, not every early debugging experiment. No training jobs are launched by this documentation.

## 6. Metrics

Report fixed-axis wrist-translation-normalized joint error and fingertip error, with mean, median, P95 and per-finger values. Independent global wrist translation, root orientation, and native joint rotation error require appropriate independent labels. PA-MPJPE is a secondary shape diagnostic only. Keep surface-marker residuals separate from anatomical metrics.

For long sessions plot error versus elapsed time, report catastrophic-error rates and usable-output coverage, and fit a drift slope only if justified by the trajectory. Use uninterrupted 20–30 minute recordings when evaluating such durations; short clips cannot support long-session claims.

Camera outages: 0.1/0.5/1/2/5/10 seconds, stratified by movement speed and hand/object occlusion. Recovery: time after reacquisition to remain inside a preregistered error tolerance for a defined interval. Publish the tolerance and interval rather than tuning them on test data. Free forecasts: 100/250/500/1000 ms with all modalities withheld after cutoff. Report both error and predictive likelihood; stochastic models require calibrated coverage rather than only best-of-N samples.

Uncertainty: NLL, empirical 68%/95% coverage, sharpness, and blockwise normalized innovation/state errors where covariance semantics permit. Include reference-label uncertainty. Latency: acquisition-to-output P50/P95 including transfer, detector, encoders, filter and repropagation; also report compute-only latency separately. FPS is not sensor rate and is not latency.

Use the same scientifically valid label/time mask across methods. A model failure counts as a failure, not as a missing label. Report stale outputs and abstentions with coverage. Bootstrap confidence intervals over subjects or sessions, not millions of correlated frames.

## 7. Release-blocking tests for the coding agent

1. Unit conversions and transform round-trips; SO(3) orthogonality/determinant; quaternion sign invariance; fixed bone lengths under motion.
2. Native joint mapping and thumb exclusion; marker-versus-joint label provenance; missing modality/label disables the corresponding loss.
3. Acquisition/arrival clock tests, timestamp reset handling, no future nearest-neighbor events, and correct dt under packet loss.
4. **Prefix invariance:** modifying any sample after time t must not alter the causal output at t. Repeat with detector/crop, normalization, and interpolation code enabled.
5. **MoCap removal:** deployment outputs remain computable when all test MoCap fields are deleted. Crops, initialization, registration and confidence must still work.
6. **State persistence:** evaluation on chunked versus continuous streams agrees when state is carried; no unintended calibration resets.
7. **Forecast masking:** a test double throws if any post-cutoff RGB/depth/glove observation is accessed in FREE-FORECAST. CAMERA-OUTAGE deliberately permits causal glove access.
8. Positive-definite bounded covariance and stable updates through complete modality loss; calibrated uncertainty validated rather than inferred from attention weights.
9. Dataset split hash check prevents windows from the same subject/session/source capture crossing forbidden boundaries; all calibration fits record their fit IDs.
10. The original delivery manifest, manual display profiles and existing web tools remain unchanged by the new training package.

## 8. Decision criteria

Continue beyond the pilot only if time/geometry tests pass and a simple supervised corrector learns on clean data. Continue with the world-model extension only if prediction/outage tests support it beyond matched recurrent/attention baselines. A prospective 20% fingertip-P95 improvement is an engineering target, not an observed result; adjust it against the measured label floor and application tolerance before freezing the test protocol.
