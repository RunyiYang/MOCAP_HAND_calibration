# HandCalib-WM: calibration-aware hand dynamics from RGB-D and an IMU glove

**Research design, 23 September 2026.** This is a proposed model and training program, not an implemented trainer or a report of trained results. Working title only. The existing repository remains a calibration/delivery toolkit. Literature identifiers [R01–R22] resolve in [RELATED_WORK.md](RELATED_WORK.md). Exact data contracts and experiments are in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md); starting hyperparameters are in [handcalib.proposed.yaml](handcalib.proposed.yaml).

## 1. Recommendation and research claim

Train a **small causal, geometry-constrained probabilistic state-space model** that jointly estimates hand motion and persistent calibration error. Use a fast motion state, a slow calibration state, observation-specific uncertainty, and differentiable forward kinematics. Use optical MoCap as supervision, not as an inference input. Begin with the **IMU-solved-pose stream that actually exists**, then add a raw-IMU observation model when sensor recordings are available.

The scientific hypothesis is that an explicit persistent calibration state improves long-session accuracy, camera-outage behavior, and reacquisition relative to a static calibration, an ordinary recurrent corrector, and a strong vision/IMU attention-fusion model. This is a hypothesis to test, not an established novelty or performance claim. AVI-HT already combines glove IMUs, vision, and MoCap supervision [R01]; merely concatenating modalities or adding attention is not a sufficient distinction. FSGlove and earlier wearable systems also make calibration a central component [R02–R04].

Do not begin with a large video diffusion model, an LLM, or a full Dreamer agent. The available supervision is much closer to articulated state estimation than to video generation or reinforcement learning. Borrow state-transition, observation, and rollout training ideas from latent dynamics [R16–R19], but preserve the explicit hand geometry and measurement process.

### What “world model” means here

The first version is an **observational hand-state world model**: it maintains a belief over the current hand and sensor state, predicts future states, and predicts the sensor observations those states should produce. It must be tested on genuine future prediction, not only per-frame correction. IMU readings are observations of movement, not control actions. Without actions, object state, and interaction signals, do not describe this as an action-conditioned hand–object physics simulator.

## 2. Repository evidence and immediate boundaries

The proposal was grounded in the current `README.md`, `docs/calibration/IMU_SOLVED_VS_MOCAP_V1.md`, `docs/dataset/NEW_CAPTURE_2026-08-31.md`, `DATASET_UNDERSTANDING.md`, `docs/calibration/IMU_VISUALIZATION_LAB_V1.md`, and the existing web-build code. The current README and the supplemental comparison protocol supersede older CMA-position descriptions in `DATASET_UNDERSTANDING.md`.

| Available evidence | Consequence for training and claims |
|---|---|
| Old takes contain solved 20-keypoint glove poses, not raw per-sensor gyro/accelerometer/quaternion streams. | The first model corrects solved poses. Physical gyro/accelerometer bias is not identifiable from these keypoints alone. |
| Old glove poses have no independent global wrist translation. | Estimate translation from visual/depth observations; never silently copy test-time MoCap translation. |
| Old MoCap positional reference is now Skeleton_0/1 BVH forward kinematics. Human.cma provides the checked time/frame axis. | Use the current source hierarchy and verified units, not the superseded positional export. |
| The main reviewed videos use MoCap wrist conditioning. The newer supplemental comparison preserves vendor wrist orientation and copies only MoCap translation. | Neither is an end-to-end independent glove global-translation evaluation. Keep these protocols separately named. |
| Take_007 contains 11 surface reflectors per hand, not an anatomical joint skeleton. Its virtual wrist is constructed from marker positions. | Supervise measured markers through a marker observation model; do not invent full joint-angle or anatomical wrist ground truth. |
| Glove and BVH thumbs have different topology. Current unambiguous mapping is glove 4–19 to BVH 5–20. | Start quantitative articulation tests on the 16 mapped non-thumb points. Add thumb evaluation only after a validated correspondence/retargeting model. |
| `Depth.mp4` is a lossy preview. Metric depth must come from validated original recordings. | Disable metric-depth losses until original depth values, scale, registration, intrinsics, and invalid-pixel semantics are verified. |
| Camera acquisition and host arrival timestamps differ; old recordings also exhibit nominal-FPS timing drift. | Use the audited device-time chain, not frame index divided by 30 or nearest callback time. |
| The reviewed Y=-44 mm translation is an operator display correction. | Never treat it as a physical sensor calibration, improved accuracy, or shared cross-session extrinsic. |

The existing visualization lab also contains an RBF same-sequence teacher-fit, MoCap-conditioned posterior/IK variants, a symmetric Gaussian smoother, and Kalman plus backward RTS. These are useful offline fits/oracles, not established held-out causal calibration models. The RBF already uses learned mapping, but its train/evaluation frames are the same.

These data are useful for a pipeline pilot and failure analysis. Their existence does not establish adequate subject, session, duration, or hardware diversity for training a generalizable large model. Raw capture assets are workspace-local, not available from the Git repository alone.

## 3. Problem formulation

Let observation events arrive asynchronously. At time t, available measurements are

\[
o_t=\{I_t,D_t,\widetilde G_t,\mathcal U_t,m_t,a_t,\Delta t\},
\]

where I is RGB, D is validated metric depth, G is the vendor-solved pose, U is an optional raw-IMU packet set, m contains availability/quality masks, and a contains acquisition age and arrival information. G and U are alternative or explicitly correlated views of the same sensors, not independent evidence by default.

Define a fast physical state x, slow nuisance state d, and session parameters c:

\[
x_t=(T_t^{W\leftarrow H},q_t,\dot q_t,v_t),\qquad
c=(\beta,C_{sensor\rightarrow bone},T^{W\leftarrow C},\alpha,\delta).
\]

Here q follows a **documented native skeleton**, beta contains fixed hand shape/bone lengths, and T is wrist/root pose. Alpha and delta describe device-clock scale/offset relative to the reference clock. Marker attachment offsets are additional calibrated observation parameters for marker-supervised captures. Fix independently calibrated camera geometry and clocks in the first version rather than allowing an unconstrained network to explain everything through extrinsic changes.

In the solved-pose version, d represents root-orientation and articulation correction parameters in a low-dimensional calibrated space. In the raw-IMU version it can additionally contain physical gyro and accelerometer biases. Do not relabel an unconstrained latent correction as a physically measured sensor bias.

The targets are

\[
p(x_t,d_t,c\mid o_{\le t}),\qquad
p(x_{t+1:t+H}\mid o_{\le t}).
\]

The first distribution is causal filtering; the second is forecasting. A forward-kinematic decoder gives

\[
J_{t,j}=p_t+R_t\,\mathrm{FK}_j(q_t,\beta).
\]

A simple transition combines kinematic integration with a learned residual, while calibration evolves more slowly:

\[
x^-_{t+1}=f_{kin}(x_t,\Delta t)\oplus f_\theta(h_t,\Delta t),\qquad
d_{t+1}=d_t+\eta_t.
\]

The operator plus applies rotation updates on SO(3), not by adding Euler angles. A remount/slip detector permits a discrete change in d; a random-walk assumption alone cannot model abrupt re-donning.

### Deployment and observability

**Primary:** RGB-D plus glove at inference; MoCap is absent. **Stress test:** RGB-D drops out while glove packets continue. **Secondary:** brief visual calibration followed by glove-only operation. **Forecasting:** all future observations are withheld.

Absolute global translation and heading cannot be guaranteed indefinitely when the observations contain no independent reference for them. A learned prior can constrain plausible movement, but cannot create new information about an unobservable degree of freedom. Glove-only output should therefore emphasize root-relative articulation, relative motion, and uncertainty. With a moving camera, a camera trajectory/SLAM reference or an explicitly camera-relative task is additionally required. Sparse finger sensors and sparse surface markers also leave some axial rotations and joint configurations ambiguous.

## 4. Architecture to build

```text
RGB + intrinsics/crop transform -> pretrained visual encoder -> visual pose/landmark likelihood
Metric depth + validity         -> small depth/point encoder -> depth likelihood
Solved glove pose OR raw IMUs   -> hand-graph temporal encoder -> glove likelihood
                                      |                         |
previous state + dt -> fast motion GRU -> predicted state/covariance
                                      |                         |
persistent session code + slow calibration GRU <- reliable innovations
                                      |
                  geometry-aware probabilistic correction
                                      |
               root SE(3) + native joint angles + uncertainty
                                      |
                 forward kinematics / optional MANO surface
                                      |
          current hand + future rollout + predicted observations

MoCap -> training losses / optional offline teacher ONLY
```

### 4.1 Observation encoders

**RGB:** begin with frozen pretrained hand features. HaMeR supplies a strong image prior [R12], while UmeTrack offers useful temporal/kinematic ideas [R13]. A large pretrained model can serve as an offline teacher; do not train it from scratch on a few takes. Use a 224–256 pixel hand crop and retain the inverse crop transform, original intrinsics, handedness, and a wider wrist/forearm context. A crop-only network without crop/camera metadata cannot consistently recover global translation. Crops must come from an inference-available detector or previous prediction, not test MoCap.

**Depth:** use a compact depth CNN or a PointNet-style encoder over approximately 256–512 valid hand points, with explicit invalid/foreground masks. A2J is a useful depth-only baseline [R14]. Keep this branch optional. Depth sees the glove surface, not the bare anatomical skin; account for shell thickness and marker protrusions rather than forcing all surfaces to coincide.

**Glove:** for the existing data, encode native keypoints, supported local rotations, temporal differences, masks, and sample age. Avoid numerically differentiating irregular/noisy positions without the timestamps. Start with 128-dimensional tokens and two graph-attention layers over a verified skeleton, followed by a causal temporal encoder. For raw recordings, use one token per sensor with gyro, specific force, supported orientation, sensor ID, placement, dt, temperature if available, and reliability flags. The actual sensor count is a configuration value to verify, not an assumed seven- or sixteen-sensor layout.

Preserve the distinction between node count, landmark count, and rotational degrees of freedom. Internally use native joint parameters and fixed topology; MANO [R11] is an optional mesh/observation adapter, not a license to fabricate missing thumb labels.

### 4.2 Fast and slow memories

**Fast state:** two-layer GRU, hidden width 256, predicts short-term joint/root evolution and process uncertainty. This is a proposed initial size, not a measured optimum. Persistent state is carried through a full recording. During truncated backpropagation, detach gradients at chunk boundaries but do not reset the physical/calibration state.

**Slow state:** a 64-dimensional GRU plus an interpretable correction head stores session-specific offsets. A 32-dimensional session code captures shape and mounting context. Update this state from reliable, sufficiently informative visual/glove innovations; hold it during visual outages and inflate uncertainty. For unsensed or unexcited calibration directions, retain a prior rather than pretending they were estimated. Detect remount/slip events and reset/adapt the appropriate subset of parameters.

This separation is an inductive bias: fast articulation should not be explained by changing hand size or camera extrinsics. It is not an identifiability proof; the ablations must show the slow state is useful.

### 4.3 Correction and uncertainty

For the structured variant, each sensor adapter produces a measurement mean, a forward observation function h, and a positive-definite observation covariance. Use an error-state update

\[
r_t=y_t-h(x_t^-,d_t,c),\quad
K_t=P_t^-H_t^\top(H_tP_t^-H_t^\top+R_t)^{-1},\quad
x_t=x_t^-\oplus K_tr_t.
\]

Learn residual dynamics and bounded Q/R parameters, while keeping the update structure explicit. Include calibration-state uncertainty and relevant cross-covariances in an augmented state or a documented block approximation. Predict Cholesky factors with positive diagonals; use Joseph-form covariance updates and stable float32 manifold math. Start with root/finger blocks rather than an unrestricted dense covariance over every latent feature.

KalmanNet [R06] is an important alternative because it learns a recurrent gain; **the proposed Q/R-learning filter is not an exact reproduction of KalmanNet**. Benchmark a learned-gain version separately. Attention weights must not be presented as calibrated probabilities. If cross-modal feature mixing is used before likelihood construction, account for the resulting correlations rather than counting the same information twice. Do not multiply independent raw-IMU and vendor-pose likelihoods when one was derived from the other.

Process delayed images at their acquisition state and repropagate buffered inertial events to the present. HandCept [R05] provides a useful delayed-observation design reference, but its rigid robotic hand setting does not establish wearable-human performance. Report actual acquisition-to-output latency, not a “zero-latency” claim.

### 4.4 Stochastic dynamics, only after the deterministic baseline

Add a 32-dimensional stochastic latent with a history-conditioned prior and a training posterior when multi-modal future prediction is actually needed. Train decoded future states and observation predictions. PlaNet/RSSM [R16], HuMoR [R17], and deep state-space work [R19] motivate this extension. There is no need for a reward head, policy, or imagined-action optimization in the current dataset. PlaNet's latent overshooting is an optional method, not a prerequisite of every RSSM.

A practical first model therefore has a small trainable fusion/dynamics core, targeting roughly 1–5 million parameters excluding the pretrained image encoder. Count the implemented parameters and profile actual memory/latency before allocating larger runs.

## 5. Loss design

Apply all losses through valid masks and normalize by valid counts, physical scales, and dimensions. Missing modalities or unsupported label types receive zero weight, not fabricated values. Prefer a staged objective over activating every term at initialization.

### 5.1 Supervised geometric pose

For reliable anatomical references, use robust joint loss through FK:

\[
L_J=\frac{\sum_{t,j}m_{tj}w_j\,\rho((\hat J_{tj}-J^*_{tj})/s_J)}{\sum_{t,j}m_{tj}w_j},
\]

with fingertip weight initially 2 and s_J initially 10 mm. Evaluate wrist-root-normalized articulation and unaligned global root/pose separately. A valid root loss must compare independently inferred root translation/orientation, not an output anchored by MoCap. Use an SO(3) geodesic loss for calibrated bone/root rotations where those labels genuinely exist; do not use Euler-angle MSE. Continuous 6D network rotation outputs [R15] may be converted to matrices, with local tangent updates in the filter.

### 5.2 Surface-marker observation loss

For Take_007-like data, use

\[
\hat m_{tk}=T_t^{W\leftarrow bone(k)}\,o_k,\qquad
L_M=\sum_{t,k}v_{tk}\,\rho(\hat m_{tk}-m^*_{tk}),
\]

or calibrated mesh-attachment coordinates. Attachment offsets o_k are fitted/measured on a separate calibration segment and then frozen, with realistic tolerances for glove deformation. Distal/base markers do not uniquely label all intermediate joints. Optical marker fitting can supervise visible surface trajectories while latent joint uncertainty remains. A virtual wrist constructed from these markers is not independent anatomical root GT.

### 5.3 Observation consistency

Use robust 2D reprojection of supported hand landmarks, plus a visibility-aware rendered-depth/point-to-plane term when metric depth is valid. Retain real camera distortion and use only visible foreground surfaces; do not penalize occluded back surfaces against an object in front of the hand. An optional silhouette loss is more manageable than asking the first model to reconstruct every RGB pixel.

For raw sensors, an explicit observation head predicts

\[
\hat\omega_i=\omega_i(x,c)+b_i^g,\qquad
\hat a_i=R_{W\leftarrow S_i}^{\top}(\ddot p_{S_i}-g)+b_i^a.
\]

The acceleration is at the sensor location, including lever-arm effects. Derivative targets require timestamp-aware filtering and label-noise handling. Noise scales should be measured from static/dynamic recordings. For pose-only data, replace these equations with a calibrated solved-pose observation model `G = h_G(x,d,c) + noise`; do not claim gyro bias supervision. The raw-IMU physics term is disabled in the first configuration.

### 5.4 Dynamics and calibration regularization

Match predicted joint/root velocity to trustworthy MoCap-derived velocity. Penalize acceleration error only where differentiation is reliable. Do not simply minimize velocity or acceleration: that can make a still hand score well while lagging real motion. Fixed bone lengths belong in FK; use weak anatomical joint-limit penalties. Add contact/nonpenetration only with valid object geometry/contact evidence. Sliding contact is not a zero-velocity anchor.

Regularize slow drift increments using dt and their process covariance, and keep shape/mount parameters consistent within a session. Permit detected change points. Synthetic corruption with known offsets can directly supervise calibration; real latent corrections are only weakly interpretable without independent sensor-bias measurements. Keep independent extrinsics/clock parameters fixed in the MVP to reduce confounding.

### 5.5 Uncertainty and predictive rollout

After robust deterministic warm-up, use negative log likelihood for selected state/observation residual blocks:

\[
L_{NLL}=\tfrac12 e^\top\Sigma^{-1}e+\tfrac12\log|\Sigma|.
\]

Include estimated label covariance, constrain variance floors/ceilings, and validate coverage on held-out data. The log-determinant prevents unlimited variance inflation from being a free solution. Avoid treating NLL plus the same Gaussian squared error as independent probabilistic evidence; a robust auxiliary pose loss is a deliberate optimization choice.

Train 100, 250, 500, and 1000 ms rollouts from an observed prefix:

\[
L_{roll}=\sum_{k\in\mathcal H}\gamma_k\,\ell(\hat x_{t+k\mid\le t},x^*_{t+k}).
\]

All future observations are removed for this loss. A separate missing-camera loss retains future causal glove packets and tests filtering under occlusion. Scheduled sampling and longer persistent chunks reduce teacher-forcing mismatch. Optional stochastic dynamics adds an annealed KL between posterior and history-only prior; begin around 0 to 1e-3 after averaging by latent dimension, then tune validation.

### Starting weights, not claimed optimal values

For dimensionless mean-reduced components: joint/marker observation 1.0 each where supported; independent root translation 1.0; rotation 0.2; velocity 0.1; 2D reprojection 0.05; metric depth 0.05; raw sensor consistency 0.1 when available; rollout 0 to 0.2; calibration regularization 0.01; limits 0.01. These weights are a starting proposal. Normalize positions by 10 mm, angles by 5 degrees, and pixels by 5 pixels; calibrate sensor residuals by measured noise. Measure gradient scales on a pilot before selecting final weights. NLL heads replace the corresponding deterministic likelihood objective after warm-up, with any auxiliary robust loss explicitly recorded.

## 6. Staged training plan and acceptance gates

### Stage A: audited data and reproducible baselines

Build a sequence manifest containing subject, session, re-donning, hardware/firmware, source take, clocks, acquisition/arrival timestamps, intrinsics/extrinsics, units, valid masks, skeleton mappings, marker definitions, and GT conditioning flags. Preserve raw/observed, interpolated, and inferred values separately. Raw data remain local with hashes; do not publish participant video by default.

Verify transforms, device clocks, native hand axes, bone lengths, marker attachments, and depth registration using independent calibration records. Quantify a label-error floor. Separate static offset, mounting error, drift, timestamp error, and motion-dependent solver error instead of assuming all deviations are cumulative IMU drift.

Implement vendor output, fixed session calibration, vision/depth-only, a causal EKF or equivalent optimizer, and a small causal GRU residual corrector. VQF is an orientation baseline only when raw data exist [R07]. A future-using RTS smoother or same-take GT-guided IK is an oracle/offline baseline, never a causal competitor.

**Gate:** overfit a short clean segment, pass geometry/timestamp/causality tests, and reproduce held-out baseline metrics without MoCap inputs. Do not launch larger training if the data path fails.

### Stage B: supervised causal fusion

Freeze pretrained visual features; train the glove encoder, fast GRU, calibrated observation heads, and FK decoder. Start on 2–4 second sequences and existing valid non-thumb supervision. Use AdamW, fusion learning rate 3e-4, weight decay 1e-4, gradient clipping 1, and a batch target of 16–32 short sequences subject to memory. Fine-tune selected image layers only later at about 1e-5.

Validate after a small 5k–10k-step pilot rather than pre-authorizing a large sweep. Cache image features for early iterations. A single 24–48 GB GPU is a provisional budget for this small-core regime, not a measured requirement. Record actual memory, throughput, parameter count, and training steps. Use three seeds for decisive final comparisons.

**Gate:** improve over fixed calibration and the small recurrent baseline on a held-out session without simply increasing latency or dropping difficult frames.

### Stage C: persistent calibration and world-model training

Enable the slow calibration state, structured uncertainty, synthetic measured-error augmentation, contiguous modality outages, and predictive rollouts. Extend chunks to 2–8 seconds and train over 30–60 second persistent streams using truncated backpropagation. Carry state across chunks. Include longer real sessions to learn/test minute-scale drift; short windows alone cannot establish this ability.

Corrupt with session mounting rotations, slowly varying bias, abrupt slips, timestamp offset/skew, sensor loss, RGB occlusion/blur, and depth holes. Estimate distributions from training recordings. White-noise corruption alone is not a realistic drift benchmark. Synthetic raw IMUs should be generated from sensor-mounted kinematics, not joint-center acceleration; synthetic vendor outputs require an actual solver or an explicitly labeled pose-corruption model.

**Gate:** improve held-out camera-outage/recovery behavior and genuine open-loop forecasting over no-calibration/no-dynamics ablations. Verify uncertainty increases appropriately when observations disappear.

### Stage D: raw-IMU extension and deployment

Collect raw gyro, acceleration, orientation when available, sensor IDs/placements, calibration flags, firmware, temperature, and acquisition/arrival clocks. Introduce physical observation losses and sensor-to-bone calibration. Compare solved-pose and raw-IMU variants with matched capture splits and available inputs. An offline bidirectional teacher may help denoise training targets or distill a smaller causal student, but the deployed model never sees future frames or MoCap.

Deploy an asynchronous event loop with delayed-image correction, repropagation, outage confidence, and reset handling. Only adapt a small calibration state online under trustworthy evidence; unconstrained self-training on the model's own outputs risks reinforcing drift.

### Proposed acquisition scale

A pilot could use 5–8 participants, three re-donnings each, and 15–20 minutes per recording, approximately 4–8 hours. A broader study could expand to 20–30 participants, three sessions, and 20–30 minutes each, approximately 20–45 hours. These are proposed collection budgets, not claims about the current dataset. Include uninterrupted long recordings, varied hand sizes, fast and slow movements, finger individuation, thumb opposition, bimanual occlusion, interaction, re-donning, and changed sensor environments. Capture without visible optical markers as well, where consent and the reference setup permit, to test marker-appearance shortcuts.

## 7. Evaluation and ablations

Primary accuracy: root-translation-normalized joint/tip error in fixed axes, mean/median/P95, and supported rotation error. Report independent unaligned wrist translation and root orientation separately when valid reference exists. Per-frame Procrustes alignment must not be the primary drift metric. Marker residuals remain separate from anatomical joint error.

Report long-session error curves, catastrophic-error frequency, usable-output coverage, and drift slope only where an increasing trend is actually present. Test camera outages at 0.1, 0.5, 1, 2, 5, and 10 seconds with glove input retained, plus genuinely all-input-free forecasts at 100–1000 ms. Measure recovery time after vision returns, uncertainty calibration/coverage, and actual end-to-end latency P50/P95. Keep time-valid evaluation masks common across methods; model failure must not silently remove hard frames.

Split by subject/session before windowing. Keep Take006a/b together because they originate from one capture. Fit registration and attachment offsets only on the designated calibration/train data. Do not let test MoCap define crops, root pose, coordinate registration, normalization, online resets, or confidence gates. An explicitly reported GT-root-conditioned articulation diagnostic can coexist with the deployment benchmark but must not replace it.

Required ablations: no slow state; fixed rather than adaptive calibration; no learned transition; no uncertainty adaptation; RGB-only; glove-only; RGB+glove without depth; solved-pose versus raw sensors; white noise versus measured drift augmentation; recurrent fusion versus AVI-HT-style attention; deterministic versus optional stochastic prediction. Use matched inputs, context, latency, and initialization. See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for run order and tests.

A prospective engineering target could be at least a 20% reduction in fingertip P95 relative to the strongest matched causal baseline, without worse output coverage or unacceptable latency. This is a proposed acceptance criterion to revise after the pilot, not a promised or observed result. Report confidence intervals over subjects/sessions, not over highly correlated frames alone.

## 8. Why this may work, and when it may fail

The modalities fail differently: vision supplies spatial constraints when the hand is visible; glove signals supply local motion when vision is missing; learned dynamics constrain short-term transitions; persistent calibration separates repeatable sensor error from motion. A structured observation model uses these constraints without requiring the neural network to rediscover hand geometry. This reasoning motivates the experiment; existing fusion/calibration results [R01–R06] do not guarantee success on this hardware.

Likely failures include poor timestamp alignment, loose/glove-deforming markers, unobserved joint twist, fast object occlusion, changing camera pose, unseen re-donning, magnetic disturbances, noisy derivative targets, and excessive reliance on a learned motion prior. Long periods without an independent spatial reference remain uncertain. Be willing to conclude that a classical filter or a simpler residual model is sufficient; a world-model claim should earn its complexity through forecasting and recovery tests.

For a later interaction-centered world model, add object geometry/SE(3), contact or tactile information, task/control variables, and interaction outcomes. GRAB and DexYCB [R20–R21] provide relevant representation/data precedents, but neither supplies this glove's synchronized raw sensor distribution.

## 9. Implementation handoff

Recommended new modules, not yet implemented: `src/handcalib/data/{manifest,clocks,skeleton,markers}.py`, `models/{vision,glove,dynamics,calibration,filter,fk}.py`, `losses.py`, `train.py`, and `evaluate.py`. Keep them separate from `src/gt_calib_delivery/` until contracts are tested.

The first coding milestone is the data contract plus five small baselines, not a large trainer. The YAML is a design configuration only and should fail closed on unavailable raw IMU/depth/labels. No training jobs, dataset uploads, checkpoint publication, or benchmark results are authorized or implied by this planning document.
