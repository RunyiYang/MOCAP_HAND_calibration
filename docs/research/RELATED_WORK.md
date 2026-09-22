# Related work and reading notes

Research cut-off: **23 September 2026**. Sources below are original papers or author/publisher pages. “Methods reviewed” means the relevant method text was inspected; “scoped reference” means the abstract/project/publisher description was checked and is used only for the stated high-level role. No experiments were reproduced. Preprints are not presented as peer-reviewed publications. Links are provided rather than redistributed paper PDFs.

## Closest work and consequences for the proposal

| Work | Relevant contribution | Design consequence and important limitation |
|---|---|---|
| AVI-HT [R01] | Adaptive vision/IMU hand fusion with temporal and hand-graph processing, MoCap supervision, UmeTrack/MANO branches. | The closest direct comparator. Fusion alone is already established. Test persistent calibration, real long-session drift, outage recovery and free forecasting, not only frame accuracy. Its acquisition/hardware and participant scale do not automatically transfer to our glove. |
| FSGlove [R02] | Inertial hand tracking with explicit shape-aware sensor/hand calibration. | Separate sensor mounting and hand shape from articulation. Its dense sensor layout and externally supplied global positioning differ from a sparse glove without wrist translation. |
| Self-Calibrated Multi-Sensor Wearable [R03] | Calibration and visual/wearable sensor combination. | Include calibration and confidence-aware classical alternatives. Its stretch sensors plus IMUs are not the same observation set as an IMU-only glove. |
| Visual-inertial hand tracking [R04] | Hand tracking designed around visual/inertial complementarity under interference and occlusion. | Robust sensor fusion predates the proposed project. Evaluate interference, contact and occlusion explicitly instead of claiming these motivations as new. |
| HandCept [R05] | Kinematics-aware visual/inertial estimation, including treatment of delayed vision. | Good filter/latency design reference, but it concerns a rigid robotic dexterous hand, not a deformable wearable human hand. |
| KalmanNet [R06] | A recurrent neural gain embedded in model-based state estimation. | Compare structured learned filtering against unconstrained recurrence. Learning Q/R in an error-state filter is our proposed variant, not an exact KalmanNet reproduction. |

### [R01] AVI-HT: Adaptive Vision-IMU Fusion for 3D Hand Tracking
**Kou et al., arXiv preprint, 2026.** [Paper and methods](https://arxiv.org/html/2605.21714v1). **Methods and experimental protocol reviewed.**

The paper uses a glove/vision pipeline with a recent IMU window and a graph-informed fusion design, with UmeTrack and MANO output branches. Its results include transformed/GT-root-oriented metrics as well as other errors, so matching metric definitions matters. Its time-shift experiment is directly relevant to our synchronization audit. Treat attention fusion as a strong baseline. The proposed distinction is a persistent, explicitly calibrated state and long-duration predictive evaluation, not the presence of an IMU encoder.

### [R02] FSGlove: An Inertial-Based Hand Tracking System with Shape-Aware Calibration
**Li et al., arXiv preprint, 2025.** [Paper and methods](https://arxiv.org/html/2509.21242v1). **Calibration method reviewed.**

DiffHCal addresses sensor installation and personalized hand shape. Use this as a calibration baseline/design reference rather than asking a latent model to absorb arbitrary mounting rotations. The system's dense inertial layout and separate global-positioning setup differ from our currently available solved keypoints. Calibration references and shape constraints should not be conflated with per-frame ground truth.

### [R03] Self-Calibrated Multi-Sensor Wearable for Hand Tracking and Modeling
**Gosala et al., IEEE TVCG, 2023; online publication 2021.** DOI: 10.1109/TVCG.2021.3131230. [Publisher](https://ieeexplore.ieee.org/document/9628050/). **Scoped reference:** publisher/institutional descriptions checked; the full institutional PDF was not accessible in this review.

Relevant to self-calibration and wearable/vision complementarity. The system combines different wearable sensing modalities, so it is not a matched raw-IMU-only baseline without adaptation. Do not attribute detailed loss formulas or exact replication settings to it from an abstract alone.

### [R04] Visual-inertial hand motion tracking with robustness against occlusion, interference, and contact
**Lee et al., Science Robotics 6, eabe1315, 2021.** [Publisher](https://www.science.org/doi/10.1126/scirobotics.abe1315). **Scoped reference.**

An important earlier visual-inertial hand-tracking precedent. It motivates stress tests and establishes that robustness through modality complementarity is not itself a new research claim. Exact reimplementation requires the full system's sensing and calibration details.

### [R05] HandCept: A Visual-Inertial Fusion Framework for Accurate Proprioception in Dexterous Hands
**Huang et al., arXiv preprint, 2025; revised June 2026.** [Paper and methods](https://arxiv.org/html/2505.08213v2). **Estimator/kinematics and delayed-update methods reviewed, including the 2026 revision.**

Relevant ideas are a known articulated model, visual/inertial correction, and updating a past state when a delayed image arrives before propagating forward. The paper’s “latency-free” terminology must not become a claim of zero acquisition or compute delay. Human glove slip and soft-tissue/marker motion create a different observation problem.

## Inertial estimation and calibration foundations

### [R06] KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics
**Revach et al., IEEE Transactions on Signal Processing, 2022.** [Paper](https://arxiv.org/abs/2107.10043). **Methods and architecture diagram reviewed.**

Combines known dynamics with recurrent gain estimation. Its innovation-driven architecture motivates keeping an explicit predict/correct decomposition. Learned gains need not provide perfectly calibrated uncertainty; our proposed covariance-learning filter must still be tested for empirical coverage.

### [R07] VQF: Highly Accurate IMU Orientation Estimation with Bias Estimation and Magnetic Disturbance Rejection
**Laidig and Seel, arXiv preprint, 2022.** [Paper](https://arxiv.org/abs/2203.17024). **Scoped reference.**

Use as a classical orientation-processing baseline when raw inertial measurements become available. It is not a full hand-articulation model and cannot be evaluated as such without a sensor-to-bone/kinematic adapter.

### [R08] On-Manifold Preintegration for Real-Time Visual-Inertial Odometry
**Forster, Carlone, Dellaert and Scaramuzza, arXiv 2015 / IEEE TRO 2017.** [Paper](https://arxiv.org/abs/1512.02363). **Scoped reference.**

Provides a principled treatment of rotation, inertial integration, noise, and bias correction. Its rigid-body VIO formulation is not directly a complete articulated-finger model. Sensor lever arms and per-link motion still need to be represented.

### [R09] Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time
**Huang et al., ACM TOG / SIGGRAPH Asia, 2018.** [Paper](https://arxiv.org/abs/1810.04703). **Scoped reference.**

Supports synthetic inertial-data generation and learned motion priors for sparse sensing. Whole-body coverage, hand articulation, and causal/look-ahead settings differ; do not transfer its accuracy or timing claims to finger tracking.

### [R10] TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors
**Yi, Zhou and Xu, ACM TOG, 2021.** [Paper](https://arxiv.org/abs/2105.04605). **Scoped reference.**

Useful precedent for structured estimation stages. Body translation can exploit support/contact assumptions that are not automatically available to a freely moving hand. Contact must be observed or modeled rather than presumed.

## Hand geometry and visual observation models

### [R11] Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO)
**Romero, Tzionas and Black, ACM TOG, 2017.** [Author publication page](https://dtzionas.com/publication/2017_tog_mano/). **Scoped reference.**

Provides a parametric hand pose/shape surface representation. Use as an optional mesh adapter for depth/silhouette observations, while preserving a validated native skeleton. Mesh landmarks, glove markers, and anatomical centers are distinct. Check model/data licenses before distributing derived assets.

### [R12] Reconstructing Hands in 3D with Transformers (HaMeR)
**Pavlakos et al., CVPR, 2024.** [Publisher paper page](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html). **Scoped reference.**

A strong pretrained image-hand prior and potential offline teacher. It does not replace the proposed temporal calibration state, and bare-hand image training may not transfer cleanly to reflective-marker gloves. Start frozen rather than training a large backbone on a few takes.

### [R13] UmeTrack: Unified multi-view end-to-end hand tracking for VR
**Han et al., SIGGRAPH Asia, 2022.** [Paper](https://arxiv.org/abs/2211.00099). **Scoped reference.**

Relevant to hand kinematics, temporal visual tracking, and camera-aware features. Match the exact skeleton and evaluation convention when comparing UmeTrack-inspired output heads with the repository's 20/21-point systems.

### [R14] A2J: Anchor-to-Joint Regression Network for 3D Articulated Pose Estimation from a Single Depth Image
**Xiong et al., ICCV, 2019.** [Paper](https://arxiv.org/abs/1908.09999). **Scoped reference.**

A useful depth-only observation baseline. Requires meaningful depth measurements; a colorized or lossy 8-bit depth-preview video is not an equivalent input.

### [R15] On the Continuity of Rotation Representations in Neural Networks
**Zhou, Barnes, Lu, Yang and Li, CVPR, 2019.** [Paper](https://arxiv.org/abs/1812.07035). **Scoped reference.**

Motivates continuous network rotation representations. The filter should still operate with valid rotations/tangent-space residuals. A 6D network representation is not six physical rotational degrees of freedom.

## Predictive state-space and motion models

### [R16] Learning Latent Dynamics for Planning from Pixels (PlaNet)
**Hafner et al., ICML, 2019.** [Proceedings and paper](https://proceedings.mlr.press/v97/hafner19a.html). **RSSM and overshooting method/diagram reviewed.**

The deterministic/stochastic recurrent split and explicit observation/prior model are useful. Its planning task includes actions and rewards absent from our data. Latent overshooting is optional; the paper does not establish it as mandatory for all RSSMs. Our first loss should directly validate supervised future hand states.

### [R17] HuMoR: 3D Human Motion Model for Robust Pose Estimation
**Rempe et al., ICCV, 2021.** [Paper](https://arxiv.org/abs/2105.04668). **Scoped reference.**

A conditional generative motion prior that motivates uncertainty-aware transition modeling. Its full-body motion distribution is not a ready-trained finger-motion prior. Transfer would need a compatible hand representation and training data.

### [R18] Mastering Diverse Domains through World Models (DreamerV3)
**Hafner et al., arXiv preprint, 2023.** [Paper](https://arxiv.org/abs/2301.04104). **Scoped reference; cited by its preprint record.**

A contrast for what an action/reward-driven world-model agent entails. It is not a recommendation to introduce reinforcement learning into a sensor-calibration dataset with no action/reward annotations.

### [R19] Deep Kalman Filters
**Krishnan, Shalit and Sontag, arXiv preprint, 2015.** [Paper](https://arxiv.org/abs/1511.05121). **Scoped reference.**

An early learned generative state-space formulation. Useful background for separating transitions, observations, and inference networks rather than calling any temporal regressor a world model.

## Data and later interaction extensions

### [R20] DexYCB: A Benchmark for Capturing Hand Grasping of Objects
**Chao et al., CVPR, 2021.** [Publisher paper page](https://openaccess.thecvf.com/content/CVPR2021/html/Chao_DexYCB_A_Benchmark_for_Capturing_Hand_Grasping_of_Objects_CVPR_2021_paper.html). **Scoped reference.**

Relevant to hand/object perception and handover evaluation. Useful for visual pretraining/domain tests, not as a replacement for this glove's paired calibration captures.

### [R21] GRAB: A Dataset of Whole-Body Human Grasping of Objects
**Taheri, Ghorbani, Black and Tzionas, ECCV, 2020.** [Paper](https://arxiv.org/abs/2008.11200). **Scoped reference.**

Provides captured motion with articulated hands, object geometry/pose, and contact-derived information. Relevant to a later interaction model or compatible motion-prior pretraining. It does not by itself supply the target glove's raw IMU distribution.

### [R22] A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting Hand Pose Estimation from a Single RGB Image
**Jiang et al., CVPR, 2023.** [Paper](https://arxiv.org/abs/2304.03635). **Scoped reference.**

A visual baseline for interacting-hand ambiguity. Consider it when the task moves from one isolated hand to bimanual occlusion. Its RGB formulation is distinct from the original depth-based A2J.

## Reading and reproduction priorities for implementation

Reproduce the dataset contract and simple baselines first. Then prioritize AVI-HT-style matched-input fusion, FSGlove-style explicit calibration where sensor coverage permits, and structured filtering inspired by KalmanNet/HandCept. Only add an RSSM-style stochastic latent once deterministic filtering and genuine free-forecast evaluation work. Obtain the full [R03–R04] system descriptions before claiming a faithful implementation of those methods. Do not claim a “first” visual-inertial hand world model without a wider systematic novelty review.
