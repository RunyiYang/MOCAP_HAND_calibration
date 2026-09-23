# HandCalib model implementation

This is the deterministic first implementation of the research plan. It is not a trained model or a faithful reproduction of every module in the slides. See `todos/IMPLEMENTATION_STATUS.md` for the distinction.

## Forward path

1. Historical local rotations, root, velocities and dt feed a state embedding and two GRUCells (256 hidden each). This path predicts a prior without current observations.
2. Solved glove keypoints feed two native-hand graph message-passing layers (128 channels). Cached visual features are projected to 256 channels and combined with camera/crop geometry. An optional depth+validity CNN produces 128 channels.
3. The 516-dimensional concatenation (256+128+128+3 availability flags+dt) is projected to 256 channels.
4. A 64-hidden slow GRU uses fused features, glove/prior innovation and a 32-dimensional session code. It updates a bounded glove-XYZ observation bias only when both glove and visual/depth evidence are available. This bias is in metres, **not** a physical gyroscope bias. It is held during visual outages.
5. An MLP predicts a tangent-space pseudo-measurement correction and bounded diagonal observation noise. Per-coordinate gains combine that correction with the prior. Rotation updates use SO(3), followed by re-orthogonalization.
6. Forward kinematics decodes 20 joints and joint frames. A first-order Jacobian propagates independent rotation variances into Cartesian coordinate variances. Their empirical coverage must be evaluated.

`rollout(state, offsets, future_dt)` calls the same historical transition repeatedly. Its API cannot accept future RGB, depth or glove measurements. No separate future-pose regressor is substituted for the transition.

## Representation

`geometry.PARENTS` follows the repository's 20-node glove topology. It is a native joint-frame model: `p_j = p_parent + R_parent * offset_j`, `R_j = R_parent * R_local_j`. The root rotation sets reference-frame orientation. The offsets are fixed calibration inputs, never fitted using test MoCap. Root-relative output retains fixed camera axes; it is not per-frame rotation aligned.

The 20 local SO(3) variables are an implementation parameterization, **not** a claim of 60 anatomical degrees of freedom. Leaf rotations are not position-observable. Anatomical hinge/twist constraints and surveyed axes need a later rig adapter. Do not report anatomical angle accuracy from position-only labels.

`joints` is root-translation-normalized; `world_joints` adds the independently predicted root and is actually in the declared camera reference frame. The key name is generic, not proof of external world tracking. Root estimates are only meaningful after independent root supervision and appropriate visual inputs. No test MoCap root is copied into predictions.

## Visual modes

- `cached` (default): externally computed, **sensor-only causal** visual features `[B,T,F]`. The default F is 256. No pretrained weights are bundled. Export HaMeR/UmeTrack-style features separately with the extractor's name, weight provenance, crop geometry, and no-future/no-MoCap audit.
- `image`: a small GroupNorm CNN, trained from scratch. It is a runnable fallback, not HaMeR or a pretrained backbone. Input RGB is `[B,T,3,H,W]` in `[0,1]`.
- `none`: no RGB branch; use for the explicitly named glove-only experiment.

Depth accepts `[B,T,1,H,W]` metres plus a validity mask. This is a feature encoder, not a differentiable rendered-depth loss.

## Inference API

```python
from models.handcalib import HandCalib, ModelConfig
model = HandCalib(ModelConfig()).eval()
# inputs contain only observation tensors; offsets are independent calibration.
outputs, state = model(inputs, offsets, state=None)
future, _ = model.rollout(state, offsets, future_dt)
state = state.detach()  # carry state across TBPTT chunks, do not reset it
```

For checkpoint inference use `python -m train.predict --help`. For deployment, initialize once per sequence/donning and call `step`. This version rejects stale observations in dataset packing; delayed-vision rewind/repropagation is **not implemented**.

## Filter interpretation

The diagonal pseudo-measurement update preserves positive variances and a predict/correct structure. It omits state/velocity/calibration cross-covariance and does not linearize a complete sensor observation model. It is **not** a full augmented error-state EKF or a KalmanNet reproduction. The model's uncertainty is an approximation, not a guarantee.

Design context: `docs/research/HAND_WORLD_MODEL_PLAN.md` and `docs/research/RELATED_WORK.md` (KalmanNet, FSGlove, AVI-HT, recurrent latent dynamics). Software APIs use standard PyTorch modules; no custom CUDA extension is required.
