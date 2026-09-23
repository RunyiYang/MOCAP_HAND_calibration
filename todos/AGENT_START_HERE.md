# Coding-agent handoff

Repository: RunyiYang/MOCAP_HAND_calibration. Work from its root. Implement and train the hand-calibration model in `models/`, `dataset/`, and `train/`. Preserve the existing calibration/delivery code and reviewed media.

## Operating constraints

Do not submit a sweep, distributed job, Slurm job, paid API request, or background training process automatically. Begin with tests and one bounded foreground pilot. Use at most one GPU and a 1,000-step cap for the first real-data run; report results before extending it. Never rewrite or delete source recordings. Do not upload raw videos, personal paths, checkpoints, or caches to GitHub. Keep local outputs under ignored `dataset/local/` and `train/artifacts/`, or an explicitly chosen external data directory.

Do not claim that synthetic fixture scores are model results. Do not call the diagonal correction a full EKF or the learned XYZ bias a physical gyro bias. Do not silently promote missing labels, marker centers, a virtual wrist, or MoCap-conditioned inputs to independent ground truth.

## 1. Inspect and test before changing code

Read these files first:

- `models/README.md`, `dataset/README.md`, `train/README.md`.
- `todos/IMPLEMENTATION_STATUS.md`, `todos/EXPERIMENTS.md`.
- `docs/calibration/IMU_SOLVED_VS_MOCAP_V1.md`.
- `docs/dataset/NEW_CAPTURE_2026-08-31.md` and the relevant local raw loader source.

```bash
git status --short
python -c "import torch, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
python -m pytest train/tests -q
```

Use the current CUDA-matched environment or an isolated `.venv-train`. Do not upgrade the cluster's PyTorch/CUDA merely because a newer build exists. No changes to `uv.lock` or delivery dependencies are needed for this source-checkout training package.

Acceptance: all learning-package tests pass. Record Python/PyTorch/CUDA/GPU versions and current git revision in `todos/reports/ENVIRONMENT.md`. Run the existing repository tests separately if their dependencies/data are present; distinguish unavailable data tests from actual regressions.

## 2. Audit local recordings (required before any real training)

Locate the original data under the user's existing workspace; do not assume a hard-coded filesystem path. Create a local inventory of subject, session, donning, hand, source take, hardware/firmware, RGB/depth files, solver streams, BVH/CMM references, clocks, and calibration files. Record their hashes in the export provenance, not their contents in Git.

For old takes, positional GT is `Skeleton_0/1.bvh` forward kinematics. `Human.cma` supplies the checked time/frame axis, not the superseded XYZ labels. Start with the 16 non-thumb correspondences: glove 4:20 to BVH 5:21. Preserve the vendor's wrist orientation; no per-frame reference rotation. Global glove wrist translation is absent.

For Take_005/006/007, CMM gives surface markers, not a complete anatomical skeleton. Do not train joint-angle GT from those points. Marker supervision needs frozen attachment offsets and a validated common native-wrist reference. The constructed virtual wrist and Y=-44 mm display offset do not provide that calibration.

Check original metric depth. Do not use Depth.mp4. Check camera intrinsics/distortion and corrected acquisition timestamps, including the documented acquisition/poll latency. Do not infer absolute time from nominal MP4 FPS. Keep `Take006a` and `Take006b`, both hands, and all windows from one capture in one split.

Acceptance: write `todos/reports/DATA_AUDIT.md` with usable sequence counts, label types, clock uncertainty, independent calibration sources, missing prerequisites, and the exact initial train/val/test split. **Do not fill audit booleans with true just to make validation pass.** If subject identities are unknown, report a cross-take/session pilot, not cross-subject generalization.

## 3. Implement the local raw-export adapter

Add an adapter under `dataset/` using the current raw readers. The generic packer already exists; the proprietary archive -> audited bundle adapter is the main remaining data integration task. Do not copy a renderer's final pose arrays without tracing their conditioning.

Export one hand per recording to a local NPZ bundle:

- Unconditioned solved points, their acquisition/arrival clocks, validity, in metres.
- Fixed camera-axis conversion and native FK offsets from independent or training calibration.
- RGB images or causal frozen visual features, camera/crop geometry, and acquisition/arrival times.
- Validated metric depth only when available.
- BVH-FK labels and masks at output times, or calibrated surface-marker labels kept separate.
- Source hashes, audit report, split identities and feature-extractor provenance.

Use existing `dataset.alignment.asof_indices` for observation selection, not the visualization's nearest-future sampler. Repeated held packets must not create repeated independent updates. Keep labels in `target_*`, never in observation fields. Label interpolation is permitted with visibility/gap constraints; sensor interpolation may not read the future.

Cached visual features must come from RGB alone with inference-available crops. Record feature dimension, checkpoint hash/license, normalization and crop convention. The default dimension is 256; change `rgb_feature_dim` consistently if necessary. Do not use MoCap-generated crops or target-derived features on real data. The small image CNN is a fallback baseline, not a pretrained hand encoder.

If camera delay exceeds the conservative observation-age threshold, either run an explicitly glove-only pilot or implement/test delayed-state rewind/repropagation. Do not relabel acquisition as arrival or simply apply a late frame to a current-state target while claiming correctly timed fusion.

```bash
python -m dataset.pack --spec /local/audited_exports/spec.json --out dataset/local/real_v1
python -m dataset.validate --manifest dataset/local/real_v1/manifest.json
```

Acceptance: inspect multiple nonadjacent frames per recording, compare topology/units/time identity, verify target removal does not change inputs, and save a written audit. The initial model must run without any test MoCap fields.

## 4. Establish a bounded baseline and warm-up

```bash
python -m train.evaluate --vendor --manifest dataset/local/real_v1/manifest.json \
  --split val --out train/artifacts/vendor_val.json
python -m train.fit --config train/configs/supervised_warmup.json \
  --manifest dataset/local/real_v1/manifest.json --out train/artifacts/warmup \
  --device cuda --max-steps 1000
```

Before the full pilot, verify a very short clean segment can be fit in a **separately labeled train-fit diagnostic**. Never reuse that train-fit sequence as the held-out validation set or publish its score as generalization. Do not run a broad grid to debug bad geometry.

Acceptance: finite losses/gradients; fixed bone lengths; documented latency/input masks; no catastrophic regressions against raw/fixed-calibration baselines. Report joint/fingertip mean, median and P95 plus coverage. No per-frame Procrustes alignment. Record model parameter count, throughput, peak GPU memory and actual steps.

## 5. Train the causal calibration/dynamics model

After the warm-up passes:

```bash
python -m train.fit --config train/configs/solved_pose.json \
  --manifest dataset/local/real_v1/manifest.json --out train/artifacts/causal_pilot \
  --init-weights train/artifacts/warmup/best.pt --device cuda --max-steps 1000
python -m train.evaluate --checkpoint train/artifacts/causal_pilot/best.pt \
  --manifest dataset/local/real_v1/manifest.json --split val --device cuda \
  --out train/artifacts/causal_pilot/val_forecast.json --forecast-ms 100 250 500 1000
python -m train.evaluate --checkpoint train/artifacts/causal_pilot/best.pt \
  --manifest dataset/local/real_v1/manifest.json --split val --device cuda \
  --out train/artifacts/causal_pilot/val_outage.json --camera-outage 2 1
```

The trainer carries state through chronological chunks; do not shuffle individual windows or reset the slow state every chunk. `--resume` restores the optimizer/cursor/state and requires identical configuration and manifest. `--init-weights` is for a new stage and resets them. Preserve this distinction.

Acceptance: compare against no-slow and no-dynamics variants only after the main pilot has a valid data path. Add a matched pure recurrent/AVI-HT-style fusion baseline before claiming the structured model is necessary. Use validation for model selection, reserve test data for a frozen final comparison.

## 6. Report, then request the next compute budget

Write `todos/reports/PILOT.md` containing: commit/config/manifest hashes, split identities, command, hardware, actual steps, training/validation curves, joint/tip metrics, coverage, forecasts, outage behavior, failures and one justified next experiment. Do not promise a paper-quality result or a fixed improvement before observing it.

Do not begin raw-IMU physics, stochastic RSSM, full covariance filtering, or multi-GPU training until their specific data/engineering prerequisites in `IMPLEMENTATION_STATUS.md` are met. No cluster training was run when this code was added.
