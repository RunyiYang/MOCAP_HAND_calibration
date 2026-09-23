# Training entry points

Start at [`../todos/AGENT_START_HERE.md`](../todos/AGENT_START_HERE.md). This package is run from a source checkout; the existing delivery package, `pyproject.toml`, `uv.lock` and media are unchanged. The folders `models/`, `dataset/` and `train/` are intentionally not added to the delivery wheel.

## Environment

Use Python 3.11+ and an existing CUDA-compatible PyTorch installation. The code uses standard APIs available in PyTorch 2.4+, but execution in this delivery was tested on **PyTorch 2.10.0 CPU**, not on the cluster's CUDA stack. Do not replace a working CUDA build to satisfy a smoke test.

```bash
python -m venv --system-site-packages .venv-train
source .venv-train/bin/activate
python -m pip install -r train/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest train/tests -q
```

## CPU wiring test

Use fresh output directories. This intentionally optimizes synthetic fixtures for two steps, not real motion.

```bash
python -m dataset.synthetic --out dataset/local/smoke --frames 96
python -m dataset.validate --manifest dataset/local/smoke/manifest.json --allow-synthetic
python -m train.fit --config train/configs/smoke.json \
  --manifest dataset/local/smoke/manifest.json --out train/artifacts/smoke \
  --device cpu --allow-synthetic
python -m train.evaluate --checkpoint train/artifacts/smoke/last.pt \
  --manifest dataset/local/smoke/manifest.json --split test \
  --out train/artifacts/smoke/test.json --forecast-ms 100 250 --allow-synthetic
python -m train.predict --checkpoint train/artifacts/smoke/last.pt \
  --manifest dataset/local/smoke/manifest.json --out train/artifacts/smoke/predictions \
  --allow-synthetic
```

## Real-data pilot (coding agent runs after audit)

```bash
python -m dataset.validate --manifest dataset/local/real_v1/manifest.json
python -m train.evaluate --vendor --manifest dataset/local/real_v1/manifest.json \
  --split val --out train/artifacts/vendor_val.json
python -m train.fit --config train/configs/supervised_warmup.json \
  --manifest dataset/local/real_v1/manifest.json --out train/artifacts/warmup \
  --device cuda --max-steps 1000
python -m train.fit --config train/configs/solved_pose.json \
  --manifest dataset/local/real_v1/manifest.json --out train/artifacts/causal_pilot \
  --init-weights train/artifacts/warmup/best.pt --device cuda --max-steps 1000
```

If there are no valid causal RGB observations, explicitly select `glove_only.json`; do not fake a visual experiment. `cached` requires a separately audited pretrained feature extractor. `rgb_image.json` is a scratch-CNN alternative, not a pretrained model. `rgbd_image.json` requires verified metric depth. Do not use synthetic data in these commands.

## Resume versus curriculum transition

`--resume PATH` restores weights, optimizer, RNG, sequence order, TBPTT cursor and persistent state. It requires the **same config and manifest hash**. The step limit is total steps, not additional steps.

```bash
python -m train.fit --config train/configs/solved_pose.json \
  --manifest dataset/local/real_v1/manifest.json --out train/artifacts/causal_pilot \
  --resume train/artifacts/causal_pilot/last.pt --device cuda --max-steps 2000
```

`--init-weights PATH` loads weights only for a new stage/ablation with compatible architecture, resetting optimizer/state. It requires the same dataset manifest. Use a new output directory. It is mutually exclusive with resume.

## Evaluation

```bash
python -m train.evaluate --checkpoint train/artifacts/causal_pilot/best.pt \
  --manifest dataset/local/real_v1/manifest.json --split val --device cuda \
  --out train/artifacts/causal_pilot/val.json --forecast-ms 100 250 500 1000
python -m train.evaluate --checkpoint train/artifacts/causal_pilot/best.pt \
  --manifest dataset/local/real_v1/manifest.json --split val --device cuda \
  --out train/artifacts/causal_pilot/outage.json --camera-outage 2 1
```

Select models only on validation. Reserve final test evaluation until model selection is frozen. All forecasts discard post-cutoff observations; only the requested integration intervals enter `rollout`. Reference frames are the first timestamp at/after each horizon, not an interpolated synthetic pose. Camera-outage evaluation keeps glove input.

Metrics include mean/median/P95 joint/tip error in fixed axes, separate surface-marker/root errors where valid, output coverage, and marginal Cartesian 95% coverage. There is no Procrustes alignment. Reported runtime includes I/O and is **not** end-to-end acquisition latency. Subject/session bootstrap intervals and recovery-time curves remain TODO.

## Configuration and implemented losses

JSON configs reject unknown model/training fields. Available configs: solved-pose fusion, supervised warm-up, no-slow-state, no-dynamics, fixed-noise, glove-only, scratch-RGB, scratch-RGB-D, and a small CPU smoke fixture.

Losses: masked robust FK joints (fingertips weighted 2x), independent root, calibrated markers, supported local rotations, velocity, undistorted 2D reprojection, solved-glove observation consistency, slow-bias increments, optional diagonal NLL, and no-observation rollout. NLL defaults off; enable only after deterministic fit and label/covariance audit. Depth-surface reconstruction, raw-IMU physics, anatomical joint limits and contact losses are not silently approximated.

The trainer uses **one recording per batch**, with chronological chunks (default 64 frames), preserving state and detaching gradients between chunks. Recordings, not individual windows, are shuffled. It is a correctness-oriented reference trainer, not an optimized multi-stream/DDP pipeline. Measure throughput before expanding the budget.

Outputs: `last.pt`, `best.pt` when a supported validation metric exists, `config.json`, `provenance.json`, `train.jsonl`, and `validation.json`. Checkpoints use atomic rename and safe `weights_only=True` loading. The ignored `train/artifacts/` and `dataset/local/` directories are for local outputs only. No experiment tracker, scheduler, remote training job, or automatic pretrained-weight download is invoked.
