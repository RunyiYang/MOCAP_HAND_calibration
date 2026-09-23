# CPU software validation

Executed 23 September 2026, using the new learning-package source. This is not a real-data experiment.

- `python -m pytest train/tests -q`: **25 passed**, 0 failed (4.70 seconds in this environment).
- CLI checks passed: synthetic generation, manifest validation, two-step CPU fit, held-out synthetic evaluation with forecasts, and label-free prediction export.
- Interrupted/resumed optimization produced exactly the same CPU model weights as four uninterrupted steps.
- A four-frame synthetic train-fit test reduced its objective in 25 local optimizer iterations.
- Default cached-feature model: **1,547,129 parameters**. No pretrained visual backbone weights are included in that count or bundled with this code.

Tests cover SO(3) validity/zero-angle gradients, fixed FK bone lengths, marker attachments, causal-prefix invariance, chunk/stream equivalence, physical removal of target arrays, forecast API isolation, uncertainty growth during rollout, slow-state hold during visual outage, complete modality loss, finite backward passes, RGB/depth branches, acquisition/arrival as-of selection, stale/duplicate packets, split leakage, source hashes, marker-versus-joint semantics, generic packing, checkpoint resume and synthetic fit.

The environment and file hashes are recorded in [validation.json](validation.json). Execution used PyTorch 2.10.0+cpu. Compatibility with the user's existing PyTorch/CUDA environment must be checked on the actual machine; no upgrade is implied.

No real captures, GPU jobs, Slurm jobs, distributed jobs, external trackers or paid APIs were used. The original delivery regression suite was not run in this separate source-only test workspace. Synthetic test metrics must not be reported as calibration performance.
