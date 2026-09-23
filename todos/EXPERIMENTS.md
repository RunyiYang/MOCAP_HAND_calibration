# Experiment queue

Run sequentially, not as a pre-authorized sweep. No task below authorizes automatic cluster submission.

| Order | Experiment / deliverable | Command/config | Gate |
|---|---|---|---|
| 0 | Learning-package correctness | `python -m pytest train/tests -q` | All tests pass |
| 1 | Local capture audit | `DATA_AUDIT.md` | Verified clocks, units, native frames, labels, splits |
| 2 | Raw export and contract validation | `dataset.pack`, `dataset.validate` | No input GT; no future samples; hashes match |
| 3 | Vendor validation | `train.evaluate --vendor --split val` | Shared label/time mask and output coverage |
| 4 | Short-segment train-fit diagnostic | Separate diagnostic export | Finite gradients; model can learn the data path |
| 5 | Supervised warm-up | `supervised_warmup.json`, max 1,000 steps | Held-out validation with no silent GT alignment |
| 6 | Causal model | `solved_pose.json`, warm-start, max 1,000 steps | Main pilot report before more compute |
| 7 | No slow state | `no_slow.json` | Same data, initialization and step budget |
| 8 | No learned dynamics | `no_dynamics.json` | Compare forecasting and recovery, not just pose EPE |
| 9 | Fixed noise | `fixed_noise.json` | Pose and uncertainty calibration both reported |
| 10 | Glove-only | `glove_only.json` | No absolute 6DoF claim |
| 11 | Scratch RGB / validated RGB-D | `rgb_image.json` / `rgbd_image.json` | Do not mislabel scratch encoder as pretrained |
| 12 | Strong recurrent/attention baseline | New implementation | Match inputs, history, calibration and latency |
| 13 | Larger multi-session training | Revised approved budget | Multiple real sessions/donnings; training-data growth justified |
| 14 | Raw-IMU/full-filter extension | Separate method/config | Raw measurements and physical model independently validated |

## Reporting protocol

Use the strongest matched causal baseline. Report mean, median and P95 joint and fingertip error, output coverage and per-session results. Keep surface-marker errors separate from anatomical errors. Foreground runtime is not measured sensor latency. Add acquisition-to-output profiling before a real-time claim.

Evaluate free prediction at 100/250/500/1000 ms with all future inputs absent. Camera outages keep glove packets; test 0.1/0.5/1/2/5/10 seconds, where recording length permits. Recovery-time criteria, session bootstrap intervals and long-duration drift curves require additional evaluator work, not an invented headline score.

Select on validation, freeze a configuration, then evaluate test once. Use three seeds for the decisive final comparisons, not for early debugging. Distinguish camera-relative/root-relative/global metrics. Do not use per-frame Procrustes to hide calibration errors. Do not reuse Take006a/b or opposite hands from one source recording across train/test.

## Proposed acceptance, not promised performance

Prioritize reduced P95 fingertip error without reduced coverage, higher latency or poorer free forecasts. The earlier 20% figure was a prospective engineering target; it is not an observed result or a required outcome to manufacture. Compare gains with the measured reference-label noise floor.
