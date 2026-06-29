# History Token Experiment Execution Log

本文件专门记录 `TOKEN_HISTORY` 主线实验的固定执行协议、配置、命令、状态和结果。后续不得在未记录的情况下临时改变轮次、采样数、输入设定或评价口径。

## 0. Runbook-derived execution rules

- Repo: `D:\PyCharm\Projects\OSTrack-main`
- Python: `D:\Anaconda\envs\track\python.exe`
- Training entry: `tracking\train.py`
- Reference runbook: `EXPERIMENT_RUNBOOK.md`
- Training logs: `output\logs`
- Checkpoints: `output\checkpoints\train\ostrack\<config>`
- A process being launched is not equal to a valid experiment. Each run must be verified from log lines containing epoch-level `train:` / `val:` output.
- If a run fails, inspect the newest log first, fix only the scoped config/code/path issue, then rerun the same experiment.
- Do not delete checkpoints or logs unless explicitly requested.

## 1. Fixed local quick-training protocol

Unless the user explicitly changes the protocol, all local mechanism-screening training runs use:

- `TRAIN.EPOCH = 10`
- `TRAIN.LR_DROP_EPOCH = 7`
- `TRAIN.SAMPLE_PER_EPOCH = 200`
- `VAL.SAMPLE_PER_EPOCH = 50`
- `TRAIN.BATCH_SIZE = 2`
- `TRAIN.NUM_WORKER = 0`
- checkpoint policy: keep latest and best only
- purpose: local mechanism screening, not final paper conclusion

Canonical command:

```powershell
D:\Anaconda\envs\track\python.exe -u tracking\train.py --script ostrack --config <config> --save_dir D:\PyCharm\Projects\OSTrack-main\output --mode single --use_lmdb 0 --use_wandb 0
```

## 2. Current module line

Main design being tested:

- `z0` is the fixed identity anchor and must not be polluted.
- `H_t` is an independent temporal history state.
- Current search reads `[Z, H, X]` through OSTrack one-stream self-attention.
- `H_t` is generated/updated from encoded target-region search tokens.
- Reliable observations update `H_t`; unreliable observations should be rejected or frozen in later reliability experiments.

## 3. Experiment ladder and status

| ID | Config | Purpose | Required protocol | Status | Result |
| --- | --- | --- | --- | --- | --- |
| H0-future | `uav_history_h0_future_local` | matched control: same future sample but ignore history | 10 epoch | done | best val IoU 0.90315, last 0.90160 |
| H1-oracle | `uav_history_h1_oracle_local` | GT observation upper-bound diagnosis | 10 epoch | done | best/last val IoU 0.90390 |
| H2-real | `uav_history_h2_real_local` | response peak observation creates history | 10 epoch | done | best/last val IoU 0.90435 |
| H0-recurrent | `uav_history_h0_recurrent_local` | matched recurrent control | 10 epoch | done | best/last val IoU 0.90223 |
| H3-recurrent | `uav_history_h3_recurrent_local` | recurrent history update over multiple searches | 10 epoch | done | best/last val IoU 0.90405 |
| H4-reliability-clean | `uav_history_h4_reliability_local` | learned reliability on clean near-neighbor samples | 10 epoch | done | best/last val IoU 0.90673; reliability target collapsed to 1.0, so bad-observation rejection not proven |
| H4-hardprobe | `uav_history_h4_reliability_hardprobe_local` | diagnostic all-hard bad-sample probe | 3 epoch diagnostic only | done | reliability target dropped below 1, proving clean H4 lacked bad samples; not comparable to 10-epoch ladder |
| alpha-0 | `uav_history_h2_alpha0_local` | HISTORY_SCALE=0 diagnostic lower-bound | 10 epoch | done | best/last val IoU 0.90624; note: not strict H0 because zeroed history token slots still exist |
| alpha-0.5 | `uav_history_h2_alpha05_local` | test whether partial history scale changes learning/use | 10 epoch | done | best/last val IoU 0.90403 |
| alpha-1.0 | `uav_history_h2_alpha1_short_local` | test full history scale under same 10-epoch protocol | 10 epoch | done | best/last val IoU 0.90689 |
| hard-mix | `uav_history_h4_reliability_hardmix_local` | mixed normal + hard observations; test whether reliability target lowers without all-hard destruction | 10 epoch | done | best val IoU 0.78049 at epoch 4, last val IoU 0.69761; final reliability mean 0.85351, target 0.82000 |

## 4. Immediate execution order

1. Run `uav_history_h2_alpha05_local` for 10 epochs and record best/last validation IoU.
2. Run `uav_history_h2_alpha1_short_local` for 10 epochs and record best/last validation IoU.
3. Run `uav_history_h4_reliability_hardmix_local` for 10 epochs and record:
   - best/last validation IoU
   - `history_reliability_mean`
   - `history_reliability_target`
   - whether target is meaningfully below 1.0
4. Only after these three are recorded, decide whether to continue H4 reliability or return to history writer design.

## 5. 2026-06-25 alpha scale and hard-mix result

Executed according to the fixed 10-epoch local protocol.

| Config | HISTORY_SCALE | Hard mix | Best val IoU | Last val IoU | Main observation |
| --- | ---: | --- | ---: | ---: | --- |
| `uav_history_h2_alpha0_local` | 0.0 | no | 0.90624 | 0.90624 | Diagnostic lower-bound; not strict H0 because zero-valued history token slots still pass through attention. |
| `uav_history_h2_alpha05_local` | 0.5 | no | 0.90403 | 0.90403 | Partial scaling did not improve the local validation result. |
| `uav_history_h2_alpha1_short_local` | 1.0 | no | 0.90689 | 0.90689 | Full history scale is slightly best in this local screen; no evidence that explicit down-scaling is needed. |
| `uav_history_h4_reliability_hardmix_local` | 1.0 | 25% hard mix | 0.78049 | 0.69761 | Reliability target is no longer all 1.0, so hard observations are present; however tracking quality drops heavily, so this hard mix is useful for reliability diagnosis but too destructive as a default training recipe. |
| `uav_history_h4_reliability_hardmix_trainonly_local` | 1.0 | 25% train-only hard mix, clean val | 0.89069 | 0.87360 | Training reliability target drops to 0.785 while clean-val target remains 1.0. This is the valid diagnostic setting for learning reject/freeze from hard observations without corrupting validation. |
| `uav_history_h4_reliability_hardmix_light_trainonly_local` | 1.0 | 10% train-only hard mix, center 4.5, scale 0.4 | 0.89940 | 0.87867 | Too weak as a bad-observation constructor by the final epoch: train target 0.965, close to clean H4; clean-val performance still drops. |
| `uav_history_h4_reliability_hardmix_mid_trainonly_local` | 1.0 | 15% train-only hard mix, center 5.0, scale 0.45 | 0.89165 | 0.89001 | Hits the desired reliability-target band: final train target 0.915 and mean 0.90893, but clean-val IoU remains below H2/H3/H4-clean. |

Current interpretation:

- `alpha_H` should not be treated as the main contribution. The one-stream transformer already learns how much to read from `Z` and `H`; explicit `HISTORY_SCALE` is only a diagnostic knob for now.
- Hard-mix validates the user's suspicion that the earlier clean H4 samples were mostly good observations. Once bad samples are injected, `history_reliability_target` drops meaningfully below 1.
- The present hard-mix setting is too harsh for performance training. It should be refined into a lighter UAV-style difficult-sample curriculum rather than used directly as the final module-training configuration.

## 6. Hard-mix validation fix

Issue found after reading the runbook and tracing the processing path:

- `DATA.HARD_MIX` was originally stored in shared `settings`.
- Both train and validation `STARKProcessing` received the same `settings`.
- Therefore `uav_history_h4_reliability_hardmix_local` applied hard-mix to validation too.

Fix applied:

- `lib/train/base_functions.py` now creates a shallow `settings_val` copy for validation.
- `settings_val.hard_mix_enable = False`.
- Result: hard-mix is a train-time difficult-observation constructor by default; validation remains clean and comparable.

New valid diagnostic config:

- `experiments/ostrack/uav_history_h4_reliability_hardmix_trainonly_local.yaml`

Result:

- best clean-val IoU: 0.89069 at epoch 2
- last clean-val IoU: 0.87360 at epoch 10
- final train reliability mean/target: 0.80125 / 0.78500
- final clean-val reliability mean/target: 0.95820 / 1.00000

Interpretation:

- The training stream now contains bad or difficult observations.
- The reliability target has a real learning signal on training data.
- Current hard-mix strength is still too strong for best clean-val tracking performance, so the next refinement should reduce hard-mix strength or schedule it rather than adding more gates.

## 7. Light/mid hard-mix follow-up

Question being tested:

- Can light hard-mix make `history_reliability_target` lower than 1 while keeping clean-val IoU close to H2/H3?
- Does `history_reliability_mean` move with the target?
- Does H4 become more stable than H2/H3 instead of gaining only on clean samples?

Results:

| Config | Train hard mix | Best clean-val IoU | Last clean-val IoU | Final train reliability mean / target | Final val reliability mean / target |
| --- | --- | ---: | ---: | --- | --- |
| H2-real | none | 0.90435 | 0.90435 | n/a | n/a |
| H3-recurrent | none | 0.90405 | 0.90405 | n/a | n/a |
| H4-clean | none | 0.90673 | 0.90673 | 0.98815 / 0.96500 | 0.99437 / 1.00000 |
| H4-light | 10%, 4.5, 0.4 | 0.89940 | 0.87867 | 0.96252 / 0.96500 | 0.98737 / 0.96000 |
| H4-mid | 15%, 5.0, 0.45 | 0.89165 | 0.89001 | 0.90893 / 0.91500 | 0.97724 / 0.98000 |
| H4-heavy train-only | 25%, 7.0, 0.6 | 0.89069 | 0.87360 | 0.80125 / 0.78500 | 0.95820 / 1.00000 |
| H4-mid aux-only | 15%, 5.0, 0.45, hard tracking weight 0 | 0.89106 | 0.88984 | 0.89315 / 0.89500 | 0.97689 / 0.98000 |
| H4-mid delay6 | 15%, 5.0, 0.45, start epoch 6 | 0.88807 | 0.86794 | 0.91684 / 0.92000 | 0.96935 / 0.98000 |

Interpretation:

- H4-mid proves the reliability branch can receive and follow a non-trivial target. `0.90893` vs target `0.91500` is a meaningful learned signal.
- H4-light is too close to clean H4 by target, yet still unstable on clean validation, so simply adding weak hard-mix is not enough.
- H4-mid/heavy lower the target, but clean-val IoU is below H2/H3. This means the current hard-mix is helping reliability supervision but hurting the tracking objective.
- Current result supports keeping reliability learning as a diagnostic/auxiliary branch, but does not yet support using hard-mix H4 as the main performance module.

Next technical direction:

- Do not increase hard-mix further.
- Do not add more gates.
- Separate reliability learning pressure from the main box-regression path more carefully, for example by reducing reliability loss weight, delaying hard-mix until the writer has warmed up, or using reliability supervision only for `r_t` while keeping the tracking loss on clean/future frames.

## 8. Aux-only hard-mix trial

Implementation:

- `lib/train/data/processing.py` now records `search_hard_mix` for each search frame.
- `lib/config/ostrack/config.py` adds `MODEL.TOKEN_HISTORY.HARD_TRACK_LOSS_WEIGHT`.
- `lib/train/actors/ostrack.py` uses `search_hard_mix` to down-weight tracking loss for hard search frames.
- New config: `experiments/ostrack/uav_history_h4_reliability_hardmix_mid_auxonly_local.yaml`.
- Setting: mid hard-mix with `HARD_TRACK_LOSS_WEIGHT = 0.0`.

Result:

- best clean-val IoU: 0.89106
- last clean-val IoU: 0.88984
- final train hard track weight mean: 0.80750
- final train reliability mean / target: 0.89315 / 0.89500
- final val reliability mean / target: 0.97689 / 0.98000

Interpretation:

- The hard sample mask is active: tracking loss is indeed reduced on hard-mix samples.
- Reliability learning remains meaningful and follows the target.
- However clean-val IoU is not recovered compared with `H4-mid` without aux-only weighting.
- Therefore the degradation is not only caused by hard samples contributing to the box loss. It is likely also caused by hard observations entering the history writer/update path and changing the representation learned by the module.

Next:

- Test delayed hard-mix only if the training loop can expose epoch to data processing cleanly.
- If delayed hard-mix is awkward, use a simpler two-stage recipe on the server later: pretrain H4-clean, then short reliability-only fine-tune with hard-mix.

## 9. Delayed hard-mix trial

Implementation:

- `DATA.HARD_MIX.START_EPOCH` added to config defaults.
- `lib/train/trainers/ltr_trainer.py` writes the current epoch into processing settings before each loader cycle.
- `lib/train/data/processing.py` enables hard-mix only when `current_epoch >= START_EPOCH`.
- New config: `experiments/ostrack/uav_history_h4_reliability_hardmix_mid_delay6_local.yaml`.
- Setting: mid hard-mix starts at epoch 6.

Result:

- best clean-val IoU: 0.88807
- last clean-val IoU: 0.86794
- final train reliability mean / target: 0.91684 / 0.92000
- final val reliability mean / target: 0.96935 / 0.98000

Interpretation:

- Delayed hard-mix preserves the desired reliability-target signal.
- It does not recover clean validation performance.
- Compared with H4-clean, H2-real, and H3-recurrent, delayed hard-mix is still worse in this local mechanism screen.

Current stop-rule conclusion:

- Hard-mix is useful as a diagnostic for whether reliability can learn reject/freeze.
- Current hard-mix variants should not be promoted as the main module-training recipe.
- For the paper-facing direction, keep the clean history-token path as the main mechanism and treat reliability/hard-mix as an auxiliary robustness study unless later full-server training reverses the result.

## 10. Clean history stability repeats and H4 reliability ablations

Purpose:

- Return to the clean history-token mainline after hard-mix diagnostics.
- Check whether the current local result is reproducible under the fixed 10-epoch protocol.
- Separate the effect of learned reliability from the effect of simply adding one recurrent history update.

Fixed local protocol:

- `TRAIN.EPOCH = 10`
- `TRAIN.LR_DROP_EPOCH = 7`
- `DATA.TRAIN.SAMPLE_PER_EPOCH = 200`
- `DATA.VAL.SAMPLE_PER_EPOCH = 50`
- `TRAIN.BATCH_SIZE = 2`
- `TRAIN.NUM_WORKER = 0`
- `DATA.SEARCH.NUMBER = 3`
- no hard-mix
- keep latest + best checkpoints

Implementation for ablations:

- Added `MODEL.TOKEN_HISTORY.RELIABILITY_MODE`.
- Added `MODEL.TOKEN_HISTORY.FIXED_RELIABILITY`.
- Added `MODEL.TOKEN_HISTORY.USE_RELIABILITY_LOSS`.
- H4 learned path is unchanged by default.
- Fixed-reliability ablations still use the same history writer and same `[Z,H,X]` one-stream ViT path.

Results from current comparable code state:

| Config | Meaning | Best val IoU | Last val IoU | Final train rel mean / target | Final val rel mean / target |
| --- | --- | ---: | ---: | --- | --- |
| `uav_history_h0_future_repeat_local` | matched no-history control, same multi-search sampling | 0.90315 | 0.90160 | n/a | n/a |
| `uav_history_h4_reliability_repeat1_local` | H4 clean learned reliability repeat 1 | 0.90673 | 0.90673 | 0.98815 / 0.96500 | 0.99437 / 1.00000 |
| `uav_history_h4_reliability_repeat2_local` | H4 clean learned reliability repeat 2 | 0.90673 | 0.90673 | 0.98815 / 0.96500 | 0.99437 / 1.00000 |
| `uav_history_h2_real_repeat_local` | one-step real history, verify s1 | 0.90689 | 0.90689 | n/a | n/a |
| `uav_history_h3_recurrent_repeat_local` | recurrent H1 from s0, H2 from s1, verify s2 | 0.90642 | 0.90642 | n/a | n/a |
| `uav_history_h4_reliability_fixed1_local` | H4 with fixed reliability = 1.0 | 0.90642 | 0.90642 | 1.00000 / 0.96000 | 1.00000 / 1.00000 |
| `uav_history_h4_reliability_fixed05_local` | H4 with fixed reliability = 0.5 | 0.90602 | 0.90602 | 0.50000 / 0.96000 | 0.50000 / 1.00000 |
| `uav_history_h4_reliability_noloss_local` | learned reliability forward, no reliability BCE loss | 0.90616 | 0.90616 | 0.98740 / 0.96000 | 0.99478 / 1.00000 |

Notes:

- `uav_history_h0_future_repeat_local` was added after the H2/H3/H4 repeats to close the current-code control loop. It reproduces the earlier H0 result, so the H0 baseline is stable under the current code state.
- The two H4 repeats are numerically identical, showing the current local chain is deterministic. They are reproducibility checks, not random-seed stability checks.
- Earlier `uav_history_h2_real_local` and `uav_history_h3_recurrent_local` logs were lower (`0.90435` and `0.90405`), but the current repeat configs match their YAML text and run higher under the current code state. For current comparisons, use the repeat/ablation group above.
- `H4 fixed=1.0` equals `H3 recurrent` at `0.90642`, which means blindly committing the second observation does not explain the learned-H4 edge.
- `H4 fixed=0.5` and `H4 no-loss` are both below learned H4. This supports that the learned reliability supervision is doing something useful locally, but the margin is small.
- `H2-real repeat` is the highest value in the current clean local table (`0.90689`), with a +0.00374 best-IoU gain over the current H0 matched control. H4 learned reliability is +0.00358 over H0 but slightly below H2.

Strict-reviewer interpretation:

- Current clean local screen supports the feasibility of adding independent temporal history tokens to OSTrack.
- It does not yet prove a paper-level advantage for recurrent reliability or anti-drift.
- For paper-facing claims, the next evidence must come from full server training and UAV benchmarks, with matched H0/H2/H3/H4 controls and hard-case breakdowns.

## 11. Fixed mainline after local mechanism screen

Current fixed direction:

- Main module candidates:
  - `H2-real clean history-token`: one-step response-derived history state.
  - `H3-recurrent clean history-token`: recurrent history-state update across nearby search frames.
- Auxiliary robustness branch:
  - `H4 reliability`: anti-contamination / bad-observation rejection ablation, not the primary performance module at this stage.
- Not promoted as main branch:
  - hard-mix H4 training recipes, because they produced a reliability signal but reduced clean validation performance in local screening.

Why this is the safer paper-facing route:

- H2/H3 directly test the core temporal modeling hypothesis: previous search observations form an independent history state `H`, and current search tokens read `Z + H` through OSTrack's one-stream ViT.
- H4 can still support the anti-drift story, but only as a controlled robustness component after H2/H3 establish the clean history mechanism.
- This avoids turning the method into a pile of gates before proving the central history-token mechanism.

Server full-training validation plan:

1. Train matched controls and module variants:
   - H0: original OSTrack / same-sampling no-history control.
   - H2: clean one-step history-token.
   - H3: clean recurrent history-token.
   - H4: reliability as auxiliary anti-contamination ablation.
2. UAV-focused datasets for the main claim:
   - VisDrone.
   - UAVDT.
   - UAV123.
   - DTB70.
3. Generality dataset:
   - LaSOT, used to verify that the module is not only overfitted to UAV scenes.
4. Required reporting:
   - AUC / Precision / Normalized Precision for every dataset.
   - Delta against matched H0 for every dataset.
   - FPS / parameters / FLOPs.
   - hard-case breakdown on UAV data: occlusion, low resolution, similar object, fast motion, viewpoint change.
   - failure cases where history helps and where history hurts.

Current claim boundary:

- Acceptable now: "local mechanism screening supports clean history-token modeling on OSTrack."
- Not acceptable yet: "the method solves UAV drift/occlusion/viewpoint-change problems."
- Needed for paper: consistent gains on UAV datasets, no unacceptable LaSOT degradation, and hard-case evidence showing why temporal history helps.

Server promotion threshold:

- UAV average AUC/Success improvement across VisDrone, UAVDT, UAV123, and DTB70 should be at least +1.0%.
- LaSOT AUC/Success degradation should stay within -1.0%.
- If this holds, the module is good enough for the UAV-tracking paper direction.
- If UAV improves but LaSOT drops by more than 1.0%, the method should be framed as UAV-specialized rather than generally stronger.

## 12. Online inference path closure and local tracking sanity

Purpose:

- Close the gap between training-time history-token experiments and test-time tracking.
- Ensure H2/H3 do not rely on GT crops during inference.
- Verify that history is generated from the model response peak and then used by later frames.

Implementation updates:

- `lib/test/tracker/ostrack.py`
  - Initializes clean template tokens `z_tokens` when `MODEL.TOKEN_HISTORY.ENABLE=True`.
  - Keeps `history_tokens` as an online tracker state.
  - First tracked frame uses only `z0`, then writes `H` from encoded search tokens around the response peak.
  - H2 uses the latest response-derived observation as the next history state.
  - H3 recurrently updates history through the learned history writer.
  - H4 can optionally blend update/freeze through learned reliability, but it remains auxiliary.
  - Uses `weights_only=False` for local checkpoint loading under PyTorch 2.6.
- `lib/config/ostrack/config.py`
  - Adds `TEST.CHECKPOINT` so local sanity configs can explicitly point to the intended checkpoint.
- Online sanity configs:
  - `experiments/ostrack/uav_history_h0_online_sanity_local.yaml`
  - `experiments/ostrack/uav_history_h2_real_online_sanity_local.yaml`
  - `experiments/ostrack/uav_history_h3_recurrent_online_sanity_local.yaml`

Important fix:

- The first H2 real-sequence run exposed a checkpoint/model mismatch:
  - checkpoint `pos_embed_x`: 256 search tokens;
  - constructed model: 400 search tokens.
- Cause:
  - sanity configs set `TEST.SEARCH_SIZE=256`, but model construction still used default `DATA.SEARCH.SIZE=320`.
- Fix:
  - set `DATA.SEARCH.SIZE=256`, `DATA.SEARCH.FACTOR=4.0`, `DATA.TEMPLATE.SIZE=128`, and `DATA.TEMPLATE.FACTOR=2.0` in H2/H3/H0 online sanity configs.

Random-image debug sanity:

| Variant | First tracked frame | Later tracked frames | History shape |
| --- | --- | --- | --- |
| H2 | no history, writes H | uses H | `(1, 64, 768)` |
| H3 | no history, writes H | uses H and recurrent update | `(1, 64, 768)` |

Real-sequence sanity:

Dataset / sequence:

- UAV123 `uav_bike1`
- 3085 frames

Commands:

```powershell
D:\Anaconda\envs\track\python.exe -u tracking\test.py ostrack uav_history_h0_online_sanity_local --dataset_name uav123 --sequence 0 --threads 0 --num_gpus 1
D:\Anaconda\envs\track\python.exe -u tracking\test.py ostrack uav_history_h2_real_online_sanity_local --dataset_name uav123 --sequence 0 --threads 0 --num_gpus 1
D:\Anaconda\envs\track\python.exe -u tracking\test.py ostrack uav_history_h3_recurrent_online_sanity_local --dataset_name uav123 --sequence 0 --threads 0 --num_gpus 1
```

Results:

| Config | Role | Result file lines | FPS |
| --- | --- | ---: | ---: |
| `uav_history_h0_online_sanity_local` | same-checkpoint H0 sanity baseline | 3085 | 80.19 |
| `uav_history_h2_real_online_sanity_local` | online H2 history | 3085 | 64.78 |
| `uav_history_h3_recurrent_online_sanity_local` | online H3 recurrent history | 3085 | 62.76 |

Interpretation:

- H0/H2/H3 all completed the same real UAV123 sequence.
- H2/H3 are slower than H0, as expected, because `[Z,H,X]` adds history tokens to the one-stream ViT path.
- The result only proves execution correctness and rough overhead on one local sequence.
- It does not prove tracking performance improvement.

Next required stage:

- Move to server full training and matched evaluation.
- Train/evaluate H0, H2, and H3 first.
- Only run H4 reliability after H2/H3 show a useful full-training signal.
- Current project entrypoints verified on 2026-06-25:
  - training: `tracking/train.py`
  - UAV/LaSOT test suite: `tracking/test_uav_suite.py`
  - UAV/LaSOT analysis: `tracking/analyze_uav_suite.py`
- The older runbook mentions `tracking/run_module_experiment.py`, but that file is not present in the current project snapshot, so do not use that entrypoint unless it is restored later.
