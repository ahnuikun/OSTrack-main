# History-token server validation plan

This file is the clean, current plan after the local mechanism screen. It supersedes scattered earlier notes when deciding what to train on the server.

## Fixed module direction

Main module candidates:

- H2-real clean history-token.
- H3-recurrent clean history-token.

Auxiliary robustness branch:

- H4 reliability, used as anti-contamination / bad-observation rejection ablation.

Do not promote the current hard-mix H4 recipe as the main module. Local screening showed that hard-mix can create a reliability learning signal, but it hurt clean validation performance.

## Core mechanism

```text
z0: clean initial template anchor
H_t: independent temporal history state written from previous search observations
x_t: current search frame

OSTrack one-stream ViT consumes [Z, H, X].
Search tokens read both template and history through self-attention.
History assists localization without overwriting z0.
```

## Server training matrix

| Role | Variant | Purpose |
| --- | --- | --- |
| Control | H0 | OSTrack / same-sampling no-history control |
| Main | H2 | One-step clean history-token |
| Main | H3 | Recurrent clean history-token |
| Auxiliary | H4 | Reliability / freeze / anti-contamination ablation |

## Required datasets

UAV main claim:

- VisDrone.
- UAVDT.
- UAV123.
- DTB70.

Generality:

- LaSOT.

LaSOT is required. If H2/H3 improves UAV datasets but clearly hurts LaSOT, the paper claim must be framed as UAV-specialized temporal history modeling, not a generally stronger tracker.

## Required metrics

For every dataset:

- AUC.
- Precision.
- Normalized Precision.
- Delta against matched H0.
- FPS.
- Parameters.
- FLOPs.

## UAV hard-case breakdown

Report at least:

- occlusion.
- low resolution.
- similar object.
- fast motion.
- viewpoint change.

For each category, report H0/H2/H3/H4 deltas and include representative success/failure cases.

## Claim boundary

Current local result supports only:

```text
Clean history-token modeling is feasible and worth full training.
```

It does not yet support:

```text
The method solves UAV drift, occlusion, or viewpoint-change problems.
```

Paper-level support requires:

- matched full training on H0/H2/H3/H4;
- gains on UAV datasets;
- no unacceptable degradation on LaSOT;
- hard-case evidence explaining when temporal history helps;
- ablations proving H4 reliability is auxiliary rather than a hidden source of all gains.

## Promotion threshold

The server-stage promotion rule is:

```text
UAV average improvement >= +1.0%
LaSOT degradation <= -1.0%
```

UAV average should be computed over:

- VisDrone.
- UAVDT.
- UAV123.
- DTB70.

Primary metric:

- AUC / Success.

Secondary metrics:

- Precision.
- Normalized Precision.

Interpretation:

- If UAV average AUC improves by at least 1% and LaSOT drops by less than or equal to 1%, the module is acceptable for the UAV-tracking paper direction.
- If UAV improves but LaSOT drops by more than 1%, the claim must be narrowed to a UAV-specialized tracker and the generality claim should be removed.
- If UAV average improvement is below 1%, the module should not be promoted as the main contribution even if individual datasets improve.

## Immediate execution plan before server training

Before full server training, the test-time online path must be closed.

Step 1: online-history inference implementation.

- H0: unchanged OSTrack inference.
- H2: first tracked search runs with `z0` only, writes `H` from the response-peak search-token crop, and subsequent frames use `[Z,H,X]`; after each frame, `H` is replaced by the current clean observation history.
- H3: first tracked search runs with `z0` only, writes `H`; subsequent frames use `[Z,H,X]` and recursively update `H`.
- H4: optional auxiliary path; use learned reliability to blend update vs freeze during inference, without GT.

Step 2: local tracking sanity check.

- Verify that a tracker can initialize and track without shape errors.
- Verify that history tokens are actually created and used.
- Verify that no GT crop is used during inference.
- Verify approximate runtime overhead.

Status on 2026-06-25:

- Random-image forward sanity passed for H2 and H3:
  - first tracked frame runs without history and writes `H`;
  - later frames use `H`;
  - history token shape is `(1, 64, 768)`.
- Real-sequence sanity passed on UAV123 `uav_bike1`:
  - H0: `uav_history_h0_online_sanity_local`, 3085 frames, about 80.19 FPS.
  - H2: `uav_history_h2_real_online_sanity_local`, 3085 frames, about 64.78 FPS.
  - H3: `uav_history_h3_recurrent_online_sanity_local`, 3085 frames, about 62.76 FPS.
- A PyTorch 2.6 checkpoint-loading issue was handled by explicitly loading the local tracker checkpoint with `weights_only=False`.
- A test config mismatch was fixed by setting `DATA.SEARCH.SIZE=256` and `DATA.TEMPLATE.SIZE=128` in the online sanity configs, matching the 10-epoch local checkpoints.
- This sanity check proves that the online history path is executable. It is not a formal performance conclusion.

Step 3: server training.

- Full-train H0/H2/H3 first.
- Evaluate on VisDrone, UAVDT, UAV123, DTB70, and LaSOT.
- Run H4 only after H2/H3 show a useful signal.

Server-stage execution rule:

- Do not use the current `*_local.yaml` configs as paper-level training configs.
- Create server configs from the server OSTrack baseline config used for full training.
- Keep all H0/H2/H3/H4 configs matched in dataset, epoch, batch size, learning rate, scheduler, search/template size, augmentation, seed, and evaluation protocol.
- The only intended architectural difference should be:
  - H0: `MODEL.TOKEN_HISTORY.ENABLE=False`
  - H2: `MODEL.TOKEN_HISTORY.ENABLE=True`, `MODE=h2_real`
  - H3: `MODEL.TOKEN_HISTORY.ENABLE=True`, `MODE=h3_recurrent`
  - H4: `MODEL.TOKEN_HISTORY.ENABLE=True`, `MODE=h4_reliability`, only after H2/H3 pass the useful-signal check

Recommended server order:

1. Train H0 full baseline.
2. Train H2 full model.
3. Train H3 full model.
4. Evaluate H0/H2/H3 on VisDrone, UAVDT, UAV123, DTB70, and LaSOT.
5. Compute UAV average AUC/Success delta and LaSOT delta.
6. Run H4 only if H2 or H3 gives a credible positive signal.
7. Run hard-case breakdown only for the final candidate and matched H0.

Command pattern:

```powershell
D:\Anaconda\envs\track\python.exe -u tracking\train.py --script ostrack --config <server_h0_config> --save_dir . --mode multiple --nproc_per_node <gpu_count>
D:\Anaconda\envs\track\python.exe -u tracking\train.py --script ostrack --config <server_h2_config> --save_dir . --mode multiple --nproc_per_node <gpu_count>
D:\Anaconda\envs\track\python.exe -u tracking\train.py --script ostrack --config <server_h3_config> --save_dir . --mode multiple --nproc_per_node <gpu_count>

D:\Anaconda\envs\track\python.exe -u tracking\test_uav_suite.py --tracker_name ostrack --tracker_param <server_h0_config> --dataset all --threads <threads> --num_gpus <gpu_count>
D:\Anaconda\envs\track\python.exe -u tracking\test_uav_suite.py --tracker_name ostrack --tracker_param <server_h2_config> --dataset all --threads <threads> --num_gpus <gpu_count>
D:\Anaconda\envs\track\python.exe -u tracking\test_uav_suite.py --tracker_name ostrack --tracker_param <server_h3_config> --dataset all --threads <threads> --num_gpus <gpu_count>

D:\Anaconda\envs\track\python.exe -u tracking\analyze_uav_suite.py --tracker_name ostrack --tracker_param <server_h0_config> --dataset all --force_evaluation
D:\Anaconda\envs\track\python.exe -u tracking\analyze_uav_suite.py --tracker_name ostrack --tracker_param <server_h2_config> --dataset all --force_evaluation
D:\Anaconda\envs\track\python.exe -u tracking\analyze_uav_suite.py --tracker_name ostrack --tracker_param <server_h3_config> --dataset all --force_evaluation
```

Paper-level stop rule:

- If neither H2 nor H3 reaches UAV average `+1.0%` with LaSOT drop within `-1.0%`, do not promote history-token as the main contribution.
- If H2 passes and H3 does not, prefer H2 because it is simpler and currently strongest in the local screen.
- If H3 passes with better hard-case behavior, use H3 as the main model and report H2 as a simplicity ablation.
