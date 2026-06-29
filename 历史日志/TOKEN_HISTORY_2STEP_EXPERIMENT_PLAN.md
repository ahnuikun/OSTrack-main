# Token History 2-Step Experiment Plan

Updated: 2026-06-29

## Fixed Objective

Build and validate the smallest 2-step token-history experiment for OSTrack:

```text
z0 fixed
H_t lightweight token state
s0 writes H1
H1 assists s1
loss = mean(loss_s0, loss_s1)
```

This is a fixed-length temporal unroll. It is not candidate rescue and it is not dynamic RGB template replacement.

## Current Implementation State

As of this plan, the repo is still baseline OSTrack plus local compatibility fixes. The 2-step token-history module is not implemented yet.

Important current constraints:

- `OSTrackActor.forward_pass()` supports only one search frame.
- `cfg.DATA.SEARCH.NUMBER` defaults to `1`.
- `VisionTransformerCE.forward_features()` concatenates only template tokens and search tokens.
- There is no `tracking/run_module_experiment.py` file in the current checkout, although `EXPERIMENT_RUNBOOK.md` mentions it.

Because of that, the first local validation for this turn is a readiness check, not a 2-step training run.

## Implementation Milestones

### M0: Baseline Readiness

Validate that the current local checkout still works under:

```text
D:\Anaconda\envs\track\python.exe
```

Checks:

```powershell
D:\Anaconda\envs\track\python.exe -m compileall -q lib tracking
D:\Anaconda\envs\track\python.exe tracking\train.py --help
D:\Anaconda\envs\track\python.exe tracking\test.py --help
D:\Anaconda\envs\track\python.exe tracking\profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300
```

### M1: H0-MF Matched Future Control

Purpose:

```text
Use the same z0, s0, s1 sample as H2, but do not pass history.
```

Required changes:

- set `DATA.SEARCH.NUMBER = 2`
- extend actor to consume two searches
- run the same network twice with `H=None`
- compute `mean(loss_s0, loss_s1)`

Output metrics:

```text
Loss_s0, IoU_s0
Loss_s1, IoU_s1
Loss_mean, IoU_mean
GPU memory
iteration time
```

### M2: H1-Query No-Update Control

Purpose:

```text
Insert learned H0 but do not update it.
```

This checks whether the extra token interface itself damages baseline tracking.

Promotion rule:

```text
H1-Query should be close to H0-MF.
If it collapses, debug token insertion before adding Writer.
```

### M3: H2-2step Detach

Purpose:

```text
s0 writes H1
H1.detach() assists s1
```

This is the first real module candidate.

Output metrics:

```text
IoU_s0, IoU_s1
Delta_s1 vs H0-MF s1
history_norm
writer_delta_norm = ||H1 - H0||
peak location statistics
finite output check
```

Promotion rule:

```text
H2 detach must not be worse than H0-MF in local fast screening.
Any positive gain is only a screening signal, not a paper result.
```

### M4: H2-2step No-Detach

Run only after M3 works.

Purpose:

```text
Let s1 loss train the s0 history writer.
```

Decision:

```text
Keep no-detach only if the gain is clear enough to justify extra memory and instability.
```

## Local Fast Validation Rules

Follow the project runbook where possible, but adapt to the current checkout:

- Use `D:\Anaconda\envs\track\python.exe`.
- Do not recursively scan `data/` or `output/`.
- Do not launch long training until baseline readiness is confirmed.
- Use available scripts because `tracking/run_module_experiment.py` is absent.
- Keep data, pretrained weights, and outputs out of Git.

## Reviewer-Level Evidence Rules

A local quick win is not enough. Before this can become a paper module:

- compare against same-code same-sampling H0-MF
- report exact deltas, not only best checkpoints
- compare detach vs no-detach
- include failure cases
- evaluate VisDrone/UAVDT/DTB70 first
- advance to UAV123 and LaSOT only if UAV gates pass
- report compute, parameters, FPS, and memory

## Stop Rules

Stop or revise if:

- H1-Query breaks baseline outputs
- H2-real is worse than matched H0-MF
- no-detach only improves training loss but not validation/tracking metrics
- the module needs candidate rescue or rollback to avoid collapse
- improvements disappear under same-code replayed baselines

## Local Validation Record

| Date | Scope | Command | Result | Notes |
|---|---|---|---|---|
| 2026-06-29 | compile | `D:\Anaconda\envs\track\python.exe -m compileall -q lib tracking` | pass | Syntax/import-bytecode check passed |
| 2026-06-29 | train entry | `D:\Anaconda\envs\track\python.exe tracking\train.py --help` | pass | Existing train CLI is available |
| 2026-06-29 | test entry | `D:\Anaconda\envs\track\python.exe tracking\test.py --help` | pass | Existing test CLI uses `--dataset_name` |
| 2026-06-29 | runbook module entry | `Test-Path tracking\run_module_experiment.py` | missing | Runbook mentions this file, but current checkout does not contain it |
| 2026-06-29 | config load | load all `experiments/ostrack/*.yaml` | pass | All configs load; current `DATA.SEARCH.NUMBER` is still `1` |
| 2026-06-29 | model dummy forward | build `vitb_256_mae_ce_32x4_ep300` and forward random tensors | pass | `pred_boxes=(1,1,4)`, `score_map=(1,1,16,16)`, all finite |
| 2026-06-29 | profile | `tracking\profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300` | pass | `21.517G` MACs, `92.121M` params, `89.46 FPS` in this run |

## Immediate Next Coding Task

Implement M1 only:

```text
H0-MF matched future-control with DATA.SEARCH.NUMBER = 2
```

Do not add the history writer until the actor can run the same baseline network over `s0` and `s1` and report separated losses. This keeps the control honest before testing H1-Query or H2-2step.
