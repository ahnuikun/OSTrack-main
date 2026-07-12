# SAMURAI-inspired OSTrack: UAV123 ablations and two optimization rounds

## Fixed protocol

- Date: 2026-07-11
- Python: `D:\Anaconda\envs\track\python.exe`
- Checkpoint: `pretrained_models\OSTrack_ep0300.pth.tar`
- Test geometry: template 128, search 256, search factor 4.0
- Candidate settings: Top-K 5, NMS kernel 3, nominal motion weight 0.35
- All scores were recomputed with `tracking/analyze_uav_suite.py --force_evaluation`.

Each UAV123 result below was computed over all 123 sequences.  The two files
per sequence in the output directory are the prediction and timing artifacts;
the evaluator confirmed 123 / 123 prediction sequences for every run.

## UAV123 causal ablation

| Method | AUC | Precision | Norm Precision | Delta AUC vs full |
| --- | ---: | ---: | ---: | ---: |
| Matched retrained OSTrack baseline | 68.27 | 88.80 | 83.81 | +0.71 |
| Full fixed-motion SAMURAI-inspired path | 67.56 | 87.79 | 82.79 | 0.00 |
| Top-K, no KF | **68.49** | **89.02** | **83.98** | **+0.93** |
| Top-K + KF reranking, no update gate | 68.01 | 88.33 | 83.34 | +0.45 |

Interpretation:

1. Replacing OSTrack's response-box averaging with the strongest NMS-separated
   response candidate is beneficial on UAV123 (+0.22 AUC over the matched
   baseline).
2. Removing the original `MIN_SCORE` / `MIN_IOU` update gate helps relative to
   the full fixed-motion path (+0.45 AUC), so the imported thresholds are not
   well calibrated for Hann-weighted OSTrack responses.
3. KF reranking remains harmful relative to Top-K without KF (-0.48 AUC).
   Thus the main UAV123 failure is the fixed motion bias, not the existence of
   multiple response candidates.

## Optimization round 1: response-peak decoder

### Design

Use NMS-separated response peaks and output the regression box attached to the
maximum response peak (`MODE: topk_no_kf`).  This is a MAP-style visual choice:
it avoids averaging boxes from spatially distinct response modes, whose mean
can lie between the target and a distractor.  It does not use a motion model.

### Validation

| Dataset | Sequences | AUC | Precision | Norm Precision | Matched baseline AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| UAV123 | 123 | **68.49** | **89.02** | **83.98** | 68.27 |
| VisDrone | 86 | **65.63** | **85.71** | **84.14** | 65.25 |

The improvement transfers to a second UAV benchmark: +0.38 AUC over the
matched VisDrone baseline.  This is the optimization retained by this work.
It is deliberately named a response-peak decoding change, not a SAMURAI
reproduction or a Kalman contribution.

### Full five-dataset transfer validation

The retained `topk_no_kf` decoder was subsequently evaluated on the complete
UAVDT, DTB70, and LaSOT datasets.  The existing VisDrone and UAV123 artifacts
were force-evaluated again.  All comparisons below use the user's matched
300-epoch retrained checkpoint, not the author-provided checkpoint.

| Dataset | Sequences | B1 AUC / P / NP | Matched B0 AUC / P / NP | Delta B1 - B0 |
| --- | ---: | --- | --- | --- |
| VisDrone | 86 / 86 | 65.63 / 85.71 / 84.14 | 65.25 / 85.17 / 83.78 | +0.38 / +0.54 / +0.36 |
| UAVDT | 50 / 50 | 62.81 / 84.76 / 73.40 | 62.31 / 84.69 / 72.82 | +0.50 / +0.07 / +0.58 |
| DTB70 | 70 / 70 | 66.78 / 87.22 / 82.07 | 66.56 / 86.92 / 81.74 | +0.22 / +0.30 / +0.33 |
| UAV123 | 123 / 123 | 68.49 / 89.02 / 83.98 | 68.27 / 88.80 / 83.81 | +0.22 / +0.22 / +0.17 |
| LaSOT | 280 / 280 | 68.74 / 74.61 / 78.43 | 68.88 / 74.81 / 78.54 | -0.14 / -0.20 / -0.11 |
| **Five-set mean** | - | **66.49 / 84.26 / 80.40** | **66.25 / 84.08 / 80.14** | **+0.24 / +0.19 / +0.27** |

The four-UAV-set mean improves by +0.33 AUC, +0.28 Precision, and +0.36
Normalized Precision.  Therefore B1 transfers positively to all four UAV
benchmarks, unlike the fixed-KF path which regressed on UAV123.  Its LaSOT
result is slightly negative, confirming that response-peak decoding is a
short-term localization improvement rather than a solution for long-term
appearance change.

Artifact validation was performed before metric reporting.  The evaluator
found all expected prediction and timing files, and the 280 LaSOT prediction
files had zero frame-count mismatches.  Newly executed wall times were 10m27s
for UAVDT, 5m30s for DTB70, and 3h01m28s for LaSOT on one GPU.

## Optimization round 2: entropy-adaptive motion and NIS update gate

### Design and theory

`MODE: adaptive_kf` scales the nominal motion weight by the normalized entropy
of the five NMS peak scores.  Motion consequently has little influence for a
decisive visual peak and more influence when candidates are visually ambiguous.
For the KF measurement update, it replaces raw response/IoU thresholds with
the normalized innovation squared (NIS).  The update is accepted when
`NIS <= 9.488`, the 95% chi-square quantile for the 4-D `(cx, cy, w, h)`
measurement model.

### Result and decision

| Dataset | AUC | Precision | Norm Precision | Delta vs round 1 AUC |
| --- | ---: | ---: | ---: | ---: |
| UAV123 | 67.99 | 88.41 | 83.36 | -0.50 |

This round did not meet its predeclared promotion criterion of exceeding the
round-1 UAV123 AUC (68.49).  It was not expanded to VisDrone, and no further
threshold scan or optimization round was run.  The result shows that, for this
constant-velocity model and response representation, uncertainty-aware timing
and probabilistic update gating alone do not repair the motion-model mismatch.

## Reproduction commands

```powershell
D:\Anaconda\envs\track\python.exe tracking\test.py ostrack vitb_256_mae_ce_32x4_ep300_fulltn_samurai_topk_no_kf --dataset_name uav123 --num_gpus 1 --checkpoint D:\PyCharm\Projects\OSTrack-main\pretrained_models\OSTrack_ep0300.pth.tar
D:\Anaconda\envs\track\python.exe tracking\test.py ostrack vitb_256_mae_ce_32x4_ep300_fulltn_samurai_topk_no_kf --dataset_name visdrone --num_gpus 1 --checkpoint D:\PyCharm\Projects\OSTrack-main\pretrained_models\OSTrack_ep0300.pth.tar
D:\Anaconda\envs\track\python.exe tracking\test.py ostrack vitb_256_mae_ce_32x4_ep300_fulltn_samurai_kf_no_gate --dataset_name uav123 --num_gpus 1 --checkpoint D:\PyCharm\Projects\OSTrack-main\pretrained_models\OSTrack_ep0300.pth.tar
D:\Anaconda\envs\track\python.exe tracking\test.py ostrack vitb_256_mae_ce_32x4_ep300_fulltn_samurai_adaptive_kf --dataset_name uav123 --num_gpus 1 --checkpoint D:\PyCharm\Projects\OSTrack-main\pretrained_models\OSTrack_ep0300.pth.tar
```
