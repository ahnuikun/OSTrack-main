# SAMURAI Test-Time Evaluation Results

## Fixed evaluation setup

- Date: 2026-07-10
- Repository: `D:\PyCharm\Projects\OSTrack-main`
- Python: `D:\Anaconda\envs\track\python.exe`
- Checkpoint: `pretrained_models\OSTrack_ep0300.pth.tar`
- Configuration: `vitb_256_mae_ce_32x4_ep300_fulltn_samurai`
- Protocol: run one dataset to completion, calculate its metrics, record the
  analysis, and only then start the next dataset.
- Dataset order: VisDrone → UAVDT → DTB70 → UAV123 → LaSOT.

## Result summary

| Dataset | Sequences | AUC | Precision | Norm Precision | Wall time | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| VisDrone | 86 / 86 | 66.21 | 86.47 | 84.50 | 25m 34s | Complete |
| UAVDT | 50 / 50 | 63.86 | 86.22 | 74.57 | 10m 28s | Complete |
| DTB70 | 70 / 70 | 66.88 | 87.50 | 82.21 | 5m 39s | Complete |
| UAV123 | 123 / 123 | 67.56 | 87.79 | 82.79 | 31m 22s | Complete |
| LaSOT | — | — | — | — | — | Pending |

## Per-dataset analysis

### VisDrone

- Results were calculated over all 86 sequences.
- The test-time candidate selection, Kalman reranking, and reliability gate
  completed without Python errors after the final 256x256 test geometry was
  restored.
- This is an absolute result only.  A matched original-OSTrack VisDrone run
  with the same checkpoint has not yet been performed, so no causal gain or
  loss from the SAMURAI-style module is claimed.

### UAVDT

- Results were calculated over all 50 sequences: AUC 63.86, Precision 86.22,
  and Norm Precision 74.57.
- The module remained numerically stable throughout the complete UAVDT run;
  no checkpoint, response-map, or Kalman-update exception occurred.
- The lower normalized precision relative to VisDrone is an absolute
  cross-dataset observation, not evidence of degradation: the two datasets
  differ in target scale, motion, and annotation distributions. A matched
  no-module baseline is required for a causal claim.

### DTB70

- Results were calculated over all 70 sequences: AUC 66.88, Precision 87.50,
  and Norm Precision 82.21.
- The full run completed without runtime errors.  This confirms the test-side
  motion module is compatible with DTB70's sequence layouts and annotations.
- These remain absolute scores; cross-dataset ranking alone must not be used
  to infer the SAMURAI-style module's causal contribution.

### UAV123

- Results were calculated over all 123 sequences: AUC 67.56, Precision 87.79,
  and Norm Precision 82.79.
- The complete UAV123 run was stable and produced every expected result file.
- A fair conclusion about this test-time module still requires comparison with
  the same 300-epoch checkpoint run using the unchanged OSTrack decoder.

## Execution log

| Time | Event | Outcome |
| --- | --- | --- |
| 2026-07-10 | Initial VisDrone launch | Stopped: YAML inherited global 320x320 test defaults, incompatible with the 256x256 checkpoint response shape. No valid result used. |
| 2026-07-10 | Corrected configuration + one-sequence smoke test | Passed; real-model inference completed at about 71.8 FPS. |
| 2026-07-10 | Full VisDrone evaluation | Completed 86/86 sequences; result aggregation successful. |
| 2026-07-10 | Full UAVDT evaluation | Completed 50/50 sequences in 10m 28s; result aggregation successful. |
| 2026-07-10 | Full DTB70 evaluation | Completed 70/70 sequences in 5m 39s; result aggregation successful. |
| 2026-07-10 | Full UAV123 evaluation | Completed 123/123 sequences in 31m 22s; result aggregation successful. |
