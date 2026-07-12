# B2: Frozen multi-scale Top-K candidate association

## Scope and fixed boundary

- Frozen tracker checkpoint: `pretrained_models/OSTrack_ep0300.pth.tar`.
- Frozen visual decoder: B1 `topk_no_kf`; no Kalman filter, gate, or OSTrack
  backbone/head update is used by B2.
- Train source: `data/got10k/train` only.  The local `data/lasot/train`
  directory was empty, so it was not used.  VisDrone, UAVDT, DTB70, UAV123,
  and LaSOT test sequences remain entirely held out.
- Candidate descriptor: final search token, block-2 pre-CE search token,
  Hann-weighted response score, and crop-normalized `cx, cy, w, h`.
- The 768-D visual tokens are compressed by fixed seeded projections; only the
  shallow/deep encoders, geometry encoder, matching head, and dustbin head are
  trainable.

## Training-only B1 trajectory collection

120 evenly distributed GOT-10k training sequences were followed by frozen B1
in free-running mode for at most 300 frames each.  Ground truth was used only
to label candidates after B1 produced its search crop and Top-K candidates.

| Item | Result |
| --- | ---: |
| Train association pairs | 12,205 |
| Validation association pairs | 3,518 |
| Top-K IoU >= 0.5 recall | 96.39% |
| Positive target pairs | 15,588 |
| Dustbin pairs | 135 |
| B1 top-response validation accuracy | 84.08% |
| B2 validation association accuracy | 86.41% |

The B2 checkpoint is `output/b2_candidate_assoc/b2_assoc.pt`.  Best validation
accuracy occurred before late-epoch overfitting; the saved checkpoint is not
the final epoch.

## Sequential test results

All result and timing artifacts were generated anew under
`output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_fulltn_b2_assoc`.
Each evaluated sequence has a matching prediction row count.

| Dataset | Sequences | B0 AUC / P / NP | B1 AUC / P / NP | B2 AUC / P / NP | B2 - B1 |
| --- | ---: | --- | --- | --- | --- |
| VisDrone | 86 / 86 | 65.25 / 85.17 / 83.78 | 65.63 / 85.71 / 84.14 | 65.72 / 85.92 / 84.00 | +0.09 / +0.21 / -0.14 |
| UAVDT | 50 / 50 | 62.31 / 84.69 / 72.82 | 62.81 / 84.76 / 73.40 | 64.55 / 87.40 / 75.25 | +1.74 / +2.64 / +1.85 |
| DTB70 | 70 / 70 | 66.56 / 86.92 / 81.74 | 66.78 / 87.22 / 82.07 | 65.61 / 85.52 / 80.30 | -1.17 / -1.70 / -1.77 |
| UAV123 | 123 / 123 | 68.27 / 88.80 / 83.81 | 68.49 / 89.02 / 83.98 | 68.22 / 88.80 / 83.64 | -0.27 / -0.22 / -0.34 |

## Decision

B2 is **not promoted**.  It gives a substantial UAVDT gain, a small VisDrone
gain, and is nearly neutral on UAV123, but regresses strongly on DTB70 and
VisDrone fails the predeclared +0.5 AUC gate over B0 (+0.47 only).  DTB70
remains in the project evaluation matrix and has not been removed; this is a
promotion decision for B2, not a dataset exclusion decision.  LaSOT was not
run for B2 because the UAV gate had already failed.

The failure is not candidate absence: the training-only B1 Top-K recall is
96.39%.  The likely first-order limitation is train/inference mismatch: B2 is
trained from ground-truth target candidates in the previous frame but must use
its own persistent selected-candidate memory at test time.  DTB70 exposes this
memory drift.  A next version should train the association memory with its own
predicted previous candidate (or short candidate tracklets) before reintroducing
any Kalman constraint.
