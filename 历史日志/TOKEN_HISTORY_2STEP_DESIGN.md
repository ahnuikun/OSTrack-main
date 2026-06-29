# Token History 2-Step Design for OSTrack

Updated: 2026-06-29

## Goal

Implement the first usable version of a token-state temporal module for OSTrack using a fixed 2-step temporal unroll.

The module should not use recursive candidate rescue, pending templates, rollback rules, or RGB template recropping. It should keep the initial template `z0` fixed, maintain a lightweight feature state `H_t`, and train the model to read and update that state over two search frames.

## Core Idea

The state is a Transformer token state:

```text
H_t: [B, Lh, C]
```

For the first experiment:

```text
Lh = 1
C = 768 for ViT-B
H_t = [B, 1, 768]
```

This is deliberately close to an ODToken-style `track_query`, but the design adds an explicit Writer/Updater so the contribution is not just "insert one learned token". The intended difference is:

```text
ODToken-like state:
    previous query token is reused as the next frame state

This module:
    encoded search observation O_t is pooled from the response peak region
    Writer(H_t, O_t) produces H_{t+1}
```

## 2-Step Train-Time Unroll

The training sample should contain one initial template and two search frames:

```text
input:
    z0, s0, s1

step 0:
    out0, H1 = Net(z0, H0, s0)

step 1:
    out1, H2 = Net(z0, H1, s1)

loss:
    loss = mean(loss_s0, loss_s1)
```

This is a truncated temporal unroll, not an infinite recurrence. It trains the network to:

```text
read H0 while tracking s0
extract an observation O0 from s0
write H1
read H1 while tracking s1
```

## H0 Policy

Use a learned initial token:

```text
H0 = learned parameter [1, Lh, C], expanded to [B, Lh, C]
```

Do not initialize `H0` from GT crops. GT-derived history can only be used later as an oracle upper-bound diagnostic, not as the real method.

## Observation O_t

The observation should be feature-space only:

```text
score_map_t -> response peak
encoded search tokens around peak -> pooled observation O_t
O_t: [B, Lh, C]
```

Do not crop a new RGB template from the predicted box. That path has the classic failure chain:

```text
wrong box -> wrong crop -> polluted template -> worse next frame
```

The first version can use the peak cell from the `16x16` search token grid for `256x256` search with stride `16`, then optionally pool a `3x3` neighborhood.

## Transformer Insertion

The intended token order is:

```text
[template tokens z0, history tokens H_t, search tokens s_t]
```

The head must still decode only search tokens. That means the model must track token lengths explicitly:

```text
lens_z = template token count
lens_h = history token count
lens_x = search token count
search_tokens = encoded[:, lens_z + lens_h:]
```

This is safer than treating history as another template image, because `H_t` is not an image and should not share the full template crop path.

## Code Integration Points

Current repo facts:

- `lib/train/actors/ostrack.py` currently asserts `len(data['template_images']) == 1` and `len(data['search_images']) == 1`.
- `lib/train/data/sampler.py` already has `num_search_frames`, but the current actor only consumes one search frame.
- `lib/models/ostrack/ostrack.py` forwards `template` and `search` once, then decodes from the last search tokens.
- `lib/models/ostrack/vit_ce.py` currently concatenates `z + x` with `combine_tokens(z, x, mode=self.cat_mode)`.
- `lib/config/ostrack/config.py` has `DATA.SEARCH.NUMBER = 1` by default.

Therefore the minimum implementation sequence is:

1. Add `cfg.MODEL.TOKEN_HISTORY` defaults.
2. Add a small `TokenHistoryWriter` module.
3. Extend the OSTrack forward path to accept `history_tokens=None`.
4. Extend the ViT CE forward path to concatenate `z + history + x`.
5. Return both search features for decoding and enough encoded search tokens to build `O_t`.
6. Add a 2-step actor path that consumes `search_images[0]` and `search_images[1]`.
7. Set a new experiment YAML with `DATA.SEARCH.NUMBER = 2`.

## Detach Policy

Run detach first:

```text
H1_for_s1 = H1.detach()
```

Reason:

- stable on local quick experiments
- lower memory
- closer to test-time state passing

Then compare no-detach on the server:

```text
H1_for_s1 = H1
```

No-detach is only worth keeping if `s1` loss clearly improves how `s0` writes future-useful history. Otherwise it adds memory and instability without a strong story.

## Minimal Experiment Ladder

Only implement the first three rungs now:

```text
H0-MF:
    same z0, s0, s1 sample
    no history
    loss = mean(loss_s0, loss_s1)

H1-Query:
    learned H0 inserted
    no update
    verifies that adding a token does not break the baseline

H2-2step:
    s0 writes H1
    H1 assists s1
    detach first
```

Do not implement reliability, 3-step, rollback, candidate rescue, or multi-branch pending logic until H2 has a clean matched-control result.

## Success Criteria

Local quick validation is only a mechanism check. It is not paper evidence.

The 2-step design can advance to code implementation if:

- baseline config still loads
- baseline model still builds and forwards
- a future 2-step actor can run with `DATA.SEARCH.NUMBER = 2`
- H1-Query does not destabilize outputs
- H2-2step does not underperform the matched H0-MF control in local fast screening

The design can advance to server experiments only if:

- matched same-sampling H0-MF is recorded
- detach/no-detach comparison is planned
- no GT crop is used in the real method
- all deltas are reported against replayed same-code baselines

## Explicit Non-Goals

- No RGB dynamic template replacement.
- No recursive active/pending/candidate selection.
- No GT crop as a real inference input.
- No reliability gate before H2-2step is validated.
- No claim of novelty from "adding history" alone.
