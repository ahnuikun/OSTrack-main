# Experiment Report: `S0-E0-official-dtb70_Animal1-20260715-r01`

## 1. 结论

- 状态：`completed`
- Gate：`Not evaluated`
- 一句话结论：注册的 official checkpoint 能以 `strict=True` 完成加载，E0 在 DTB70 `Animal1` 的 147 帧上端到端运行、落盘并通过正式评测器重算，基线路径通畅。

## 2. 实验身份

| 字段 | 值 |
|---|---|
| Variant | E0 |
| Association backend | A0 appearance Top-1 |
| Git commit / dirty | `58f2ecbdcb5ab40bc121d50a09d8c95bde71a664` / true |
| Tracked diff hash | `9d6f091f0eb4b367b1f2a7bf666f76bb1e678be8` |
| Checkpoint ID | `official_vitb256_ce_ep300` |
| Checkpoint SHA256 | `8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41` |
| Dataset / sequences | `dtb70_local_v1` / `Animal1` only |
| Start / finish | 2026-07-15 14:08:24 / 14:09:45 +08:00 |

## 3. Smoke 结果

| Scope | AUC | Precision | Norm Precision | FPS |
|---|---:|---:|---:|---:|
| Animal1 only | 53.16 | 68.71 | 65.31 | 75.11 |

这些是 `--skip_missing_seq` 下 1/70 序列的通路验证数字，不是 DTB70 正式基线，不进入论文主表。

## 4. 完整性检查

- 结果文件：1 个，名称与序列一致。
- 结果行数：147，与输入帧数一致。
- 每行字段数：4。
- 所有数值有限，宽高均为正。
- 测试进程与 `--force_evaluation` 评测进程退出码均为 0。

## 5. Gate 判断

本实验只证明当前 E0 端到端路径可运行，不判断 E1-E6 的研究 gate。后续仍需在模块实现后完成 output-parity、单元测试和冻结 development 序列实验。
