# Experiment Report: `S0-E0shadow-official-dtb70_Animal1-20260715-r01`

## 结论

- 状态：`completed`
- Gate：`Go`（仅指闭环接入完整性）
- 模块集成路径在 CMC/KF/关联全部关闭时，与官方 E0 的 147 帧结果逐元素完全一致。

| 检查项 | 结果 |
|---|---:|
| Official / shadow shape | 147×4 / 147×4 |
| Max absolute box difference | 0.0 |
| Exact array equality | true |
| Diagnostic records | 147 |
| Blank JSONL records | 0 |
| Frame IDs contiguous | true |
| Selected rank > 1 | 0 |
| FPS | 63.74 |

该实验通过了“诊断和候选提取不能改变 E0 输出”的基础 gate。FPS 只用于 smoke，正式效率结论需要重复运行并分离 checkpoint hash、模型加载和逐帧子模块耗时。
