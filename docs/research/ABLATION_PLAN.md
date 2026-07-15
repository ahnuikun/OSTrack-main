# 论文导向实验与消融计划

## 1. 固定对照原则

所有同表实验必须保持：checkpoint、YAML、搜索几何、数据集版本、序列集合、硬件、随机种子和正式评测器一致。推理模块不重新训练，因此主要差异只来自测试时模块。

E0/E1 必须与新模块在同一新项目重新验证一次；旧项目结果只作为历史对照，不能替代新项目基线完整性检查。

## 2. 主消融链 E0-E6

| 编号 | CMC | KF | Top-K 关联 | 可靠性/更新控制 | 目的 |
|---|---:|---:|---:|---|---|
| E0 | 否 | 否 | 否 | 无 | 官方 OSTrack 基线 |
| E1 | 是 | 否 | 否 | 仅 CMC fallback | 测量 CMC crop 贡献 |
| E2 | 否 | 是 | 是 | A2 固定 IoU 权重 | 测量未补偿 KF 预测裁剪与候选重排 |
| E3 | 是 | 是 | 是 | 与 E2 完全相同的 A2 固定 IoU 权重 | 只增加 camera-propagated KF |
| E4 | 是 | 是 | 是 | E3 + CMC 质量门控 | 验证 CMC 正确性是否减少误导 |
| E5 | 是 | 是 | 是 | E4 + KF 稳定性/新息门控 | 验证 KF 正确性控制 |
| E6 | 是 | 是 | 是 | E5 + 外观歧义门控 + 拒绝低可信更新 | 完整版本 |

E2 只是消融对照，不改变“完整模块必须先 CMC 再 KF”的主设计。

## 3. 关联后端与适配创新消融

在 E6 其他条件固定后比较：

| 编号 | 关联方式 | 对应借鉴 |
|---|---|---|
| A0 | appearance Top-1 | OSTrack |
| A1 | raw response × KF-IoU + Top-1 keep gate | UOSTrack MBPP |
| A2 | response + 固定权重 KF-IoU | SAMURAI-style 评分 |
| A3 | response + KF Mahalanobis likelihood | 标准 KF 数据关联 |
| A4 | short tracklet + bipartite matching | NeighborTrack |

A0-A3 是已有论文技术的参考基线，不是设计上限。合理兼容本框架的新方法使用 `N1`、`N2`……编号，并紧跟其最接近的 A* 基线比较。

每个 N* 必须预注册：

| 字段 | 内容 |
|---|---|
| Source limitation | 上游方法在哪个坐标、状态、候选或更新假设上不兼容 |
| Local adaptation | 本项目具体修改 |
| Expected mechanism | 为什么预期减少错误或提高恢复 |
| Closest baseline | A1/A2/A3 中直接对照项 |
| Toggle | 独立启用/关闭参数 |
| Diagnostics | 能直接验证机制的逐帧字段 |
| Paper claim candidate | 若有效，论文中准备如何准确表述 |

同一 N* 只允许引入一个主要变通。多个有效 N* 最后组合成 full 版本时，还需增加组合消融，证明收益不是偶然抵消。

## 4. 机制验证表

### 4.1 CMC

- 有效帧率、fallback 分布、内点率、重投影误差、空间覆盖率。
- camera-propagated prior 相对“不使用 CMC 的 KF prior”的中心误差与 IoU。
- CMC rescue/harm 与相机位移强度分桶。

### 4.2 KF

- 一步预测中心误差、IoU、Mahalanobis 分布。
- 正常帧、快速运动、遮挡前后、CMC 有效/无效分组。
- update、predict-only、rejected measurement 比例。

### 4.3 候选

- Top-1/Top-3/Top-5 oracle recall。
- Top-1 错误但 Top-K 存在正确候选的 recoverable rate。
- rescue rate、harm rate、selected-rank 分布和连续错误段长度。

### 4.4 必做归因 shadow

正式比较 E2/E3/E6 前，必须额外报告不进入主消融编号的机制 shadow：

- `M0`：官方裁剪 + Top-K 只记录不生效，验证候选提取 output parity；
- `M1`：KF/CMC 预测裁剪 + appearance Top-1，隔离 crop prior 收益；
- `M2`：与 M1 相同裁剪 + A2 关联，`M2-M1` 才是候选重排净贡献；
- `M3`：与 M2 相同输出选择 + 条件 KF update，隔离观测拒绝净贡献。

若 E2/E3 相对 E0 提升而 `M2-M1≈0`，结论必须写成“预测裁剪有效，候选关联尚未证实”，不能归功于重排。

## 5. 阶段与数据隔离

| 阶段 | 数据用途 | 可做的决策 |
|---|---|---|
| S0 | 单序列 smoke | 只修代码、路径和数值错误 |
| S1 | 冻结 development 序列 | 验证机制、选择 A1/A2、设定有限阈值 |
| S2 | 冻结 development + 未打开 holdout | 冻结 E6 参数和 promotion gate |
| S3 | holdout、四个 UAV 主数据集与 LaSOT | 只评测，不回调参数 |

序列清单必须在首次开发实验前写入 manifest 并冻结。不得根据结果把困难序列移出 development 或 holdout。

## 6. Promotion gate

S1 → S2：

- Top-5 oracle recall@0.5 至少比 Top-1 高 `2.0` 个百分点，且冻结 development 集至少有 `10` 个 recoverable 帧；
- camera-propagated KF 在 CMC 有效帧的一步中心误差中位数至少降低 `5%`，且 P95 不得恶化超过 `5%`；
- A1/A2/N1 的 rescue 数必须高于 harm，且相对同裁剪 M1 不得新增超过 `10` 帧的连续失败段。

S2 → S3：

- E6 在 development 上相对 E0 的 AUC 至少 `+0.30` 点，且不低于 E1；
- 主要提升不是来自单个序列；
- 负增益序列比例不超过 `45%`，最差单序列 AUC 回归不超过 `5.0` 点，P95/P99 单帧 IoU 回归不超过 `0.03/0.08`；
- 参数、代码、checkpoint 和记录全部冻结。

完整 benchmark 判断：

- DTB70、UAVDT、VisDrone、UAV123 四个 UAV 数据集均为主实验，必须全部完成并同等汇报；
- 四个 UAV 数据集的运行先后只按耗时安排，任何一个早期结果都不能决定是否跳过后续数据集；
- UAV 主结论同时报告四个数据集的逐数据集结果、平均趋势与长尾分布，不能只选择有利数据集；
- LaSOT 在四个 UAV 主实验之后运行，单独验证跨场景泛化与通用性；
- 若平均提升但长尾明显恶化，结论必须是 No-Go 或“仅条件有效”。

## 7. 论文表格预留

- Table 1：E0、E1、E3、E6 在四个 UAV 主数据集上的结果。
- Table 2：E0-E6 逐项消融。
- Table 3：A0-A3 论文基线与 N* 适配创新。
- Table 4：Top-K coverage、rescue/harm、KF 预测、CMC 质量。
- Table 5：FPS、各子模块耗时、显存。
- Table 6：LaSOT 泛化与通用性结果。
- Appendix：每序列 delta、最差案例、P95/P99、失败段。

任何实验若不能明确进入上述某张表或解释某个 gate，默认不运行。
