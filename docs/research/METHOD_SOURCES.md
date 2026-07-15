# 已发表方法、代码来源与本地适配登记

已有工作是技术起点，不是设计上限。实现时既不能把多个论文模块生搬硬套，也不能把已有代码换名后称为创新。本表同时登记上游来源和为兼容本框架所做的本地变通。

| 来源 | 论文 | 核对的官方代码版本 | 可借鉴内容 | 使用边界 |
|---|---|---|---|---|
| OSTrack | [ECCV 2022](https://arxiv.org/abs/2203.11991) | [botaoye/OSTrack](https://github.com/botaoye/OSTrack) `33b5e125`，MIT | 单次前向、Hann 后中心响应、size/offset 解码 | 当前网络基线；本项目不改权重和训练路径 |
| UOSTrack / MBPP | [arXiv:2301.01482v5](https://arxiv.org/abs/2301.01482) | [LiYunfengLYF/UOSTrack](https://github.com/LiYunfengLYF/UOSTrack) `ff90817a`，MIT | raw response Top-30、box NMS、`response × KF-IoU`，Top-1 与 KF IoU > 0.6 时保持 Top-1 | A1 论文基线；原实现无 CMC、无条件用选中框更新 KF，不直接作为 full 版本 |
| SAMURAI | [arXiv:2411.11922](https://arxiv.org/abs/2411.11922) | [yangchris11/samurai](https://github.com/yangchris11/samurai) `76ba1959`，Apache-2.0 | 稳定帧后以固定权重融合外观 IoU 与 KF-IoU，并用外观/对象/KF 分数筛选记忆 | A2 评分基线；SAM2 mask 候选、学习式 IoU 和 memory bank 与 OSTrack 不同，只借鉴可对应的评分/稳定门控思想 |
| NeighborTrack | [arXiv:2211.06663](https://arxiv.org/abs/2211.06663) | [franktpmvu/NeighborTrack](https://github.com/franktpmvu/NeighborTrack) `29c6267c`，根目录未提供许可证 | 候选/干扰物短轨迹、反向验证、二分图匹配、KF 候选 | A4 论文级对照候选；未解决许可证前不复制代码，且不进入第一轮核心实现 |
| MATA | [arXiv:2603.03904v2](https://arxiv.org/abs/2603.03904) | 未发现论文关联的公开官方仓库 | 稀疏光流 ego-motion、相机/目标动态分离、EKF、低可信观测处理 | 只借鉴模块分工；其异步调度与本项目逐帧单前向同步约束不同 |

## 适配登记模板

```text
adaptation_id: N#
source_name:
paper_version:
repository_url:
upstream_commit:
upstream_file_and_symbol:
license:
usage: copied | adapted | reimplemented_from_paper
source_limitation:
local_adaptation:
expected_mechanism:
local_file_and_symbol:
independent_toggle:
diagnostic_fields:
closest_published_baseline: A#
related_ablation: E#/N#
verified_on:
```

## 当前优先研究的适配点

这些是候选研究问题，不代表未经实验就成立：

1. MATA 类 ego-motion 与标准 box KF 的接口需要适配到 OSTrack 每帧同步、单次前向流程。
2. UOSTrack/SAMURAI 的固定 KF-IoU 加权需要适配 OSTrack 响应峰分布和 CMC 可靠性。
3. KF 状态、速度和协方差需要在单应传播后保持一致，而不是只移动预测框中心。
4. 候选输出与 KF 观测写入需要解耦，避免错误重排立即污染运动状态。
5. NeighborTrack 类短轨迹如果使用，应在相机补偿后的坐标中重新定义距离与匹配代价。

## 已登记本地适配

```text
adaptation_id: N1
source_name: UOSTrack MBPP + SAMURAI motion-aware scoring + MATA motion separation
paper_version: arXiv:2301.01482v5; arXiv:2411.11922; arXiv:2603.03904v2
repository_url: https://github.com/LiYunfengLYF/UOSTrack; https://github.com/yangchris11/samurai
upstream_commit: ff90817a544d8feb739880cdc034586c953a0f18; 76ba195984892b0d1e3db5d9c9f90bb62175680a
upstream_file_and_symbol: lib/test/tracker/ostrack.py::OSTrack.track; sam2/sam2/modeling/sam2_base.py::_forward_sam_heads
license: MIT; Apache-2.0
usage: independently reimplemented and adapted
source_limitation: fixed motion weights do not account for CMC estimation quality, KF uncertainty, or whether OSTrack Top-1 is visually unambiguous; candidate selection and state update are coupled in MBPP
local_adaptation: use the same fixed-IoU equation in E2/E3, then only scale its motion weight by q_cmc, q_kf, and q_ambiguity in E4/E5/E6; separate selected output from conditional KF update
expected_mechanism: retain fixed-weight behavior when all qualities are one, reduce motion override when any prerequisite is unreliable, and prevent a low-confidence selected box from contaminating KF state
local_file_and_symbol: lib/test/tracker/cmc_kf_assoc/association.py::associate_candidates; lib/test/tracker/ostrack_cmc_kf_assoc.py::OSTrackCMCKFAssoc.track
independent_toggle: cmc_quality_gate; kf_quality_gate; appearance_ambiguity_gate; observation_rejection
diagnostic_fields: q_cmc; q_kf; q_ambiguity; motion_weight; selected_rank; measurement_accepted; rejection_reason
closest_published_baseline: A2
related_ablation: E3-E6
verified_on: S0 output parity plus frozen development_v1 (20 sequences, 12,311 frames) as of 2026-07-15; N1/E6 reached +3.70 AUC points over E0 but failed association-effect and tail-safety gates, so it is not promoted
```

### N2 候选（尚未实现、尚未获准评测）

S1 结果显示 N1 的单帧加权没有约束“微小分数优势造成的闭环轨迹分叉”。下一适配候选借鉴 SAMURAI 的稳定期思想与 NeighborTrack 的时序验证思想，但不复制其代码：仅当 rank 改选具有足够决策间隔并通过短时相机补偿坐标一致性时才提交，否则保持 appearance Top-1。N2 是看到 `development_v1` 结果后提出的，因此不得在原 split 上调参后把结果称为独立验证；必须预注册阈值、冻结新的开发划分，并与 A2/N1/M1 直接比较。

每一项只有在对应 N* 消融优于最接近的 A* 基线、机制诊断一致且长尾安全时，才可以进入论文贡献候选。

## 代码合规规则

- 未核对许可证前，不直接复制外部仓库的大段代码。
- 根据论文公式重写也要记录论文版本和本地差异。
- 外部默认阈值只可作为预注册起点，不能冒充本数据集验证得到的最优参数。
- 找到新的相关方法或官方仓库时先更新本表，再开始移植。
