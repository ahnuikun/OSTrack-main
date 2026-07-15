# 模块设计：CMC-KF Candidate Association

本文档规定实现职责、数据接口和允许的论文方法变体。代码结构必须服从本文档，而不是按实验临时复制 tracker 文件。

## 1. 总体数据流

```text
I(t-1), I(t), previous box
        │
        ▼
CameraMotionEstimator ── H(t-1→t), q_cmc
        │
        ▼
CameraStatePropagator ── camera-propagated prior
        │
        ▼
KalmanBoxPredictor ───── predicted box, covariance, q_kf
        │
        ├───────────────┐
        │               │
        ▼               ▼
OSTrack one forward   ResponseCandidateExtractor
                        Top-K boxes + response scores
                                │
                                ▼
CandidateAssociator ── selected candidate + decision reason
                                │
                                ▼
ObservationController ─ output box; KF update or predict-only
```

裁剪中心必须显式记录。E0 使用上一输出框；E1 使用单应传播框；E2 使用未补偿 KF 预测框；E3-E6 使用“单应传播 KF 状态后再 predict”的预测框。候选框映射回图像时必须使用本帧实际裁剪参考框，不能继续引用旧 `self.state`。

## 2. 子模块职责

下述设计是第一版兼容框架，不是对论文代码的逐行拼接。任何上游方法若假设不同的候选形式、坐标系、更新频率或状态定义，必须先通过接口适配再进入本项目。

### 2.1 CameraMotionEstimator

输入：相邻完整 RGB 帧、上一帧目标框。
输出：单应矩阵 `H`、有效标志、内点率、重投影误差、空间覆盖率和耗时。

约束：

- 特征点优先来自目标框外背景；使用稀疏 Lucas-Kanade 光流和 RANSAC/USAC 单应估计。
- CMC 只描述相机导致的图像平面传播，不直接选择目标候选。
- `q_cmc` 必须由可解释诊断量计算；无效 CMC 时退化为 identity，并令运动关联权重减弱或归零。

### 2.2 CameraStatePropagator

输入：上一 KF 状态/协方差和 `H`。
输出：传播到当前相机坐标的状态先验。

中心通过单应矩阵投影；宽高第一版可由投影框四角或局部 Jacobian 传播。协方差加入 CMC 噪声项。这里不维护 EWMA 残余状态。

### 2.3 KalmanBoxPredictor

第一版采用标准常速度 KF，建议状态：

```text
x = [cx, cy, w, h, vx, vy, vw, vh]
z = [cx, cy, w, h]
```

每帧顺序固定为：相机传播 → KF predict → candidate association → 条件 update。输出预测框、创新协方差、Mahalanobis distance、状态年龄、连续 predict-only 帧数和 `q_kf`。

### 2.4 ResponseCandidateExtractor

输入：Hann 加权响应图、size map、offset map。
输出：空间 NMS 后的 Top-K 候选。

第一版固定 `K=5`，每个候选保存：rank、grid 坐标、响应分数、归一化框、图像坐标框。Top-1 解码必须与官方 OSTrack 输出数值对齐。

### 2.5 CandidateAssociator

用于复现与对照的关联后端：

- `A0 appearance_top1`：官方 OSTrack，仅作对照。
- `A1 mbpp_product`：UOSTrack MBPP 对照，raw response Top-30、box NMS、`response × KF-IoU`，并保留其 Top-1/KF IoU 0.6 判断。
- `A2 fixed_iou_rerank`：SAMURAI-style 固定权重加和；E2/E3 v2 固定使用同一公式。
- `A3 mahalanobis_likelihood`：标准 KF 创新协方差下的候选似然，用于比较 IoU 与协方差感知距离。
- `A4 neighbor_tracklet`：NeighborTrack 风格短轨迹与二分图匹配；第一轮不实现，需先确认 A1-A3 和 Top-K 上限。

这些后端不是封闭列表。新的适配方法使用 `N1`、`N2`……登记，例如相机补偿坐标中的短轨迹、CMC/KF/响应联合动态权重或延迟观测提交。新增后端必须满足：

1. 明确指出现有论文方法在本框架中的不兼容点；
2. 只解决一个可验证机制问题；
3. 保持单次 OSTrack 前向和在线因果性；
4. 有独立开关、逐帧诊断和对应消融；
5. 与 A1/A2/A3 中最接近的论文基线直接比较。

实现顺序不是强制照搬 A1→A2→A3；应先实现能回答当前机制问题的最小论文基线，再加入一个适配变化。禁止在同一版本首次混入多项变化，否则无法判断创新变通是否有效。

### 2.6 ObservationController

候选选择与 KF 写入分开：

- 输出：按关联器选择候选；若 motion gate 关闭则保持 appearance Top-1。
- 更新：只有响应、创新和状态稳定性通过门控时才 correct；否则 predict-only。
- 不允许用当前 GT 决定输出或更新。
- 连续拒绝后允许增大 KF 不确定性，但不得引入旧残余记忆。

## 3. 第一版评分

E2/E3 v2 固定 IoU 重排：

```text
lambda = 0.25
score_i = (1 - lambda) * appearance_i + lambda * IoU(box_i, box_kf)
```

可靠性重排：

```text
lambda_t = 0.25 * q_cmc * q_kf * q_ambiguity
score_i = (1 - lambda_t) * appearance_i + lambda_t * motion_i
```

E4 只启用 `q_cmc`，E5 再启用 `q_kf`，E6 再启用 `q_ambiguity` 与观测拒绝。任一质量开关关闭时其因子固定为 1。这样 E3→E6 每一步只有一个主要变化。`q_ambiguity` 仅在 Top-1/Top-2 外观接近时增大，防止运动模型推翻明显可靠的 Top-1。

该公式是起始基线，不是最终结构限制。若固定加权与 OSTrack 响应分布或 CMC 后 KF 不确定性不兼容，可以修改概率归一化、门控或关联形式；修改前须在实验 manifest 中登记 adaptation ID 和预期机制。

## 4. 代码目录契约

```text
lib/test/tracker/
├── ostrack_cmc_kf_assoc.py
└── cmc_kf_assoc/
    ├── __init__.py
    ├── camera_motion.py
    ├── state_propagation.py
    ├── kalman_box.py
    ├── response_candidates.py
    ├── association.py
    ├── observation.py
    └── diagnostics.py

lib/test/parameter/
├── ostrack.py
├── ostrack_checkpoint_registry.py
└── ostrack_cmc_kf_assoc.py

tests/cmc_kf_assoc/
├── test_camera_motion.py
├── test_state_propagation.py
├── test_kalman_box.py
├── test_response_candidates.py
└── test_association.py

tools/cmc_kf_assoc/
├── run_experiment.py
├── run_subset.py
├── analyze_experiment.py
└── analyze_subset.py
```

禁止为每个 beta、阈值或数据集复制一个 tracker Python 文件。参数差异必须进入 manifest 或显式配置对象；tracker 名称只表达算法结构，不表达一次运行的超参数。

当前实现以 `ostrack` 表示官方 E0，以 `ostrack_cmc_kf_assoc` 表示参数化的 E0-shadow/E1-E6。`v1` 保留最初 MBPP smoke，`v2` 保留“固定加权但误用 raw appearance”的失败证据，`v3` 保留绝对 response 拒绝导致 KF 长时间 predict-only 的失败证据；`v4` 首次引入“清晰 Top-1 或与 KF 一致才写入”的观测策略，但 M3 暴露出观测歧义与关联门控共用常量的隐藏耦合。正式 development 使用 `v5`：A1/MBPP 使用 raw response，A2/N1 对所有候选统一使用 Hann response；真实外观歧义始终供 ObservationController 使用，是否把它用于关联权重则由独立开关控制。

## 7. S1 后的设计约束

`development_v1` 的结果表明预测裁剪有效，但一次极小优势的候选改选也可能改变后续闭环轨迹。`uav0000222_00900_s` 第 280 帧中，A2/N1 以 `0.9286` 对 `0.9225` 的最终分数微差把 rank-1 改为 rank-2；当时两者都未命中 GT，但该改选改变了后续搜索轨迹，使 M1 在第 643 帧重新捕获目标而 E6 未能重捕。由此增加以下硬约束：

1. “当前帧 selected IoU 未下降”不能证明关联安全，必须检查闭环后续失败段和重捕能力。
2. 仅因 Top-1/Top-2 外观接近而增大运动权重不够；任何改选还必须具有明确的决策优势或时序确认。
3. N1/v5 保留为负结果，不在同一 `development_v1` 上继续调阈值后宣称独立验证。
4. 下一适配候选应单独登记为 N2：在不增加网络前向的前提下，引入改选滞回/延迟提交，或在相机补偿坐标中做短时反向一致性；最近基线仍为 A2，并在新的冻结开发划分上验证。
5. 在 N2 通过前，安全默认方案是 M1（CMC 传播 + KF 预测裁剪 + appearance Top-1），而不是 E6 关联输出。

## 5. 必须记录的逐帧字段

- CMC：valid、fallback reason、inliers、inlier ratio、reprojection error、coverage、homography、elapsed time。
- KF：predicted box、covariance summary、age、predict-only count、innovation、Mahalanobis、`q_kf`。
- 响应：Top-1 score、Top-2 margin、Top-K candidate boxes/scores。
- 关联：backend、`q_cmc`、`q_kf`、`q_ambiguity`、lambda、每个候选 motion/final score、selected rank、reason。
- 更新：measurement accepted/rejected、rejection reason、output box、submodule timing。

GT 字段只能由离线分析器追加，不得进入在线决策对象。

## 6. 单元验证

- identity H 不改变状态和协方差几何。
- 纯平移 H 正确传播中心和速度端点。
- KF predict/update 与手算或可信参考实现一致。
- Top-1 候选与官方 OSTrack 最大峰解码一致。
- `lambda=0` 时输出与 appearance Top-1 完全一致。
- CMC/KF 无效时自动退化，不产生 NaN/Inf。
- diagnostics 不改变输出。
