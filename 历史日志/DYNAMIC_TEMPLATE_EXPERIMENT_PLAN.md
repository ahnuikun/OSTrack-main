# Temporal History Token 实验计划

更新时间：2026-06-24

## 固定目标

验证在 OSTrack 框架上加入独立 `history tokens` 是否能提升无人机目标追踪，尤其是减少遮挡、低分辨率、相似目标、快速运动、视角变化导致的漂移。

本轮不再继续旧的 `TOKEN_TEMPLATE_UPDATE` 主线。旧路线结论保留为负结果：

```text
raw search patch token 更新 active template 没有超过 unchanged future-control；
learned gate 收缩接近 0；
fixed gate 也没有救回来。
```

因此下一轮核心不是继续调 gate，而是改成：

```text
z0 固定
history 独立
search 读取 z0 + history
history 从搜索帧递归生成和更新
```

## 统一采样

本机快速实验先使用近邻多帧样本：

```text
z0: 初始模板
s0: history 生成帧
s1: history 验证帧
s2: 递归验证帧
```

最小输入：

```text
1 template + 2 search
```

用于 H1/H2。

推荐输入：

```text
1 template + 3 search
```

用于 H3/H4，因为要体现：

```text
s0 -> H1
s1 -> H2
s2 -> 验证递归状态
```

## H0：无 history 对照

定义：

```text
z0 + s_t
不使用 history
```

要求：

- 与 H1-H4 使用同一采样方式；
- 主动忽略额外搜索帧；
- 记录 `s1` 或 `s2` 上的 matched future-control，避免比较对象不一致。

通过条件：

- 能稳定训练；
- 作为后续 delta 的唯一基线。

## H1-oracle：history 思想上限

定义：

```text
用 s0 的 GT 目标区域生成 O0
H1 = InitWriter(O0)
z0 + H1 + s1
```

目的：

验证“history 辅助 z0”是否有潜力。

注意：

- 这是 oracle，只能当上限；
- 不能作为真实方法结果；
- 如果 H1-oracle 不优于 H0，直接停止该方向。

核心指标：

```text
delta_oracle = IoU(z0 + H1_oracle + s1) - IoU(z0 + s1)
```

## H2-real：真实 response history

定义：

```text
z0 + s0 -> response peak
从 peak 附近抽取 O0
H1 = InitWriter(O0)
z0 + H1 + s1
```

目的：

验证推理可复现的 history 生成是否有效。

核心指标：

```text
delta_real = IoU(z0 + H1_real + s1) - IoU(z0 + s1)
gap = delta_oracle - delta_real
```

判断：

- 如果 H1-oracle 有效但 H2-real 无效，问题在 observation 生成；
- 如果 H2-real 接近 H1-oracle，进入 H3；
- 如果 H2-real 明显伤害，先修正 O0 来源，不做 reliability。

## H3-recurrent：递归 history

定义：

```text
z0 + s0 -> H1
z0 + H1 + s1 -> H2
z0 + H2 + s2 -> loss / eval
```

目的：

验证 history 是否能持续辅助后续帧，而不是只在一帧上偶然有效。

核心指标：

```text
IoU_H3_s2 - IoU_H0_s2
H2 是否比 H1 更好解释 s2
```

失败信号：

- s1 有收益，s2 变差；
- 递归越多漂移越大；
- history 学成无效常数。

## H4-reliability：学习型写入控制

定义：

```text
r_t = ReliabilityVerifier(z0, H_t, O_t, response_t)
H_{t+1} = r_t * UpdateWriter(H_t, O_t) + (1-r_t) * H_t
```

训练目标：

```text
好观测 -> r_t 高
坏观测 -> r_t 低
```

注意：

- H4 只能在 H2/H3 有正证据后做；
- 不能先靠 reliability 掩盖 observation 不成立；
- 推理时不能用 GT 判断可靠性。

## H5-hard UAV：论文级难例统计

统计维度：

- occlusion；
- low resolution；
- similar object；
- fast motion；
- viewpoint change。

要求：

- 每类至少报告 AUC / Precision / Normalized Precision；
- 给出失败案例；
- 给出 history 写入/冻结可视化；
- 报告参数量、FLOPs、FPS。

## 当前执行顺序

1. 新增 `MODEL.TOKEN_HISTORY` 配置；
2. 新增 `TemporalHistoryWriter`；
3. 扩展 ViT CE，支持 `z0 + history + search` 前向；
4. 实现 H0 matched future-control；
5. 实现 H1-oracle；
6. 实现 H2-real；
7. 本机快速跑 H0/H1/H2；
8. 只有 H1/H2 有正证据，才进入 H3；
9. H3 通过后再做 H4；
10. H5 留给服务器完整训练和数据集评估。

## 本机晋级门槛

快速实验只用于筛查机制，不直接作为论文结论。

晋级标准：

```text
H1-oracle: best/mean IoU 明显高于 H0 matched control
H2-real: 不能明显低于 H0，最好有正 delta
H3: 递归后不恶化，并在部分样本上提供正收益
```

停止规则：

```text
oracle 无效 -> 停止 history 路线
oracle 有效但 real 无效 -> 只优化 O_t，不加 reliability
real 有效但 recurrent 无效 -> 限制为短期 history
recurrent 有效 -> 再进入 learned reliability
```

## 2026-06-24 本机快速实验记录

统一条件：

```text
dataset: GOT10K_vottrain / GOT10K_votval
train samples: 200 per epoch
val samples: 50 per epoch
epochs: 10
batch size: 2
search/template: 256/128
CE: disabled for history-token mechanism test
checkpoint: keep best + last
```

结果：

| 阶段 | 定义 | 对照 | best val IoU | 结论 |
|---|---|---:|---:|---|
| H0-s1 | z0 + s1，无 history | - | 0.90315 | H1/H2 对照 |
| H1-oracle | GT 中心生成 O0，H1 辅助 s1 | H0-s1 | 0.90390 | 只有极小上限，约 +0.00075 |
| H2-real | response peak 生成 O0，H1 辅助 s1 | H0-s1 | 0.90435 | 极小正收益，约 +0.00120 |
| H0-s1s2 | z0 解释 s1/s2，无 history | - | 0.90223 | H3/H4 对照 |
| H3-recurrent | s0->H1，s1->H2，验证 s1/s2 | H0-s1s2 | 0.90405 | 极小正收益，约 +0.00182 |
| H4-reliability | learned r_t 控制 s1 观测是否写入 H2 | H0-s1s2 | 0.90673 | 数值最好，约 +0.00450 |

重要诊断：

```text
H3 history_gate_mean last val = 0.98391
H4 history_gate_mean last val = 0.97617
H4 reliability_mean last val = 0.99437
H4 reliability_target last val = 1.00000
```

解释：

- H2/H3/H4 在本机快速实验上有正数值，但幅度仍小；
- H4 的提升不能直接解释为“学会拒绝坏观测”，因为当前近邻验证样本几乎全是可靠观测；
- H4 当前只能说明：`z0 + history + search` 这条结构在干净近邻样本上没有崩，并且有初步正收益；
- 要证明抗漂移，必须进入 hard UAV 或人工构造坏观测统计，否则严格审稿人不会接受“可靠性模块有效”的结论。

下一步：

```text
先不继续堆模块。
优先补 H5-hard UAV 统计或 hard-sample 快速诊断：
    occlusion / low resolution / similar object / fast motion / viewpoint change
如果本机没有带属性标注的数据，就至少构造 response-low、IoU-low、尺度突变、搜索偏心四类训练/验证切片。
```

### H4 hard-sample probe

配置：

```text
experiments/ostrack/uav_history_h4_reliability_hardprobe_local.yaml
MAX_SAMPLE_INTERVAL = 200
SEARCH.CENTER_JITTER = 7
SEARCH.SCALE_JITTER = 0.6
epochs = 3
```

目的：

```text
不看最终性能，只观察更难采样下 reliability_target 是否会降低。
```

结果：

| epoch | val IoU | reliability_target | reliability_mean | s1_iou_for_reliability |
|---:|---:|---:|---:|---:|
| 1 | 0.28592 | 0.34000 | 0.37662 | 0.30157 |
| 2 | 0.25206 | 0.28000 | 0.28449 | 0.28879 |
| 3 | 0.22887 | 0.24000 | 0.27173 | 0.24534 |

判断：

- 原 H4 验证集后期 `reliability_target≈1.0`，主要是因为近邻样本太干净；
- hard-probe 能让 `reliability_target` 明显下降，说明当前标签机制确实能产生坏观测监督；
- 但该 hard-probe 太强，整体 IoU 过低，只能作为坏样本标签诊断；
- 后续需要构造更合理的 hard mix，而不是把所有样本都变成极难样本。
