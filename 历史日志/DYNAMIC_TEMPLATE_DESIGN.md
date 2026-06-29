# Temporal History Token for OSTrack：模块设计记录

更新时间：2026-06-24

## 当前主线

现在的主线不再是“动态替换模板”，而是：

```text
固定初始模板 z0 负责身份锚定
可更新 history tokens 负责历史外观状态
当前搜索帧 s_t 通过 OSTrack 一流式 self-attention 同时读取 z0 与 history
head 仍然只在 search tokens 上输出目标位置
```

也就是：

```text
过去帧信息 -> 形成 H_t
z0 + H_t + s_t -> 输出目标位置
s_t 的可靠观测 -> 更新 H_{t+1}
```

关键边界：

- `z0` 永远不被污染；
- `history` 是隐式在线状态，初始不存在；
- `history` 不是新模板，也不是 RGB 图像融合；
- 训练时必须用多帧展开体现 `history` 的生成、读取、更新；
- 推理时 `history` 随视频递归维护。

## history 如何产生

初始帧没有 history：

```text
t=0:
    z0 + s0 -> pred0
    从 s0 的目标区域特征生成 O0
    H1 = InitWriter(O0)
```

从第二个搜索帧开始：

```text
t>=1:
    z0 + H_t + s_t -> pred_t
    从 s_t 的目标区域特征生成 O_t
    H_{t+1} = UpdateWriter(H_t, O_t, r_t)
```

其中 `O_t` 不应来自 RGB 裁剪，也不应优先用 raw patch token。更合理的来源是：

```text
OSTrack encoder 后的 search target-region tokens
```

原因：这些 tokens 已经通过一流式 self-attention 读取过模板信息，比 raw patch token 更接近“被模板解释后的目标表征”。

## history token 形态

本阶段建议使用空间 history：

```text
H_t: [B, 64, C]
```

即与 `128x128` 模板的 `8x8` token 网格一致。这样可以保留目标局部结构，避免把目标压成一个向量后丢失空间信息。

接入 ViT 时：

```text
[z0 tokens, history tokens, search tokens] -> transformer -> search head
```

位置编码原则：

- `z0` 使用 template position + template segment；
- `history` 使用 template-grid position + history segment；
- `search` 使用 search position + search segment；
- head 只读取最后的 search tokens。

## H0-H5 实验阶梯

### H0：无 history 对照

```text
z0 + s_t
```

不使用 history。所有 history 实验必须使用相同多帧采样，只是主动忽略历史帧。

### H1-oracle：GT 区域 history 上限诊断

```text
z0 + s0 -> 只用于建立时间上下文
GT 区域生成 O0
H1 = InitWriter(O0)
z0 + H1 + s1 -> 验证
```

目的不是作为真实方法，而是判断“history 作为辅助状态”有没有理论潜力。

如果 H1-oracle 都没有提升，说明这个方向不值得继续。

### H2-real：response peak 生成 history

```text
z0 + s0 -> response peak
从 peak 周围抽取 O0
H1 = InitWriter(O0)
z0 + H1 + s1 -> 验证
```

目标是验证真实推理可用的 `O0` 是否能带来收益。

### H3-recurrent：递归 history

```text
z0, s0, s1, s2
H1 从 s0 来
H2 从 s1 更新
验证 s2
```

目的：验证 history 是否能连续累积有效信息，而不是只做一次性增强。

### H4-reliability：可靠性写入

加入 learned `r_t`：

```text
好观测 -> 更新 history
坏观测 -> 冻结 history
```

这里的可靠性不是手写阈值主模块，而是训练出来的写入控制。训练标签可由未来帧表现或 GT 诊断产生，但推理输入必须只依赖模型可见信息。

### H5-hard UAV：难例专项统计

按无人机追踪难点单独报告：

- 遮挡；
- 低分辨率；
- 相似目标；
- 快速运动；
- 视角剧烈变化。

H5 不是新模块，而是论文级证据要求。

## 当前禁止事项

- 不把 search RGB 裁剪图像和模板图像做融合；
- 不把 history 直接替换 `z0`；
- 不用 GT crop 作为真实推理 candidate；
- 不继续堆复杂 gate 来弥补错误 observation；
- 不把 oracle 结果包装成真实方法收益；
- 不在 H2 之前集成 candidate rescue。

## 严格审稿人视角的最低要求

要作为论文主模块，至少需要证明：

1. H0 同采样对照下，history 收益不是采样变化造成的；
2. H1-oracle 存在明显上限；
3. H2-real 能接近 H1-oracle 的一部分收益；
4. H3-recurrent 不因递归写入导致漂移加重；
5. H4-reliability 能降低坏观测污染；
6. H5-hard UAV 难例中有可解释收益；
7. 计算量、参数量、推理延迟清楚可控。

## 当前本机证据状态

2026-06-24 快速实验显示：

```text
H0-s1 best IoU        = 0.90315
H1-oracle best IoU    = 0.90390
H2-real best IoU      = 0.90435
H0-s1s2 best IoU      = 0.90223
H3-recurrent best IoU = 0.90405
H4-reliability best   = 0.90673
```

当前判断：

- history-token 主线比旧 active-template-update 更合理；
- 数值上有初步正证据；
- 但 H1-oracle 上限很薄，说明普通 GOT10K 近邻样本并不强依赖 history；
- H4 的 reliability 在当前验证集几乎全预测为接受，尚未证明坏观测冻结能力；
- 下一步必须面向 UAV hard cases，而不是继续堆结构。

因此当前模块设计应保持精简：

```text
z0 fixed identity anchor
H_t spatial temporal state
encoded-search observation O_t
InitWriter / UpdateWriter
minimal learned reliability r_t
```

暂不加入 candidate rescue、top-K tracklet、多模板竞争等复杂分支。
