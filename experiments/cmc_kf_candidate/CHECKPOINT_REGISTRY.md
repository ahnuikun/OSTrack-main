# Checkpoint Registry

本表是实验中引用模型权重的唯一身份来源。实验报告和 manifest 必须使用 `checkpoint_id`，不能只写 `OSTrack_ep0300.pth.tar`。

| checkpoint_id | 身份 | 文件路径 | 大小（bytes） | SHA256 |
|---|---|---|---:|---|
| `official_vitb256_ce_ep300` | 官方 OSTrack ViT-B/256 CE，epoch 300 | `output/checkpoints/train/ostrack/official_vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar` | 370179249 | `8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41` |
| `retrained_vitb256_ce_ep300` | 本地重新训练 ViT-B/256 CE，epoch 300 | `output/checkpoints/train/ostrack/retrained_vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar` | 1111836284 | `91f53eec3f33fb08f06d5f9ee726cb33fbf572520b68b0dc360044cfed458b63` |

## 使用规则

1. 同一对照表中的方法必须使用相同 `checkpoint_id`。
2. `official` 与 `retrained` 必须分表或分 panel 汇报，不能把其中一个当作另一个的 baseline。
3. 权重重命名、移动或替换后必须重新计算 SHA256；hash 改变视为新的 checkpoint。
4. `pretrained_models/mae_pretrain_vit_base.pth` 是训练初始化权重，不是测试 checkpoint，不能填入本表的实验 checkpoint 字段。
5. 参数文件必须显式解析到本表中的一个路径，不允许依赖含糊的默认 checkpoint 目录。

E0 基线的规范参数名：

```text
e0__official_vitb256_ce_ep300__v1
e0__retrained_vitb256_ce_ep300__v1
```

对应原始结果目录分别为：

```text
output/test/tracking_results/ostrack/e0__official_vitb256_ce_ep300__v1
output/test/tracking_results/ostrack/e0__retrained_vitb256_ce_ep300__v1
```

## 新增登记模板

```text
checkpoint_id:
origin:
architecture:
training_data:
epoch:
path:
size_bytes:
sha256:
registered_at:
notes:
```
