# CMC-KF Candidate Association Experiments

本目录只保存实验定义和小型可追溯记录，不保存 tracker 实现或大型原始结果。

## 目录

```text
cmc_kf_candidate/
├── README.md
├── CHECKPOINT_REGISTRY.md
├── DATASET_REGISTRY.md
├── manifests/
│   └── manifest.template.yaml
├── records/
│   └── README.md
└── templates/
    └── report.template.md
```

## 规则

1. 每个实验先从 `manifest.template.yaml` 创建独立 manifest。
2. 实验 ID 遵守根目录 `EXPERIMENT_RUNBOOK.md`。
3. manifest 一旦开始运行即不可原地修改；配置变化必须新建实验 ID。
4. 每个实验的小型记录进入 `records/<experiment_id>/`。
5. 大型结果、逐帧日志和缓存保存在环境 `save_dir`，由 manifest 记录绝对路径。
6. E0-E6 含义只能来自 `docs/research/ABLATION_PLAN.md`。
7. 旧项目数据只能以 `source: archived_project` 登记，不能复制旧模块代码。
8. 论文方法兼容性变通使用 N* 编号，并在 `docs/research/METHOD_SOURCES.md` 和 manifest 同时登记。
9. 所有权重必须引用 `CHECKPOINT_REGISTRY.md` 中的唯一 `checkpoint_id` 和 SHA256。
10. 所有数据集必须引用 `DATASET_REGISTRY.md` 中的唯一 `dataset_id`。

## 当前阶段状态（2026-07-15）

- 基础实现与 v5 S0 已完成，E0-shadow 逐元素等价。
- 冻结 `development_v1` 的 E0/E1/M1/E3/E6 已完成；结果见 `docs/research/S1_RESULTS_20260715.md`。
- S1 promotion 为 `FAIL`：平均指标通过，但关联效果和尾部安全未通过。
- `holdout_v1` 未打开；不得开始 S3 或完整 benchmark。

## manifest 生命周期

```text
draft → registered → running → completed | failed | aborted
```

- `draft`：允许编辑。
- `registered`：实验参数冻结，生成实验 ID。
- `running`：只更新 status 和完成进度。
- `completed`：通过完整性检查并生成正式指标。
- `failed/aborted`：保留记录，修复后新建 run 编号。
