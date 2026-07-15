# Experiment Records

每个运行创建独立目录：

```text
records/<experiment_id>/
├── manifest.yaml
├── commands.ps1
├── environment.txt
├── status.json
├── metrics.csv
├── sequence_metrics.csv
└── report.md
```

此目录保存可复现论文数字所需的小文件。不要提交 checkpoint、逐帧响应图、原始视频或大型 JSONL；它们由 manifest 中的绝对 artifact 路径引用。

失败和中止记录不得删除或覆盖。重跑时递增 `r01`、`r02`。

`S1-E0-official-development_v1-20260715-r01` 因启动它的前端 stdout 管道提前关闭而失败于 12/20；`r02` 通过显式 `resume_from_experiment` 验证并复用前 12 条输出、补齐剩余序列。两份记录都必须保留。
