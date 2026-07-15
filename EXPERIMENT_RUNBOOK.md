# CMC-KF Candidate Association Experiment Runbook

本文档是新项目的唯一实验操作入口。研究目标、模块边界和消融编号分别见：

- `docs/research/EXPERIMENT_CHARTER.md`
- `docs/research/MODULE_DESIGN.md`
- `docs/research/ABLATION_PLAN.md`
- `docs/research/METHOD_SOURCES.md`
- `experiments/cmc_kf_candidate/README.md`

任何新实验开始前必须先阅读上述四份文档。临时命令、聊天记录或旧项目报告不能覆盖这里的规则。

## 1. 固定环境与目录

- 新项目：`D:\PyCharm\Projects\OSTrack-main-cmc-kf-candidate`
- 旧实验档案（只读）：`D:\PyCharm\Projects\OSTrack-main`
- Python：`D:\Anaconda\envs\track\python.exe`
- GPU：默认只使用 GPU 0，测试参数固定为 `--num_gpus 1`
- 主配置：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml`
- 实验定义：`experiments/cmc_kf_candidate/manifests/`
- Checkpoint 身份：`experiments/cmc_kf_candidate/CHECKPOINT_REGISTRY.md`
- 数据集身份：`experiments/cmc_kf_candidate/DATASET_REGISTRY.md`
- 小型可追溯记录：`experiments/cmc_kf_candidate/records/<experiment_id>/`
- 大型原始输出：环境配置中的 `save_dir`

新项目不得导入旧项目的 Python 模块。旧目录只允许读取已经完成的结果、日志和报告，用于同 checkpoint 对照。

## 2. 实验开始前检查

每次运行必须完成：

```powershell
$python = 'D:\Anaconda\envs\track\python.exe'
$repo = 'D:\PyCharm\Projects\OSTrack-main-cmc-kf-candidate'

Set-Location $repo
git status --short
git rev-parse HEAD
& $python --version
& $python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

然后核对：

1. `lib/test/evaluation/local.py` 和 `lib/train/admin/local.py` 指向新项目及正确的数据、checkpoint、结果目录。
2. manifest 中的 `code_commit`、`checkpoint_id`、checkpoint 路径和 SHA256 已填写，并与 checkpoint registry 一致。
3. 对照组与实验组使用完全相同的 checkpoint、测试 YAML、搜索尺寸和 `dataset_id`。
4. 实验编号没有被复用；已有记录目录不得覆盖。
5. 当前阶段允许访问的数据集或序列与 manifest 一致。

## 3. 标准实验编号

格式：

```text
<stage>-<variant>-<checkpoint>-<dataset>-<date>-r<run>
```

示例：

```text
S1-E3-official-visdrone_dev-20260715-r01
S3-E6-official-uavdt-20260720-r01
```

字段含义：

- `stage`：`S0` 环境与基线、`S1` 机制验证、`S2` 冻结开发、`S3` 完整评测。
- `variant`：必须来自 `ABLATION_PLAN.md` 的 `E0` 至 `E6` 或已登记的关联后端编号。
- `checkpoint`：例如 `official`、`retrained`，不能使用含糊的 `default`。
- `dataset`：数据集或冻结序列组名称。
- `run`：同一配置的重复运行编号。

论文方法的兼容性变通必须额外填写 `adaptation_id: N#`。N* 不是随意改参数：它必须登记上游不兼容点、本地变通、预期机制、最接近的论文基线、独立开关和诊断字段。

## 4. 每次运行必须保存的记录

在运行前复制 manifest 模板，创建：

```text
experiments/cmc_kf_candidate/records/<experiment_id>/
├── manifest.yaml
├── command.ps1
├── environment.txt
├── status.json
├── metrics.csv
├── sequence_metrics.csv
└── report.md
```

大型逐帧日志、响应图和结果框保存在 `save_dir`，并在 manifest 的 `artifacts` 字段记录绝对路径。小型记录目录必须保留，不能只留下终端输出。

`status.json` 至少包含：`experiment_id`、`state`、`started_at`、`finished_at`、`exit_code`、`completed_sequences`、`expected_sequences` 和 `failure_reason`。

只有同时满足以下条件才能写 `state: completed`：

- 进程退出码为 0；
- 结果文件数量与预期序列数一致；
- 结果无 NaN、Inf、空文件或框数量不一致；
- 已使用正式评测器重新计算指标；
- `report.md` 已填写结论和下一 gate。

### 原始结果命名契约

OSTrack 框架原始框保存在：

```text
output/test/tracking_results/<tracker_name>/<parameter_name>/
```

命名固定为：

```text
tracker_name  = 算法结构，例如 ostrack_baseline / ostrack_cmc_only / ostrack_cmc_kf_assoc
parameter_name = <variant>__<checkpoint_id>__v<config_version>
```

示例：

```text
ostrack_cmc_kf_assoc/e6__official_vitb256_ce_ep300__v5
```

逐帧诊断和分析产物进入：

```text
output/cmc_kf_candidate/<experiment_id>/
├── diagnostics/
├── evaluator/
└── figures/
```

禁止在 tracker/parameter 名称中使用 `default`、`final`、`new`、`test` 等不可追溯词。配置或结构发生变化时递增 `config_version` 或新建 variant/adaptation ID，不能覆盖旧结果目录。

当前正式 development 版本为 `v5`。`v1`-`v4` 仅保留历史烟雾、失败诊断和修复证据，不得与 v5 混入同一正式结果表。

## 5. 运行顺序

所有模块版本按固定阶段推进：

1. `S0`：单元测试、checkpoint 严格加载、E0 基线复现、单序列 smoke。
2. `S1`：CMC 几何、KF 单步预测、Top-K oracle coverage 和候选关联救回/伤害分析。
3. `S2`：仅在冻结 development 序列上比较 E0-E6，选择唯一 full 配置并冻结参数。
4. `S3`：先打开 holdout，再完成 DTB70、UAVDT、VisDrone、UAV123 四个同等重要的 UAV 主数据集；最后运行 LaSOT 验证泛化与通用性。

禁止跳过 S1 直接根据完整 benchmark 调参；禁止在读取 holdout 后返回 development 修改参数并再次宣称 holdout 独立。

### 数据集执行顺序

默认顺序按当前帧数和预估耗时从短到长：

```text
DTB70 → UAVDT → VisDrone → UAV123 → LaSOT
```

前四项全部属于 UAV 主实验，必须完整测试和同等汇报。排序只用于更快获得阶段性结果，不构成 promotion gate；早期数据集效果好坏不能决定是否跳过后续 UAV 数据集。LaSOT 最后运行并单独形成泛化/通用性结果表。

已有论文方法与当前框架不兼容时，不要求原样复现。先实现最接近的论文基线，再用单独 N* 实验验证适配变化；不能把基线和多项适配同时首次引入。

## 6. 标准测试命令

单序列 smoke：

```powershell
& $python -u tracking\test.py `
  <tracker_name> <tracker_param> `
  --dataset_name <dataset> `
  --sequence <sequence> `
  --threads 0 `
  --num_gpus 1
```

单数据集：

```powershell
& $python -u tracking\test_uav_suite.py `
  --tracker_name <tracker_name> `
  --tracker_param <tracker_param> `
  --dataset <dataset> `
  --threads 0 `
  --num_gpus 1
```

正式评测：

```powershell
& $python -u tracking\analyze_uav_suite.py `
  --tracker_name <tracker_name> `
  --tracker_param <tracker_param> `
  --dataset <dataset> `
  --force_evaluation `
  --per_sequence
```

不得用截图、旧缓存或不同 checkpoint 的结果代替 `--force_evaluation` 输出。

## 7. 结果汇报顺序

报告必须先给数字，再解释机制：

1. 主表：AUC、Precision、Normalized Precision，以及相对 E0/E1 的 delta。
2. 消融表：E0-E6，保持 checkpoint、数据集和协议一致。
3. 逐序列表：平均 delta、最差序列、负增益序列比例、回归损失 P95/P99。
4. 机制表：CMC 有效率、KF 预测误差、Top-K oracle recall、候选切换率、rescue/harm。
5. 效率表：总 FPS、CMC/KF/候选关联各自耗时和显存。
6. 结论：Go/No-Go、证据、失败模式、下一步是否获准。

不得只汇报平均增益。任何 full 版本如果出现显著长尾恶化，都必须在结论中单独说明。

## 8. 长任务与恢复

优先使用前台顺序运行。必须后台运行时，使用隐藏窗口并验证进程仍存活、GPU 有负载、日志持续增长；打印 PID 不等于实验正在运行。

后台启动模板：

```powershell
$record = 'experiments\cmc_kf_candidate\records\<experiment_id>'
$stdout = Join-Path $record 'stdout.log'
$stderr = Join-Path $record 'stderr.log'
$args = @(
  '-u', 'tracking\test_uav_suite.py',
  '--tracker_name', '<tracker_name>',
  '--tracker_param', '<tracker_param>',
  '--dataset', '<dataset>',
  '--threads', '0',
  '--num_gpus', '1'
)
$process = Start-Process -FilePath $python -ArgumentList $args `
  -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content (Join-Path $record 'process.pid')
```

启动后至少验证一次：

```powershell
$pidValue = Get-Content (Join-Path $record 'process.pid')
Get-Process -Id $pidValue -ErrorAction Stop
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader
Get-Item $stdout, $stderr | Select-Object Name,Length,LastWriteTime
Get-Content $stdout -Tail 40
Get-Content $stderr -Tail 40
```

后续不高频轮询；按批次或用户约定的检查点读取状态。若后台进程已退出，必须根据退出码、日志和结果完整性判断成功或失败，不能只看 PID 文件。

长批次按 manifest 声明的序列顺序运行。已完成序列允许跳过，失败序列必须记录失败原因后重跑；不得删除已经完成的结果来制造“整洁运行”。

不递归扫描大型 `data/`、`output/` 或旧项目目录。检查进度时只读取当前实验的 `status.json`、日志尾部和预期输出目录。

## 9. 错误处理和停止条件

出现错误时：

1. 保存原始错误、命令、Git hash 和已完成序列数。
2. 判断是代码、配置、路径、资源还是数据问题。
3. 只修复当前实验范围内的问题。
4. 新建 run 编号重跑，不覆盖失败记录。
5. checkpoint、数据集和旧项目档案默认永不删除。

以下情况必须停止模块版本晋级或参数冻结：

- baseline 不能在同 checkpoint 下复现；
- Top-K 中缺少足够的正确候选，关联模块没有可选择对象；
- KF 预测误差或 CMC 失败使运动分数持续误导候选；
- aggregate 有增益但逐序列最差值或 P95/P99 回归超过冻结 gate；
- 实验记录不完整，无法证明使用了哪个代码、checkpoint 或参数。

这些 gate 管理的是模块版本和参数冻结，不管理四个 UAV 数据集是否运行；进入完整主实验后，四个 UAV 数据集均必须完成。

## 10. 旧结果迁移规则

旧项目保留原样，不复制旧残余模块代码。需要比较旧结果时，只迁移以下小型证据到新实验记录：

- 原始结果路径；
- tracker/parameter 名称；
- checkpoint 路径和 SHA256；
- 数据集、序列数量和评测命令；
- 正式指标及报告路径；
- 来源标记 `source: archived_project`。

如果上述任一信息缺失，旧数字只能作为参考，不能进入论文主表。
