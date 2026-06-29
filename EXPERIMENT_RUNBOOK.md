# OSTrack Experiment Runbook

本文档记录在当前 Windows 工作站上启动和轮询 OSTrack 长时间训练/测试任务的方法。后续上下文压缩后，优先按这里执行。

## 环境

- Repo: `D:\PyCharm\Projects\OSTrack-main`
- Python: `D:\Anaconda\envs\track\python.exe`
- 主要训练入口：`tracking\train.py`
- 模块实验入口：`tracking\run_module_experiment.py`
- Launcher logs: `output\launcher_logs`
- Training logs: `output\logs`
- Module logs: `output\module_experiments\<config>`
- Checkpoints: `output\checkpoints\train\ostrack\<config>`

## 启动原则

优先使用最稳定、最直接的方式运行训练和测试：直接调用训练本体或 `tracking/run_module_experiment.py`。不要为了形式上的 VSCode/cmd/隐藏窗口而改变启动链。

长时间任务可以用 shell 前台或后台运行；如需后台运行，必须确认日志持续增长、GPU 有实际训练/测试负载，并按规定轮询。不要把“进程已启动”等同于“实验已运行”。

不要在训练和测试过程中递归扫描整个项目，`data/` 和 `output/` 很大，会导致轮询命令变慢。

## 固定测试 Gate

后续所有模块测试必须按以下顺序执行，不得临时改成其他顺序：

1. 先测试并汇报 `visdrone_test`、`uavdt`、`dtb70` 三个数据集。
2. 汇报时必须给出详细数据表：AUC、Baseline AUC、Delta AUC、Precision、Baseline Precision、Delta Precision、Norm Precision、Baseline Norm Precision、Delta Norm Precision。
3. 如果 `visdrone_test` 和 `uavdt` 的 `Delta AUC` 都达到 `+0.5` 以上，则补测 `uav123`。
4. 如果 `uav123` 的 `Delta AUC` 也达到 `+0.5` 以上，则再测试 `lasot`。
5. `dtb70` 必须测试和汇报，但它不是当前主线晋级硬门槛；当 VisDrone/UAVDT 均明显提升时，不要把结论过度集中在 DTB70。
6. 最终汇报必须包含所有已测数据集的完整结果，以及每个 gate 的继续/停止原因。

推荐使用自动 gate：

```powershell
D:\Anaconda\envs\track\python.exe -u tracking\run_module_experiment.py `
    --config <config_name> `
    --gate_sequence `
    --analyze `
    --gpu 0 `
    --seed 42
```

## 启动模块训练

```powershell
$python = 'D:\Anaconda\envs\track\python.exe'
$repo = 'D:\PyCharm\Projects\OSTrack-main'
$config = '<config_name>'
$args = @(
    'tracking\run_module_experiment.py',
    '--config', $config,
    '--train',
    '--gpu', '0'
)
Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repo `
    -WindowStyle Hidden
```

## 启动无人机测试与分析

```powershell
$python = 'D:\Anaconda\envs\track\python.exe'
$repo = 'D:\PyCharm\Projects\OSTrack-main'
$config = '<config_name>'
$args = @(
    'tracking\run_module_experiment.py',
    '--config', $config,
    '--test_uav',
    '--analyze',
    '--force_evaluation',
    '--gpu', '0'
)
Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repo `
    -WindowStyle Hidden
```

## 启动 LaSOT 测试

```powershell
$python = 'D:\Anaconda\envs\track\python.exe'
$repo = 'D:\PyCharm\Projects\OSTrack-main'
$config = '<config_name>'
$args = @(
    'tracking\run_module_experiment.py',
    '--config', $config,
    '--test_lasot',
    '--analyze',
    '--force_evaluation',
    '--gpu', '0'
)
Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repo `
    -WindowStyle Hidden
```

## 训练轮询

训练时每五分钟轮询一次：

```powershell
Start-Sleep -Seconds 300
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime
Get-ChildItem output\logs -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 5 Name,Length,LastWriteTime
Get-Content -Path output\logs\ostrack-<config>.log -Tail 80
```

关键训练事件：

```powershell
Select-String -Path output\logs\ostrack-<config>.log `
    -Pattern 'train:|val:|EarlyStopping|Saving|best' |
    Select-Object -Last 120
```

## 测试轮询

测试时每十分钟轮询一次：

```powershell
Start-Sleep -Seconds 600
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime
Get-ChildItem output\module_experiments\<config> -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 Name,Length,LastWriteTime
```

## 错误处理

训练或测试出现错误时，不需要等待额外指令，只要问题明确且修改范围局限于当前实验，就按下面流程处理：

1. 读取最新 launcher log、module log 和 training log。
2. 定位失败点。
3. 如果是代码、配置或路径问题，直接修复。
4. 重新启动同一个实验。
5. 不删除 checkpoint，除非用户明确要求清理。

当前已经明确要求清理旧实验，因此旧分支日志和结果可以删除；之后新的实验日志默认保留到用户再次要求清理。
