# Dataset Registry

验证日期：2026-07-15
验证环境：`D:\Anaconda\envs\track\python.exe`，通过项目正式 `get_dataset()` 加载器读取。

| dataset_id | 本地目录 | 加载器名称 | 序列数 | 帧数 | 实验角色 |
|---|---|---|---:|---:|---|
| `dtb70_local_v1` | `data/DTB70` | `dtb70` | 70 | 15,777 | UAV 主实验 |
| `uavdt_local_v1` | `data/uavdt` | `uavdt` | 50 | 37,084 | UAV 主实验 |
| `visdrone_local_v1` | `data/visdrone` | `visdrone` | 86 | 69,941 | UAV 主实验 |
| `uav123_local_v1` | `data/uav123` | `uav123` | 123 | 112,578 | UAV 主实验 |
| `lasot_test_local_v1` | `data/lasot/test` | `lasot` | 280 | 685,360 | 泛化与通用性验证 |

四个 UAV 数据集地位相同，都是方法主结果的组成部分。表中顺序仅按当前总帧数从少到多排列，目的是更快获得阶段性分析，不代表重要性、晋级关系或可选性。

项目还包含 `data/coco`、`data/got10k`、`data/lasot`、`data/UAV123_10fps` 等目录，但当前研究是纯测试时模块；未登记为正式评测集前不得自动加入实验矩阵。

## 使用规则

1. manifest 必须填写 `dataset_id`，不能只写显示名称。
2. 每次完整实验开始前重新验证序列数和帧数；任一变化必须新建 dataset ID。
3. development/holdout 序列清单必须单独保存并冻结，不能根据结果移除困难序列。
4. 数据集目录不得进入 Git，manifest 只记录本地路径、序列清单和版本证据。
5. 旧项目已经不再持有 `data/`；所有新实验只从本表路径读取。

## 新增登记模板

```text
dataset_id:
display_name:
loader_name:
local_path:
sequence_count:
frame_count:
split_or_subset:
source_version:
registered_at:
notes:
```
