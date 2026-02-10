# RankMixer 封装版说明

**Python API（库调用）**
最简调用（封装版函数）：

```python
from rankmixer import train

train(
    task="RankMixer_Shen0202",
    time_str="20260201",
)
```

更短入口：

```python
from rankmixer import run_train

run_train("RankMixer_Shen0202", "20260201")
```

带 overrides / groups：

```python
from rankmixer import train

train(
    task="RankMixer_Shen0202",
    time_str="20260201",
    overrides_path="/path/to/overrides.json",
    groups_path="/path/to/semantic_groups.json",
)
```

也可以直接创建实例：

```python
from rankmixer import create_rankmixer

rm = create_rankmixer(
    task="RankMixer_Shen0202",
    overrides_path="/path/to/overrides.json",
)
rm.train(
    ckpt_dir="/data/share/opt/model/RankMixer_Shen0202/ckpt",
    time_str="20260201",
    data_path="/data/share/opt/data/RankMixer_Shen0202",
)
```

也可以直接运行示例脚本：

```bash
python rankmixer/example_api.py \
  --task RankMixer_Shen0202 \
  --time_str 20260201 \
  --overrides_path /path/to/overrides.json \
  --groups_path /path/to/semantic_groups.json
```
测试脚本（等价于 `bin/test.sh <TASK> <DATE>` 的封装版）：
```bash
CUDA_VISIBLE_DEVICES=0 \
nohup python rankmixer/test_equiv.py \
  --task RankMixer_Shen0209 \
  --config_task RankMixer_Shen0202 \
  --date 20260201 \
  > /data/share/opt/model/RankMixer_Shen0209/logs/RankMixer_Shen0206.log 2>&1 &
```
默认会额外写入：
- `logs/main_<end_date>.log`（训练日志）
- `logs/<end_date>.eval`（评估日志）

**指定配置模块（这里并不依赖 task 命名）**

```bash
python -m rankmixer --config config.RankMixer_Shen0202.packaged --mode train --time_str 20260201
```

也支持路径形式：

```bash
python -m rankmixer --config config/RankMixer_Shen0202/packaged.py --mode train --time_str 20260201
```

**目录与默认路径**

- `--task` 会自动映射到 `config/<task>/packaged.py`
- `--task` 未提供时，需要手动指定 `--ckpt_dir` / `--data_path` / `--export_dir`
- 默认路径会读取环境变量：`MODEL_ROOT=/data/share/opt/model`，`DATA_ROOT=/data/share/opt/data`
- 如果输出目录想用 `RankMixer_Shen0209`，但配置仍引用 `RankMixer_Shen0202`：请用 `--task RankMixer_Shen0209` 搭配 `--config` 或 `--config_task RankMixer_Shen0202`

**给其他人/其他项目调用**

- 前提：本仓库代码可访问（至少包含 `rankmixer/` 与 `config/`）
- CLI 方式：把仓库根目录加入 `PYTHONPATH`

```bash
PYTHONPATH=/path/to/yoyo_model_shen2 \
python -m rankmixer --config config.RankMixer_Shen0202.packaged --mode train --time_str 20260201
```

- Python 方式：在脚本里加入仓库路径，然后调用封装函数或实例

```python
import sys
sys.path.insert(0, "/path/to/yoyo_model_shen2")
from rankmixer import train, create_rankmixer

train(task="RankMixer_Shen0202", time_str="20260201")
# 或
rm = create_rankmixer(task="RankMixer_Shen0202")
rm.train(ckpt_dir="...", time_str="20260201", data_path="...")
```

- 注意：`--config config/xxx/packaged.py` 需要在仓库根目录执行，或保证仓库根目录在 `PYTHONPATH` 中

**常用场景**
训练：

```bash
python -m rankmixer --task RankMixer_Shen0202 --mode train --time_str 20260201
```

评估：

```bash
python -m rankmixer --task RankMixer_Shen0202 --mode eval --time_str 20260201
```

导出：

```bash
python -m rankmixer --task RankMixer_Shen0202 --mode export --time_str 20260201
```

推理：

```bash
python -m rankmixer --task RankMixer_Shen0202 --mode infer --time_str 20260201
```

**时间参数**

- `--time_str` 必填，支持 `YYYYMMDD` 或 `YYYYMMDDHHMM`
- `--end_time_str` 可选，用于批量区间

**参数覆盖与语义分组**

- `--overrides`：JSON 字符串，直接覆盖 `TrainConfig`（深度合并 dict）
- `--overrides_path`：JSON/YAML 文件（优先级高于 `--overrides`）
- `--groups_path`：JSON/YAML 文件，直接作为 `semantic_groups`
- `--group_overrides`：JSON 字符串，按组名覆盖或整体替换
- 规则：如果提供 `--groups_path`，则忽略 `--group_overrides`

示例（文件方式，推荐）：

```bash
python -m rankmixer \
  --task RankMixer_Shen0202 \
  --mode train \
  --time_str 20260201 \
  --overrides_path /path/to/overrides.json \
  --groups_path /path/to/semantic_groups.json
```

**Python 直接使用**

```python
from rankmixer import RankMixer

rm = RankMixer(
    config_module="config.RankMixer_Shen0202.packaged",
    overrides={
        "train_params": {
            "rankmixer": {"d_model": 256, "num_layers": 6},
            "optimize_config": {"learning_rate": 2e-4},
        }
    },
    groups=[
        {"name": "user_basic", "features": ["user_id", "gender", "age"]},
        {"name": "context", "features": ["adslot_id", "channel_id"]},
    ],
)

estimator = rm.build_estimator()
# rm.train(ckpt_dir="...", time_str="202602010000", data_path="...")
```

**依旧选择 bin 脚本但要覆盖配置**
不改 `bin/` 脚本，直接加环境变量：

```bash
RANKMIXER_OVERRIDES_PATH=/path/to/overrides.json \
RANKMIXER_GROUPS_PATH=/path/to/semantic_groups.json \
nohup bash bin/test.sh RankMixer_Shen0202 20260201 > ... 2>&1 &
```

**CLI 参数一览**

- `--task` 任务名（映射到 `config/<task>/packaged.py`）
- `--config` 配置模块或路径
- `--mode` `train|eval|export|infer|feature_eval`
- `--time_str` / `--end_time_str`
- `--ckpt_dir` / `--data_path` / `--export_dir`
- `--overrides` / `--overrides_path`
- `--groups_path` / `--group_overrides`
- `--file_list` / `--slot` / `--job_name`
- `--extra_arg` 额外透传给 `main.py` 的参数（可重复）
