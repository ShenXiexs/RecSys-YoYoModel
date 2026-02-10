# RankMixer Wrapper Guide

**Python API (library usage)**
Minimal usage (wrapper function):

```python
from rankmixer import train

train(
    task="RankMixer_Shen0202",
    time_str="20260201",
)
```

Shorter entry:

```python
from rankmixer import run_train

run_train("RankMixer_Shen0202", "20260201")
```

With overrides / groups:

```python
from rankmixer import train

train(
    task="RankMixer_Shen0202",
    time_str="20260201",
    overrides_path="/path/to/overrides.json",
    groups_path="/path/to/semantic_groups.json",
)
```

You can also create an instance directly:

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

You can also run the example script directly:

```bash
python rankmixer/example_api.py \
  --task RankMixer_Shen0202 \
  --time_str 20260201 \
  --overrides_path /path/to/overrides.json \
  --groups_path /path/to/semantic_groups.json
```

Test script (wrapper equivalent to `bin/test.sh <TASK> <DATE>`):

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup python rankmixer/test_equiv.py \
  --task RankMixer_Shen0209 \
  --config_task RankMixer_Shen0202 \
  --date 20260201 \
  > /data/share/opt/model/RankMixer_Shen0209/logs/RankMixer_Shen0206.log 2>&1 &
```

By default it also writes:
- `logs/main_<end_date>.log` (training log)
- `logs/<end_date>.eval` (evaluation log)

**Specify config module (independent of task naming)**

```bash
python -m rankmixer --config config.RankMixer_Shen0202.packaged --mode train --time_str 20260201
```

Path form also supported:

```bash
python -m rankmixer --config config/RankMixer_Shen0202/packaged.py --mode train --time_str 20260201
```

**Directories and default paths**

- `--task` is automatically mapped to `config/<task>/packaged.py`
- If `--task` is not provided, specify `--ckpt_dir` / `--data_path` / `--export_dir` manually
- Default paths read environment variables: `MODEL_ROOT=/data/share/opt/model`, `DATA_ROOT=/data/share/opt/data`
- If your output directory should be `RankMixer_Shen0209` but the config still references `RankMixer_Shen0202`:
  use `--task RankMixer_Shen0209` together with `--config` or `--config_task RankMixer_Shen0202`

**Use from other repos/projects**

- Prereq: this repo is accessible (at least `rankmixer/` and `config/`)
- CLI: add the repo root to `PYTHONPATH`

```bash
PYTHONPATH=/path/to/yoyo_model_shen2 \
python -m rankmixer --config config.RankMixer_Shen0202.packaged --mode train --time_str 20260201
```

- Python: add the repo path, then call the wrapper function or instance

```python
import sys
sys.path.insert(0, "/path/to/yoyo_model_shen2")
from rankmixer import train, create_rankmixer

train(task="RankMixer_Shen0202", time_str="20260201")
# or
rm = create_rankmixer(task="RankMixer_Shen0202")
rm.train(ckpt_dir="...", time_str="20260201", data_path="...")
```

- Note: `--config config/xxx/packaged.py` must be run from the repo root, or ensure the repo root is in `PYTHONPATH`

**Common scenarios**
Training:

```bash
python -m rankmixer --task RankMixer_Shen0202 --mode train --time_str 20260201
```

Evaluation:

```bash
python -m rankmixer --task RankMixer_Shen0202 --mode eval --time_str 20260201
```

Export:

```bash
python -m rankmixer --task RankMixer_Shen0202 --mode export --time_str 20260201
```

Inference:

```bash
python -m rankmixer --task RankMixer_Shen0202 --mode infer --time_str 20260201
```

**Time arguments**

- `--time_str` is required, supports `YYYYMMDD` or `YYYYMMDDHHMM`
- `--end_time_str` is optional, for batch ranges

**Overrides and semantic groups**

- `--overrides`: JSON string, directly overrides `TrainConfig` (deep-merge dict)
- `--overrides_path`: JSON/YAML file (higher priority than `--overrides`)
- `--groups_path`: JSON/YAML file, used directly as `semantic_groups`
- `--group_overrides`: JSON string, override or replace by group name
- Rule: if `--groups_path` is provided, `--group_overrides` is ignored

Example (file-based, recommended):

```bash
python -m rankmixer \
  --task RankMixer_Shen0202 \
  --mode train \
  --time_str 20260201 \
  --overrides_path /path/to/overrides.json \
  --groups_path /path/to/semantic_groups.json
```

**Python direct usage**

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

**Still using bin scripts but overriding config**
No need to modify `bin/` scripts; just add environment variables:

```bash
RANKMIXER_OVERRIDES_PATH=/path/to/overrides.json \
RANKMIXER_GROUPS_PATH=/path/to/semantic_groups.json \
nohup bash bin/test.sh RankMixer_Shen0202 20260201 > ... 2>&1 &
```

**CLI argument overview**

- `--task` task name (maps to `config/<task>/packaged.py`)
- `--config` config module or path
- `--mode` `train|eval|export|infer|feature_eval`
- `--time_str` / `--end_time_str`
- `--ckpt_dir` / `--data_path` / `--export_dir`
- `--overrides` / `--overrides_path`
- `--groups_path` / `--group_overrides`
- `--file_list` / `--slot` / `--job_name`
- `--extra_arg` extra args passed through to `main.py` (repeatable)
