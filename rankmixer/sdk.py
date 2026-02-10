import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from .api import RankMixer


def _load_json_or_yaml(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as rf:
        text = rf.read()
    if path.endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required for .yaml overrides.") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _normalize_config_module(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if value.endswith(".py") or "/" in value:
        module_path = value.replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]
        return module_path
    return value


def _resolve_config_module(task: Optional[str], config: Optional[str]) -> str:
    if config:
        return config
    if task:
        return f"config.{task}.packaged"
    return "config.RankMixer_Shen0202.packaged"


def _resolve_paths(
    task: Optional[str],
    ckpt_dir: Optional[str],
    data_path: Optional[str],
    export_dir: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not task:
        return ckpt_dir, data_path, export_dir
    model_root = os.environ.get("MODEL_ROOT", "/data/share/opt/model")
    data_root = os.environ.get("DATA_ROOT", "/data/share/opt/data")
    if not ckpt_dir:
        ckpt_dir = os.path.join(model_root, task, "ckpt")
    if not data_path:
        data_path = os.path.join(data_root, task)
    if not export_dir:
        export_dir = os.path.join(model_root, task, "export_dir")
    return ckpt_dir, data_path, export_dir


def _load_overrides(
    overrides: Optional[Union[str, Dict[str, Any]]],
    overrides_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    if overrides_path:
        loaded = _load_json_or_yaml(overrides_path)
        if loaded is None:
            return None
        if not isinstance(loaded, dict):
            raise TypeError("overrides_path must point to a JSON/YAML dict.")
        return loaded
    if overrides is None:
        return None
    if isinstance(overrides, dict):
        return overrides
    if isinstance(overrides, str):
        loaded = json.loads(overrides)
        if not isinstance(loaded, dict):
            raise TypeError("overrides JSON must be a dict.")
        return loaded
    raise TypeError("overrides must be a dict or JSON string.")


def _load_group_overrides(
    group_overrides: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]]
) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    if group_overrides is None:
        return None
    if isinstance(group_overrides, (dict, list)):
        return group_overrides
    if isinstance(group_overrides, str):
        return json.loads(group_overrides)
    raise TypeError("group_overrides must be a dict/list or JSON string.")


def create_rankmixer(
    task: Optional[str] = None,
    config: Optional[str] = None,
    overrides: Optional[Union[str, Dict[str, Any]]] = None,
    overrides_path: Optional[str] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    groups_path: Optional[str] = None,
    group_overrides: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    set_env: bool = True,
) -> RankMixer:
    """Create a RankMixer instance with normalized config and overrides."""
    config_module = _normalize_config_module(_resolve_config_module(task, config))
    overrides_dict = _load_overrides(overrides, overrides_path)
    group_overrides_obj = _load_group_overrides(group_overrides)
    return RankMixer(
        config_module=config_module,
        overrides=overrides_dict,
        groups=groups,
        groups_path=groups_path,
        group_overrides=group_overrides_obj,
        set_env=set_env,
    )


def build_estimator(
    task: Optional[str] = None,
    config: Optional[str] = None,
    overrides: Optional[Union[str, Dict[str, Any]]] = None,
    overrides_path: Optional[str] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    groups_path: Optional[str] = None,
    group_overrides: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    run_config=None,
    params: Optional[Dict[str, Any]] = None,
    model_dir: Optional[str] = None,
) :
    rm = create_rankmixer(
        task=task,
        config=config,
        overrides=overrides,
        overrides_path=overrides_path,
        groups=groups,
        groups_path=groups_path,
        group_overrides=group_overrides,
    )
    return rm.build_estimator(run_config=run_config, params=params, model_dir=model_dir)


def train(
    *,
    task: Optional[str] = None,
    config: Optional[str] = None,
    time_str: str,
    end_time_str: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    file_list: Optional[str] = None,
    slot: str = "",
    job_name: str = "",
    overrides: Optional[Union[str, Dict[str, Any]]] = None,
    overrides_path: Optional[str] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    groups_path: Optional[str] = None,
    group_overrides: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    extra_args: Optional[List[str]] = None,
):
    ckpt_dir, data_path, _ = _resolve_paths(task, ckpt_dir, data_path, None)
    if not ckpt_dir:
        raise ValueError("ckpt_dir is required when task is not provided.")
    if not data_path:
        raise ValueError("data_path is required when task is not provided.")
    rm = create_rankmixer(
        task=task,
        config=config,
        overrides=overrides,
        overrides_path=overrides_path,
        groups=groups,
        groups_path=groups_path,
        group_overrides=group_overrides,
    )
    return rm.train(
        ckpt_dir=ckpt_dir,
        time_str=time_str,
        end_time_str=end_time_str,
        data_path=data_path,
        file_list=file_list,
        slot=slot,
        job_name=job_name,
        extra_args=extra_args,
    )


def run_train(
    task: str,
    date: str,
    *,
    config: Optional[str] = None,
    overrides: Optional[Union[str, Dict[str, Any]]] = None,
    overrides_path: Optional[str] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    groups_path: Optional[str] = None,
    group_overrides: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    end_time_str: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    file_list: Optional[str] = None,
    slot: str = "",
    job_name: str = "",
    extra_args: Optional[List[str]] = None,
):
    """Short-hand train entry: task + date -> time_str."""
    if not task:
        raise ValueError("task is required.")
    if not date:
        raise ValueError("date is required.")
    return train(
        task=task,
        config=config,
        time_str=date,
        end_time_str=end_time_str,
        ckpt_dir=ckpt_dir,
        data_path=data_path,
        file_list=file_list,
        slot=slot,
        job_name=job_name,
        overrides=overrides,
        overrides_path=overrides_path,
        groups=groups,
        groups_path=groups_path,
        group_overrides=group_overrides,
        extra_args=extra_args,
    )


def eval(
    *,
    task: Optional[str] = None,
    config: Optional[str] = None,
    time_str: str,
    ckpt_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    file_list: Optional[str] = None,
    slot: str = "",
    job_name: str = "",
    overrides: Optional[Union[str, Dict[str, Any]]] = None,
    overrides_path: Optional[str] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    groups_path: Optional[str] = None,
    group_overrides: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    extra_args: Optional[List[str]] = None,
):
    ckpt_dir, data_path, _ = _resolve_paths(task, ckpt_dir, data_path, None)
    if not ckpt_dir:
        raise ValueError("ckpt_dir is required when task is not provided.")
    if not data_path:
        raise ValueError("data_path is required when task is not provided.")
    rm = create_rankmixer(
        task=task,
        config=config,
        overrides=overrides,
        overrides_path=overrides_path,
        groups=groups,
        groups_path=groups_path,
        group_overrides=group_overrides,
    )
    return rm.eval(
        ckpt_dir=ckpt_dir,
        time_str=time_str,
        data_path=data_path,
        file_list=file_list,
        slot=slot,
        job_name=job_name,
        extra_args=extra_args,
    )


def export(
    *,
    task: Optional[str] = None,
    config: Optional[str] = None,
    time_str: str,
    ckpt_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    export_dir: Optional[str] = None,
    file_list: Optional[str] = None,
    slot: str = "",
    job_name: str = "",
    overrides: Optional[Union[str, Dict[str, Any]]] = None,
    overrides_path: Optional[str] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    groups_path: Optional[str] = None,
    group_overrides: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    extra_args: Optional[List[str]] = None,
):
    ckpt_dir, data_path, export_dir = _resolve_paths(task, ckpt_dir, data_path, export_dir)
    if not ckpt_dir:
        raise ValueError("ckpt_dir is required when task is not provided.")
    if not export_dir:
        raise ValueError("export_dir is required when task is not provided.")
    rm = create_rankmixer(
        task=task,
        config=config,
        overrides=overrides,
        overrides_path=overrides_path,
        groups=groups,
        groups_path=groups_path,
        group_overrides=group_overrides,
    )
    return rm.export(
        ckpt_dir=ckpt_dir,
        time_str=time_str,
        export_dir=export_dir,
        data_path=data_path,
        file_list=file_list,
        slot=slot,
        job_name=job_name,
        extra_args=extra_args,
    )


def run(
    *,
    mode: str,
    task: Optional[str] = None,
    config: Optional[str] = None,
    time_str: str,
    end_time_str: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    export_dir: Optional[str] = None,
    file_list: Optional[str] = None,
    slot: str = "",
    job_name: str = "",
    overrides: Optional[Union[str, Dict[str, Any]]] = None,
    overrides_path: Optional[str] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    groups_path: Optional[str] = None,
    group_overrides: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    extra_args: Optional[List[str]] = None,
):
    ckpt_dir, data_path, export_dir = _resolve_paths(task, ckpt_dir, data_path, export_dir)
    if not ckpt_dir:
        raise ValueError("ckpt_dir is required when task is not provided.")
    rm = create_rankmixer(
        task=task,
        config=config,
        overrides=overrides,
        overrides_path=overrides_path,
        groups=groups,
        groups_path=groups_path,
        group_overrides=group_overrides,
    )
    return rm._run_main(
        mode=mode,
        ckpt_dir=ckpt_dir,
        time_str=time_str,
        end_time_str=end_time_str,
        data_path=data_path,
        export_dir=export_dir,
        file_list=file_list,
        slot=slot,
        job_name=job_name,
        extra_args=extra_args,
        check=True,
    )
