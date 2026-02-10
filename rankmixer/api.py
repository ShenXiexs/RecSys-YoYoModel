import json
import os
import subprocess
import sys
from copy import deepcopy
from importlib import import_module

from .config_override import apply_overrides


class RankMixer:
    """RankMixer wrapper with configurable TrainConfig overrides and semantic groups."""

    def __init__(
        self,
        config_module="config.RankMixer_Shen0202.packaged",
        overrides=None,
        groups=None,
        groups_path=None,
        group_overrides=None,
        set_env=True,
    ):
        self.config_module = config_module
        self.overrides = deepcopy(overrides) if overrides else {}
        self.groups_path = groups_path
        self.group_overrides = group_overrides
        self._set_env = bool(set_env)
        if groups is not None:
            self.overrides.setdefault("train_params", {}).setdefault("rankmixer", {})[
                "semantic_groups"
            ] = groups

        if self._set_env:
            self._apply_env()

        self.TrainConfig = self._load_train_config()

    def _train_config_env_value(self):
        """Return TRAIN_CONFIG value as a path-like string for downstream scripts."""
        if not self.config_module:
            return None
        if self.config_module.endswith(".py") or "/" in self.config_module:
            return self.config_module
        return self.config_module.replace(".", "/") + ".py"

    def _apply_env(self):
        if not self._set_env:
            return
        train_config_env = self._train_config_env_value()
        if train_config_env:
            os.environ["TRAIN_CONFIG"] = train_config_env
        if self.overrides:
            os.environ["RANKMIXER_OVERRIDES"] = json.dumps(self.overrides, ensure_ascii=False)
        if self.groups_path:
            os.environ["RANKMIXER_GROUPS_PATH"] = self.groups_path
        if self.group_overrides:
            os.environ["RANKMIXER_GROUP_OVERRIDES"] = json.dumps(
                self.group_overrides, ensure_ascii=False
            )

    def _load_train_config(self):
        module = import_module(self.config_module)
        TrainConfig = getattr(module, "TrainConfig")
        if self.overrides or self.groups_path or self.group_overrides:
            apply_overrides(
                TrainConfig,
                overrides=self.overrides,
                groups_path=self.groups_path,
                group_overrides=self.group_overrides,
            )
        return TrainConfig

    def reload(self):
        """Reload TrainConfig after updating overrides/groups."""
        self._apply_env()
        self.TrainConfig = self._load_train_config()
        return self

    def set_overrides(self, overrides):
        self.overrides = deepcopy(overrides) if overrides else {}
        return self.reload()

    def set_groups(self, groups=None, groups_path=None, group_overrides=None):
        if groups is not None:
            self.overrides.setdefault("train_params", {}).setdefault("rankmixer", {})[
                "semantic_groups"
            ] = groups
        if groups_path is not None:
            self.groups_path = groups_path
        if group_overrides is not None:
            self.group_overrides = group_overrides
        return self.reload()

    def set_train_params(self, **kwargs):
        self.overrides.setdefault("train_params", {}).update(kwargs)
        return self.reload()

    def set_rankmixer_params(self, **kwargs):
        self.overrides.setdefault("train_params", {}).setdefault("rankmixer", {}).update(kwargs)
        return self.reload()

    def set_optimizer_params(self, **kwargs):
        self.overrides.setdefault("train_params", {}).setdefault("optimize_config", {}).update(kwargs)
        return self.reload()

    def set_input_params(self, **kwargs):
        self.overrides.setdefault("inp_fn_config", {}).update(kwargs)
        return self.reload()

    def set_run_config(self, **kwargs):
        self.overrides.setdefault("es_run_config", {}).update(kwargs)
        return self.reload()

    def set_device(
        self,
        device=None,
        gpu_list=None,
        gpu_memory_limit=None,
        gpu_memory_growth=None,
    ):
        attrs = {}
        if device is not None:
            attrs["device"] = device
        if gpu_list is not None:
            attrs["gpu_list"] = gpu_list
        if gpu_memory_limit is not None:
            attrs["gpu_memory_limit"] = gpu_memory_limit
        if gpu_memory_growth is not None:
            attrs["gpu_memory_growth"] = gpu_memory_growth
        self.overrides.update(attrs)
        return self.reload()

    def set_attr(self, **kwargs):
        """Set arbitrary TrainConfig attributes."""
        self.overrides.update(kwargs)
        return self.reload()

    def set_param(self, path, value):
        """Set an override by dotted path, e.g. 'train_params.rankmixer.d_model'."""
        keys = [k for k in str(path).split(".") if k]
        if not keys:
            return self
        cur = self.overrides
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value
        return self.reload()

    def available_params(self):
        """Return current TrainConfig params for inspection."""
        return {
            "model_version": getattr(self.TrainConfig, "model_version", None),
            "model_modul": getattr(self.TrainConfig, "model_modul", None),
            "train_params": deepcopy(getattr(self.TrainConfig, "train_params", {})),
            "inp_fn_config": deepcopy(getattr(self.TrainConfig, "inp_fn_config", {})),
            "es_run_config": deepcopy(getattr(self.TrainConfig, "es_run_config", {})),
            "device": getattr(self.TrainConfig, "device", None),
            "gpu_list": getattr(self.TrainConfig, "gpu_list", None),
        }

    def build_estimator(self, run_config=None, params=None, model_dir=None):
        """Build a tf.estimator.Estimator using the configured TrainConfig."""
        import tensorflow as tf

        model_fn_path = self.TrainConfig.model_modul
        model_fn_modul, _, model_fn_str = model_fn_path.rpartition(".")
        model_fn = getattr(import_module(model_fn_modul), model_fn_str)

        final_params = {
            "mode": "train",
            "ps_num": 1,
            "task_number": 1,
            "task_type": "chief",
            "task_idx": 0,
            "slot": "",
            "restrict": False,
            "device": getattr(self.TrainConfig, "device", "CPU"),
            "gpu_ids": getattr(self.TrainConfig, "gpu_list", "").split(","),
        }
        final_params.update(deepcopy(self.TrainConfig.train_params))
        if params:
            final_params.update(params)

        return tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir or (run_config and run_config.model_dir) or None,
            params=final_params,
            config=run_config,
        )

    def _run_main(
        self,
        mode,
        ckpt_dir,
        time_str,
        end_time_str=None,
        data_path=None,
        export_dir=None,
        file_list=None,
        slot="",
        job_name="",
        extra_args=None,
        check=True,
        log_path=None,
    ):
        code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_path = os.path.join(code_dir, "main.py")

        cmd = [sys.executable, main_path, "--mode", mode, "--ckpt_dir", ckpt_dir, "--time_str", time_str]
        if job_name:
            cmd.extend(["--job_name", job_name])
        if end_time_str:
            cmd.extend(["--end_time_str", end_time_str])
        if data_path:
            cmd.extend(["--data_path", data_path])
        if export_dir:
            cmd.extend(["--export_dir", export_dir])
        if file_list:
            cmd.extend(["--file_list", file_list])
        if slot:
            cmd.extend(["--slot", slot])
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        train_config_env = self._train_config_env_value()
        if train_config_env:
            env["TRAIN_CONFIG"] = train_config_env
        if self.overrides:
            env["RANKMIXER_OVERRIDES"] = json.dumps(self.overrides, ensure_ascii=False)
        if self.groups_path:
            env["RANKMIXER_GROUPS_PATH"] = self.groups_path
        if self.group_overrides:
            env["RANKMIXER_GROUP_OVERRIDES"] = json.dumps(self.group_overrides, ensure_ascii=False)

        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as wf:
                return subprocess.run(cmd, env=env, check=check, stdout=wf, stderr=wf)
        return subprocess.run(cmd, env=env, check=check)

    def train(self, ckpt_dir, time_str, end_time_str=None, data_path=None, **kwargs):
        return self._run_main(
            mode="train",
            ckpt_dir=ckpt_dir,
            time_str=time_str,
            end_time_str=end_time_str,
            data_path=data_path,
            **kwargs,
        )

    def eval(self, ckpt_dir, time_str, data_path=None, **kwargs):
        return self._run_main(
            mode="eval",
            ckpt_dir=ckpt_dir,
            time_str=time_str,
            data_path=data_path,
            **kwargs,
        )

    def export(self, ckpt_dir, time_str, export_dir, data_path=None, **kwargs):
        return self._run_main(
            mode="export",
            ckpt_dir=ckpt_dir,
            time_str=time_str,
            export_dir=export_dir,
            data_path=data_path,
            **kwargs,
        )
