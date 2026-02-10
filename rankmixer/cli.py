import argparse
import json
import os
import sys

from .api import RankMixer
from .config_override import _load_json_or_yaml


def _load_json_arg(value, label):
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc


def _resolve_config_module(task, config):
    if config:
        return config
    if task:
        return f"config.{task}.packaged"
    return "config.RankMixer_Shen0202.packaged"


def _normalize_config_module(value):
    if not value:
        return value
    if value.endswith(".py") or "/" in value:
        module_path = value.replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]
        return module_path
    return value


def _resolve_paths(task, ckpt_dir, data_path, export_dir):
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


def build_parser():
    parser = argparse.ArgumentParser(description="RankMixer wrapper CLI")
    parser.add_argument("--task", help="Model task name, e.g. RankMixer_Shen0202")
    parser.add_argument("--config", help="TrainConfig module or path, e.g. config.RankMixer_Shen0202.packaged")
    parser.add_argument("--mode", default="train", help="train|eval|export|infer|feature_eval")
    parser.add_argument("--time_str", required=True, help="Training time string, e.g. 20260201 or 202602010000")
    parser.add_argument("--end_time_str", help="Optional end time string")
    parser.add_argument("--ckpt_dir", help="Checkpoint directory")
    parser.add_argument("--data_path", help="Data path")
    parser.add_argument("--export_dir", help="Export directory (for export mode)")
    parser.add_argument("--file_list", help="Optional file list path")
    parser.add_argument("--slot", default="", help="Slot name")
    parser.add_argument("--job_name", default="", help="Job name")
    parser.add_argument("--overrides", help="Overrides JSON string")
    parser.add_argument("--overrides_path", help="Overrides JSON/YAML file path")
    parser.add_argument("--groups_path", help="Semantic groups JSON/YAML file path")
    parser.add_argument("--group_overrides", help="Group overrides JSON string")
    parser.add_argument("--extra_arg", action="append", default=[], help="Extra arg to pass to main.py")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = None
    if args.overrides_path:
        overrides = _load_json_or_yaml(args.overrides_path)
    elif args.overrides:
        overrides = _load_json_arg(args.overrides, "overrides")

    group_overrides = _load_json_arg(args.group_overrides, "group_overrides")

    config_module = _normalize_config_module(_resolve_config_module(args.task, args.config))
    ckpt_dir, data_path, export_dir = _resolve_paths(
        args.task, args.ckpt_dir, args.data_path, args.export_dir
    )

    if not ckpt_dir:
        parser.error("ckpt_dir is required when task is not provided.")
    if not data_path and args.mode in ("train", "eval", "feature_eval", "infer"):
        parser.error("data_path is required when task is not provided.")

    rm = RankMixer(
        config_module=config_module,
        overrides=overrides,
        groups_path=args.groups_path,
        group_overrides=group_overrides,
    )

    mode = args.mode
    if mode == "train":
        result = rm.train(
            ckpt_dir=ckpt_dir,
            time_str=args.time_str,
            end_time_str=args.end_time_str,
            data_path=data_path,
            file_list=args.file_list,
            slot=args.slot,
            job_name=args.job_name,
            extra_args=args.extra_arg or None,
        )
    elif mode == "eval":
        result = rm.eval(
            ckpt_dir=ckpt_dir,
            time_str=args.time_str,
            data_path=data_path,
            file_list=args.file_list,
            slot=args.slot,
            job_name=args.job_name,
            extra_args=args.extra_arg or None,
        )
    elif mode == "export":
        if not export_dir:
            parser.error("export_dir is required for export mode.")
        result = rm.export(
            ckpt_dir=ckpt_dir,
            time_str=args.time_str,
            export_dir=export_dir,
            data_path=data_path,
            file_list=args.file_list,
            slot=args.slot,
            job_name=args.job_name,
            extra_args=args.extra_arg or None,
        )
    else:
        result = rm._run_main(
            mode=mode,
            ckpt_dir=ckpt_dir,
            time_str=args.time_str,
            end_time_str=args.end_time_str,
            data_path=data_path,
            export_dir=export_dir,
            file_list=args.file_list,
            slot=args.slot,
            job_name=args.job_name,
            extra_args=args.extra_arg or None,
            check=True,
        )

    if hasattr(result, "returncode"):
        sys.exit(result.returncode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
