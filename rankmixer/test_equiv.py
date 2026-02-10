#!/usr/bin/env python3
# Test script to mimic: bin/test.sh <TASK> <DATE>
# while using the RankMixer wrapper (Python API).
#
# Example (equivalent to the bash command in effect):
#   CUDA_VISIBLE_DEVICES=0 \
#   nohup python rankmixer/test_equiv.py \
#     --task RankMixer_Shen0209 \
#     --config_task RankMixer_Shen0202 \
#     --date 20260201 \
#     > /data/share/opt/model/RankMixer_Shen0209/logs/RankMixer_Shen0206.log 2>&1 &

import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rankmixer import create_rankmixer


def _parse_date(value: str) -> datetime:
    s = str(value).strip()
    if len(s) >= 10:
        date_part = s[:8]
        hour_part = s[8:10]
        return datetime.strptime(date_part + hour_part, "%Y%m%d%H")
    return datetime.strptime(s[:8], "%Y%m%d")


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def _contains_date(donefile: str, date_str: str) -> bool:
    if not os.path.exists(donefile):
        return False
    with open(donefile, "r", encoding="utf-8") as rf:
        for line in rf:
            if date_str in line:
                return True
    return False


def _append_line(path: str, msg: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as wf:
        wf.write(msg.rstrip() + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="RankMixer test script (bin/test.sh equivalent).")
    parser.add_argument("--task", required=True, help="Task name, e.g. RankMixer_Shen0202")
    parser.add_argument("--date", required=True, help="YYYYMMDD or YYYYMMDDHH")
    parser.add_argument("--end_date", help="Optional end date, default = date")
    parser.add_argument("--config_task", help="Config task name, e.g. RankMixer_Shen0202")
    parser.add_argument("--config", help="Config module or path, e.g. config.RankMixer_Shen0202.packaged")
    parser.add_argument("--overrides_path", help="Overrides JSON/YAML file path")
    parser.add_argument("--groups_path", help="Semantic groups JSON/YAML file path")
    parser.add_argument("--no_eval", action="store_true", help="Skip eval")
    parser.add_argument("--no_clear", action="store_true", help="Skip clear_history_data")
    parser.add_argument("--main_log", help="Main log path (default: logs/main_<end_date>.log)")
    parser.add_argument("--eval_log", help="Eval log path (default: logs/<end_date>.eval)")
    return parser


def main():
    args = build_parser().parse_args()

    model_root = os.environ.get("MODEL_ROOT", "/data/share/opt/model")
    data_root = os.environ.get("DATA_ROOT", "/data/share/opt/data")
    model_dir = os.path.join(model_root, args.task)
    ckpt_dir = os.path.join(model_dir, "ckpt")
    data_path = os.path.join(data_root, args.task)

    start_dt = _parse_date(args.date)
    end_dt = _parse_date(args.end_date) if args.end_date else start_dt

    time_str = _format_dt(start_dt)
    end_time_str = _format_dt(end_dt)

    os.makedirs(os.path.join(model_dir, "logs"), exist_ok=True)
    donefile = os.path.join(model_dir, "logs", "donefile.0")
    main_log = args.main_log or os.path.join(model_dir, "logs", f"main_{end_time_str}.log")
    eval_log = args.eval_log or os.path.join(model_dir, "logs", f"{end_time_str}.eval")

    config_module = args.config
    if not config_module and args.config_task:
        config_module = f"config.{args.config_task}.packaged"

    rm = create_rankmixer(
        task=args.task,
        config=config_module,
        overrides_path=args.overrides_path,
        groups_path=args.groups_path,
    )

    # Match bin/test.sh: skip if date already in donefile.
    if not _contains_date(donefile, end_time_str[:8]):
        _append_line(main_log, f"[train] task={args.task} config={config_module} time_str={time_str} end_time_str={end_time_str}")
        rm.train(
            ckpt_dir=ckpt_dir,
            time_str=time_str,
            end_time_str=end_time_str,
            data_path=data_path,
            log_path=main_log,
        )
    else:
        _append_line(main_log, f"Skip training: {end_time_str[:8]} already in {donefile}")

    # Match bin/test.sh behavior: eval runs when end_date >= backup (20250211).
    if not args.no_eval and end_time_str[:8] >= "20250211":
        infer_delta = 24  # hours
        ckpt_time = _format_dt(end_dt - timedelta(hours=infer_delta))
        eval_ckpt_dir = os.path.join(model_dir, ckpt_time)
        _append_line(eval_log, f"[eval] task={args.task} config={config_module} time_str={end_time_str} ckpt_dir={eval_ckpt_dir}")
        rm.eval(
            ckpt_dir=eval_ckpt_dir,
            time_str=end_time_str,
            data_path=data_path,
            log_path=eval_log,
        )

    # Match bin/clear.sh basic behavior (data cleanup only).
    if not args.no_clear:
        clear_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "common", "clear_history_data.py")
        os.system(
            f"python {clear_script} --data_path {data_root} --curr_date {end_time_str[:8]}"
        )


if __name__ == "__main__":
    main()
