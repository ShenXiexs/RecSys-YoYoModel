# -*- coding: utf-8 -*-
# @Time : 2025/10/30 9:35
# @Author : huangmian
# @File : clear_history_data.py
import os
import shutil
import argparse
from datetime import datetime, timedelta
from common.utils import train_config as TrainConfig


def _normalize_date_str(date_str):
    return str(date_str).strip().replace("-", "")


def _in_keep_ranges(del_date, keep_ranges):
    for item in keep_ranges:
        if isinstance(item, str):
            if "-" not in item:
                continue
            start, end = item.split("-", 1)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            continue
        start = _normalize_date_str(start)
        end = _normalize_date_str(end)
        if len(start) == 4 and len(end) == 4:
            # Support MMDD ranges across years, e.g. 1101-1110.
            md = del_date[4:8]
            if start <= md <= end:
                return True
        elif start <= del_date <= end:
            return True
    return False


def _should_skip_delete(del_date):
    keep_dates = set(getattr(TrainConfig, "keep_dates", []))
    if del_date in keep_dates:
        return True
    keep_ranges = getattr(TrainConfig, "keep_date_ranges", [])
    if keep_ranges and _in_keep_ranges(del_date, keep_ranges):
        return True
    return False


def get_args():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--data_path", type=str, help="数据路径", required=True)
    parser.add_argument("--curr_date", type=str, help="curr_date", required=True)
    parser.add_argument("--del_date", type=str, help="删除N天前的数据", default=None)
    args, unknown = parser.parse_known_args()
    print((args, unknown))
    return args


if __name__ == '__main__':
    args = get_args()
    data_nm = getattr(TrainConfig, "data_nm", TrainConfig.model_version)
    del_days = getattr(TrainConfig, "del_days", 30)
    del_date = args.del_date
    if not del_date:
        curr_date = datetime.strptime(args.curr_date, '%Y%m%d')
        del_date = (curr_date - timedelta(days=del_days)).strftime('%Y%m%d')
    if _should_skip_delete(del_date):
        print(f"---------skip delete for {del_date}--------")
        raise SystemExit(0)
    print(f"---------del_path={args.data_path}/{data_nm}/{del_date}--------")
    del_path = os.path.join(args.data_path, data_nm, del_date)
    if os.path.exists(del_path):
        shutil.rmtree(del_path)

