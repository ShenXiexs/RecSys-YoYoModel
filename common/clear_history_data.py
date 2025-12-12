# -*- coding: utf-8 -*-
# @Time : 2025/10/30 9:35
# @Author : huangmian
# @File : clear_history_data.py
import os
import shutil
import argparse
from datetime import datetime, timedelta
from common.utils import train_config as TrainConfig


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
    print(f"---------del_path={args.data_path}/{data_nm}/{del_date}--------")
    del_path = os.path.join(args.data_path, data_nm, del_date)
    if os.path.exists(del_path):
        shutil.rmtree(del_path)

