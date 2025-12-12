#!/usr/bin/env python
# coding=utf-8
import os
import time
import oss2
import json
import argparse
from odps import ODPS
from datetime import datetime, timedelta

from common.connect_config import *
try:
    from common.utils import train_config
except:
    from utils import train_config

def get_oss():
    """ 链接oss """
    access_key_id = os.getenv('OSS_TEST_ACCESS_KEY_ID', 'ALIYUN_ACCESS_KEY_ID_PLACEHOLDER')
    access_key_secret = os.getenv('OSS_TEST_ACCESS_KEY_SECRET', 'ALIYUN_ACCESS_KEY_SECRET_PLACEHOLDER')
    bucket_name = os.getenv('OSS_TEST_BUCKET', 'adx-oss')
    endpoint = os.getenv('OSS_TEST_ENDPOINT', 'http://oss-cn-beijing-internal.aliyuncs.com') # 内网
    # endpoint = os.getenv('OSS_TEST_ENDPOINT', 'http://oss-cn-beijing.aliyuncs.com') # 外网
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
    return bucket


def get_odps():
    """ 链接odps """
    odps_obj = ODPS(your_accesskey_id, your_accesskey_secret, your_default_project, endpoint=your_end_point,
                    tunnel_endpoint=tunnel_endpoint)
    return odps_obj

def save_eval_metric(metric_data_str,table_name,model_ver, eval_type='off_ctr'):
    parts = metric_data_str.split(', ')
    data = {}
    for part in parts:
        if 'ctcvr/auc' in part or 'loss' in part or 'ctcvr/pcoc' in part or 'time' in part:
            key_value = part.split(' = ')
            key = key_value[0].strip().replace('task1_ctcvr/', '')
            value = key_value[1].strip()
            if key in ['time', 'auc', 'loss', 'pcoc']:
                data[key] = float(value) if '.' in value else int(value)
        if "auc" not in data or "pcoc" not in data:
            if 'ctr/auc' in part or 'loss' in part or 'ctr/pcoc' in part or 'time' in part:
                key_value = part.split(' = ')
                key = key_value[0].strip().replace('task1_ctr/', '')
                value = key_value[1].strip()
                if key in ['time', 'auc', 'loss', 'pcoc']:
                    data[key] = float(value) if '.' in value else int(value)
    # 输出结果
    print(data)
    dm_date, auc, log_loss, pcoc =str(data['time'])[0:8], data['auc'], data['loss'], data['pcoc']
    cal_n, pcoc_lst, calc_time =0.00, [], ''
    records = [model_ver, auc, pcoc, log_loss, cal_n, pcoc_lst, eval_type, calc_time]
    all_t = get_odps().get_table(table_name, project="adx_dmp")
    try:
        with all_t.open_writer(partition=f"dm_date={dm_date}", create_partition=True) as w:
            w.write(records)
            time.sleep(5)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 关闭连接
        if w is not None:
            try:
                w.close()
            except Exception as e:
                print(f"Failed to close writer: {e}")

def get_metric(metric_data_str):
    data = {}
    for part in metric_data_str.split(', '):
        key, value = part.split(' = ')
        key = key.strip()
        value = value.strip()
        data[key] = float(value) if '.' in value else int(value)
    return data

def save_eval_metric_table():
    with open(args.eval_path, "r", encoding='utf-8') as f:
        line = None
        for line in f:
            continue
        print(f"line={line}")
        if line:
            eval_type = getattr(train_config, "eval_type", "ctr")
            metrics = {}
            for l in line.split(","):
                l = l.strip().split("=")
                if len(l) == 2:
                    metrics[l[0].strip()] = l[1].strip()
            records = [args.model_version, json.dumps(metrics), eval_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            all_t = get_odps().get_table(train_config.metric_table, project="adx_dmp")
            try:
                print("partition=", f"dm_date={args.dm_date}")
                with all_t.open_writer(partition=f"dm_date={args.dm_date}", create_partition=True) as w:
                    print(f"write table value({records})")
                    w.write(records)
                    time.sleep(5)
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                # 关闭连接
                if w is not None:
                    try:
                        w.close()
                    except Exception as e:
                        print(f"Failed to close writer: {e}")

def get_args():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--eval_path", default="", type=str, help="离线模型eval输入路径")
    parser.add_argument("--model_version", default=train_config.model_version, type=str, help="模型版本")
    parser.add_argument("--dm_date",
                        default=(datetime.now()-timedelta(days=1)).strftime("%Y%m%d"),
                        type=str, help="模型eval数据日期")
    args, unknown = parser.parse_known_args()
    print((args, unknown))
    return args

if __name__ == "__main__":
    # 检查是否有足够的参数
    args = get_args()
    if not args.eval_path:
        raise ValueError("eval_path is null!")
    save_eval_metric_table()
