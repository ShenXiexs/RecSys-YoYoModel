# coding=utf-8
import os
import time
import gzip
import multiprocessing
import logging
import concurrent.futures
import argparse
import psutil
import pandas as pd
from odps import ODPS
from odps.tunnel import TableTunnel
from dateutil.parser import parse

from common.connect_config import *

try:
    from common.utils import schema, slots_dict, train_config as TrainConfig, compression_type, features_sep, field_sep, \
        seq_feat_in_dataset_pad, seq_features_index, seq_features_config, seq_idxs, seq_feat_pad
except:
    from utils import schema, slots_dict, train_config as TrainConfig

multiprocessing.log_to_stderr()
logger = multiprocessing.get_logger()
logger.setLevel(logging.INFO)
slots = list(slots_dict.keys())

def get_odps():
    """Connect to ODPS."""
    odps_obj = ODPS(your_accesskey_id, your_accesskey_secret, your_default_project, endpoint=your_end_point,
                    tunnel_endpoint=tunnel_endpoint)
    return odps_obj


def features_mapper(features):
    lst = features.strip("\n").split(features_sep)
    if len(lst) != len(schema):
        return ""
    return features_sep.join([v for k, v in zip(schema, lst) if k in slots])


def new_features_mapper(features):
    lst = features.strip("\n").split(features_sep)
    # Add a column index prefix to avoid hash collisions across feature fields.
    return features_sep.join(
        [f"{col_idx}:{fea_val}" for col_idx, (fea_name, fea_val) in enumerate(zip(schema, lst)) if fea_name in slots])


def seq_features_mapper(features):
    lst = features.strip("\n").split(features_sep)
    new_lst = []
    for i, (start, end, index) in enumerate(seq_idxs):
            new_lst.extend([f"{fea_val}" if fea_val == seq_feat_pad[i] else f"{index}:{fea_val}"
                            for fea_val in lst[start:end]
                            ])
    return features_sep.join(new_lst)


def write(file, batch, data_schema):
    df = batch.to_pandas()[data_schema]
    df["features"] = df["features"].map(lambda x: new_features_mapper(x))
    if seq_features_index:
        df["seq_features"] = df["seq_features"].map(lambda x: seq_features_mapper(x))
    df.to_csv(file, mode="a", sep=field_sep, compression=compression_type.lower(), index=False, header=None)


def task(idx_date, idx, start, step,
         data_dir,
         table_name,
         partitions,
         data_schema):
    time.sleep(1)
    logger.info(f'table_name:${table_name},partition:{partitions} idx_date: {idx_date} {idx} {start} {step}')
    t0 = time.time()
    #################################################################################
    odps = get_odps()
    tunnel = TableTunnel(odps)
    #################################################################################
    download_session = tunnel.create_download_session(table_name, partition_spec=partitions)
    file = f"{data_dir}/part-r-{str(idx).zfill(2)}-{start}-{step}.gz"
    with download_session.open_arrow_reader(start, step) as reader:
        for i, batch in enumerate(reader):
            write(file, batch, data_schema)
            if i % 100 == 0:
                logger.info(f"进度: {idx_date} {start} {step} {i} {i * 1024 / step * 100:.2f}%")

    t1 = time.time()
    return f'idx_date: {idx_date} {start} {step} waste: {(t1 - t0) // 60}'


def get_args():
    save_path = os.environ.get("DATA_ROOT", "/data/share/opt/data")
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--start_date", default="", type=str, help="数据下载开始日期")
    parser.add_argument("--end_date", default="", type=str, help="数据下载结束日期")
    parser.add_argument("--save_path", default=save_path, type=str, help="数据下载存储路径")
    parser.add_argument("--max_workers", default=16, type=int, help="数据下载进程数")
    parser.add_argument("--durations", default="1", type=str, help="数据durations分区")
    parser.add_argument("--task", default="", type=str, help="任务名")
    args, unknown = parser.parse_known_args()
    print((args, unknown))
    return args


def wait_for_process_exit(
        target_script: str = "downodps.py",
        target_task: str = "task O35_mutil_cvr",
        check_interval: int = 60,  # Check interval (seconds)
        max_wait_seconds: int = 24 * 3600,  # Max wait time (24 hours)
        **kwargs
) -> bool:
    """
    Wait for target process to exit: check for processes containing the script and task name;
    keep waiting if found, return False on timeout.

    Args:
        target_script: target script name (e.g., "downodps.py")
        target_task: target task identifier (e.g., "task O35_mutil_cvr")
        check_interval: check interval (seconds)
        max_wait_seconds: max wait time (seconds)

    Returns:
        bool: True if process exits; False if timeout
    """
    start_time = time.time()  # Record start time
    current_pid = os.getpid()  # Get current Python PID
    while True:
        # Compute elapsed wait time
        elapsed_time = time.time() - start_time
        if elapsed_time > max_wait_seconds:
            print(f"⚠️  等待超时（已超过 {max_wait_seconds / 3600:.1f} 小时），目标进程仍在运行，退出等待")
            return False
        # Track whether target process is found
        process_found = False
        # Iterate processes and check cmdline
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                # Get process cmdline list (filter invalid processes)
                cmdline = proc.info.get("cmdline", [])
                # Convert cmdline list to string (for matching)
                try:
                    cmd_str = " ".join(cmdline)
                except Exception as e:
                    print(f" e:{e} , cmdline={cmdline} ")
                    continue
                #
                proc_pid = proc.info["pid"]
                # Skip current process to avoid self-match
                if proc_pid == current_pid:
                    print(f"cmd={cmd_str}, pid={proc_pid}, current_pid={current_pid}")
                    continue
                if not cmdline:
                    continue  # Skip processes without cmdline (e.g., system processes)
                # Check whether cmdline contains both script name and task identifier (same logic as ps)
                if target_script in cmd_str and target_task in cmd_str:
                    flag = True
                    for v in kwargs.values():
                        flag = v in cmd_str and flag
                    if flag:
                        print(f"✅ 找到目标进程（PID: {proc_pid}），继续等待... "
                              f"（已等待 {elapsed_time / 3600:.1f} 小时，"
                              f"剩余 {max(0, max_wait_seconds - elapsed_time) / 3600:.1f} 小时）")
                        process_found = True
                        break  # Found one; no need to continue
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Ignore processes without access or already exited
                continue
        # If target process not found, it has exited; return True
        if not process_found:
            print(f"🎉 目标进程已退出，结束等待（总等待时间：{elapsed_time / 60:.1f} 分钟）")
            return True
        # If found, sleep for interval then recheck
        time.sleep(check_interval)


if __name__ == '__main__':
    args = get_args()
    end_date = args.start_date
    if args.end_date and parse(args.end_date) > parse(args.start_date):
        end_date = parse(args.end_date).strftime('%Y%m%d')
    try:
        dates = pd.date_range(args.start_date, end_date).map(lambda x: x.strftime("%Y%m%d")).tolist()
    except:
        dates = TrainConfig.downodps_datas
    # Define model branch and data download storage path
    model_ver = getattr(TrainConfig, "data_nm", TrainConfig.model_version)
    data_root_dir = f'{args.save_path}/{model_ver}'
    feature_data_table_name = TrainConfig.binning_table_name
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers)
    data_schema = TrainConfig.data_schema
    for day in dates:
        ###############add by huangmian##############
        wait_for_process_exit(
            target_script="downodps.py",
            target_task=f"task {TrainConfig.data_nm}",
            check_interval=60,  # Check interval (seconds)
            max_wait_seconds=20 * 3600,  # Max wait time (hours)
            start=f"--end_date {day}"
        )
        ##############################################
        basepath = f"{data_root_dir}/{day}"
        success = f"{basepath}/_SUCCESS"
        if os.path.exists(success):
            logger.info(f"success file exits: {success}")
            continue
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        t0 = time.time()
        odps = get_odps()
        tunnel = TableTunnel(odps)
        table = odps.get_table(feature_data_table_name)
        #
        # partitions = f"idx_date={day},feature_version={model_ver},durations={args.durations}"
        partitions = TrainConfig.partitions.format(day=day)
        for i in range(1440):
            if table.exist_partition(partitions):
                break
            if i % 10 == 0:
                logger.info(f"table:{feature_data_table_name} partitions not exits: {partitions}")
            time.sleep(60)

        download_session = tunnel.create_download_session(feature_data_table_name, partition_spec=partitions)
        step = download_session.count // max(os.cpu_count() - 1, 1)
        start = 0
        idx = 0

        futures = []
        while start < download_session.count:
            cnt = min(download_session.count - start, step)
            future = executor.submit(task, day, idx, start, cnt, basepath,
                                     feature_data_table_name, partitions, data_schema)
            start += step
            idx += 1

            futures.append(future)

        for future in futures:
            result = future.result()
            logger.info(f"result: {result}")

        t1 = time.time()
        f = gzip.open(success, "w")
        f.write(f"{(t1 - t0) / 60:.2f}".encode())
        f.close()

    executor.shutdown()

