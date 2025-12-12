# -*- coding: utf-8 -*-
# @Time : 2025/10/29 15:47
# @Author : huangmian
# @File : run_datawork_check_multi_cvr_o35_push_oss

'''PyODPS 3
请确保不要使用从 MaxCompute下载数据来处理。下载数据操作常包括Table/Instance的open_reader以及 DataFrame的to_pandas方法。
推荐使用 PyODPS DataFrame（从 MaxCompute 表创建）和MaxCompute SQL来处理数据。
更详细的内容可以参考：https://help.aliyun.com/document_detail/90481.html
'''
# utf-8
"""
    主要在datawork中check oss 的模型是否生成，
        第一步先将特征推入到redis中
        sleep 10 分钟
        第二步再将模型推入到指定的目录
"""

import os
import sys
import time
import json
import oss2
import logging
from tqdm import tqdm
from odps import ODPS
from redis.client import Redis

try:
    from common.connect_config import (
        your_accesskey_id,
        your_accesskey_secret,
        your_default_project,
        tunnel_endpoint,
        your_end_point,
        redis_username,
        redis_password,
        redis_host,
        redis_port,
        redis_feature_username,
        redis_feature_password,
        redis_feature_host,
        redis_feature_port,
    )
except ImportError:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    from common.connect_config import (
        your_accesskey_id,
        your_accesskey_secret,
        your_default_project,
        tunnel_endpoint,
        your_end_point,
        redis_username,
        redis_password,
        redis_host,
        redis_port,
        redis_feature_username,
        redis_feature_password,
        redis_feature_host,
        redis_feature_port,
    )

model_date = args["model_date"]
model_hour = args["model_hour"]
print(f"---> model_date: {model_date} <--- ")
#定义模型分支号
model_version = "O35"
metric_model_version = "O35_mutil_cvr_v7"

def logger():
    """ 日志输出 """
    log = logging.getLogger(name="run_datawork_check_model_push_oss_honor")
    log.setLevel(level=logging.INFO)
    if not log.handlers:
        log_out = logging.StreamHandler()
        fmt = "%(asctime)s - [%(funcName)s-->line:%(lineno)d] - %(levelname)s:%(message)s"
        log_fmt = logging.Formatter(fmt=fmt)
        log_out.setFormatter(fmt=log_fmt)
        log.addHandler(log_out)
    return log


def get_oss():
    """ 链接oss """
    access_key_id = os.getenv('OSS_TEST_ACCESS_KEY_ID', '')
    access_key_secret = os.getenv('OSS_TEST_ACCESS_KEY_SECRET', '')
    bucket_name = os.getenv('OSS_TEST_BUCKET', 'adx-oss')
    endpoint = os.getenv('OSS_TEST_ENDPOINT', 'http://oss-cn-beijing-internal.aliyuncs.com')  # 内网
    # endpoint = os.getenv('OSS_TEST_ENDPOINT', 'http://oss-cn-beijing.aliyuncs.com') # 外网
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
    return bucket


def _env_or_config(name, fallback):
    """Resolve secret from environment with config fallback."""
    return os.getenv(name, fallback if fallback is not None else "")


def get_redis():
    """ 链接redis """
    host = _env_or_config("YOYO_REDIS_HOST", redis_host)
    port = int(os.getenv("YOYO_REDIS_PORT", str(redis_port)))
    user = _env_or_config("YOYO_REDIS_USERNAME", redis_username)
    pwd = _env_or_config("YOYO_REDIS_PASSWORD", redis_password)
    if not host or not pwd:
        raise EnvironmentError("缺少Redis连接配置，请设置 YOYO_REDIS_HOST/YOYO_REDIS_PASSWORD 环境变量。")
    return Redis(host=host, port=port, username=user or None, password=pwd, db=1)


def get_redis_feature():
    """ 链接redis """
    host = _env_or_config("YOYO_REDIS_FEATURE_HOST", redis_feature_host)
    port = int(os.getenv("YOYO_REDIS_FEATURE_PORT", str(redis_feature_port)))
    user = _env_or_config("YOYO_REDIS_FEATURE_USERNAME", redis_feature_username)
    pwd = _env_or_config("YOYO_REDIS_FEATURE_PASSWORD", redis_feature_password)
    if not host or not pwd:
        raise EnvironmentError("缺少特征Redis连接配置，请设置 YOYO_REDIS_FEATURE_HOST/YOYO_REDIS_FEATURE_PASSWORD 环境变量。")
    return Redis(host=host, port=port, username=user or None, password=pwd, db=0)


def oss_sync_file(org_file, target_file):
    logger().info(f"org_file: {org_file} --> target_file: {target_file} --> start ")
    get_oss().copy_object(source_bucket_name="adx-oss", source_key=org_file, target_key=target_file)
    logger().info(f"org_file: {org_file} --> target_file: {target_file} --> finish ")


def file_exists(offline_model_path, threshold_cnt):
    logger().info(f"offline_model_path: {offline_model_path}")
    cnt = 0
    while True:
        is_success = get_oss().object_exists(offline_model_path)
        if is_success is False:
            logger().warning(f"offline_model_path: {offline_model_path} is not exist, sleep 2m ...")
            logger().warning(f"已经等待: {cnt} 次，curr_time: {time.strftime('%Y%m%d%H%M')}")
            time.sleep(120)
        else:
            logger().info(f"offline_model_path: {offline_model_path} is exist.. model train success...")
            break
        cnt += 1
        if cnt >= threshold_cnt:
            raise TimeoutError("Waiting 12 hour, time out!, model train fail...")


def read_file(oss_target_file):
    """ 读取文件传过来的是 SUCCESS、FAIL"""
    logger().info(f"==> oss_target_file: {oss_target_file}")
    result_tag = str(get_oss().get_object(oss_target_file).read())
    logger().info(f"==> result_tag: {result_tag}")
    if "SUCCESS" in result_tag:
        return True
    else:
        return False


def get_odps():
    """ 链接odps """
    access_key_id = _env_or_config("ALIYUN_ACCESS_KEY_ID", your_accesskey_id)
    access_key_secret = _env_or_config("ALIYUN_ACCESS_KEY_SECRET", your_accesskey_secret)
    default_project = _env_or_config("ALIYUN_DEFAULT_PROJECT", your_default_project)
    maxcompute_endpoint = _env_or_config("ALIYUN_ENDPOINT", your_end_point)
    tunnel = _env_or_config("ALIYUN_TUNNEL_ENDPOINT", tunnel_endpoint)
    if not access_key_id or not access_key_secret:
        raise EnvironmentError("缺少 MaxCompute AccessKey，请设置 ALIYUN_ACCESS_KEY_ID/ALIYUN_ACCESS_KEY_SECRET。")
    return ODPS(
        access_key_id,
        access_key_secret,
        default_project,
        endpoint=maxcompute_endpoint,
        tunnel_endpoint=tunnel,
    )


def odps2redis(table_nm, partition, key_col, key_prefix="", feats=[], default="-1024", sep=",", rn=0):
    if rn >= 6:
        raise ValueError(
            "===table_nm=%s partition=%s to redis error, 重试次数%d, 任务异常结束===" % (table_nm, partition, rn))
    logger().info("========table_nm=%s partition=%s to redis==========" % (table_nm, partition))
    try:
        feat_rc = get_redis_feature()
        # push pkg_ctr_binning to redis
        data_t = get_odps().get_table(table_nm)
        reader = data_t.open_reader(partition=partition)
        for row in tqdm(reader):
            redis_key = f'{key_prefix}{row[key_col]}'
            redis_value = []
            for feat in feats:
                redis_value.append(row[feat] or default)
            # print(redis_key, f"{sep}".join(redis_value))
            feat_rc.set(redis_key, f"{sep}".join(redis_value))
    except Exception as e:
        logger().error("===table_nm=%s partition=%s to redis error, 重试次数%d===" % (table_nm, partition, rn + 1))
        time.sleep(600)
        odps2redis(table_nm, partition, key_col, key_prefix, feats, default, sep, rn + 1)
    logger().info("=" * 30)


def get_measure_result(table_name, model_ver):
    """ 获取模型的测试集数据情况 """
    start_time = time.time()
    max_wait_time=10800
    while True:
        data_t = get_odps().get_table(table_name)
        try:
            reader = data_t.open_reader(partition=f'dm_date={model_date}0000')
            model_version_data = {}
            cnt = 0
            while True:
                for row in reader:
                    model_version_data[row["model_version"]]=json.loads(row["eval_metric"])
                is_exist = 'exist' if model_ver in model_version_data else 'not_exist'
                if is_exist == 'not_exist':
                    logger().warning(f"{model_version}模型eval指标未同步odps表, sleep 5m ...")
                    logger().warning(f"已经等待: {cnt} 次，curr_time: {time.strftime('%Y%m%d%H%M')}")
                    time.sleep(300)
                else:
                    logger().info(f"{model_version}模型eval指标已同步odps表")
                    break
                cnt += 1
            return model_version_data[model_ver]
            # 如果分区存在，退出循环
            break
        except Exception as e:
            # 如果分区不存在，捕获异常并等待
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= max_wait_time:
                raise TimeoutError(f"Partition dm_date={model_date} did not appear within {max_wait_time} seconds.")
            else:
                time.sleep(300)  # 等待一段时间再检查
                continue


def check_eval_metric_isok():
    bucket = get_oss()
    oss_offline_base = f"deep_model/offline/{model_version}"
    feature_file_off_p = os.path.join(oss_offline_base, model_date, "_FEATURE_SUCCESS")
    bucket.put_object(feature_file_off_p, "SUCCESS")
    # 离线验证集上的指标数据
    table_name = "adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm"
    # 验证集指标达标阈值才会推送模型
    metric_threshold_dict = {
        "awake/auc": (0.7, 0.9)
    }
    #
    logger().info("--" * 20)
    logger().info(f"---model_version : {model_version}---")
    logger().info(f"---metric_model_version : {metric_model_version}---")
    logger().info(f"---model_date : {model_date}---")
    logger().info(f"---table_name : {table_name}---")
    # 获取测试指标结果
    model_eval_metric_dict = get_measure_result(table_name, metric_model_version)
    for metric_nm, metric_threshold in metric_threshold_dict.items():
        min_threshold, max_threshold = metric_threshold
        logger().info(f"---metric_nm : {metric_threshold}---")
        if metric_nm in model_eval_metric_dict:
            if min_threshold <= model_eval_metric_dict[metric_nm] <= max_threshold:
                logger().info(f"---{metric_nm}={model_eval_metric_dict[metric_nm]}, measure result is OK---")
            else:
                raise ValueError(f"指标不符合上线阈值, "
                                 f"{metric_nm}={model_eval_metric_dict[metric_nm]}不在阈值范围内: {metric_threshold}"
                                )
        else:
            raise ValueError(f"---{metric_nm} not in {list(model_eval_metric_dict.keys())}---")


def check_features_isready():
    feature_file = "_FEATURE_SUCCESS"
    oss_offline_base = f"deep_model/offline/{model_version}"
    oss_config_dir = f"deep_model/config_v3/{model_version}"
    oss_online_base = f"deep_model/online/{model_version}"
    threshold_cnt = 360  # 12 小时
    redis_key = f"model_sync_ts_v2_{model_version}"

    # 判断模型是否准备好
    feature_file_off_p = os.path.join(oss_offline_base, model_date, feature_file)
    logger().info(f"feature_file_off_p: {feature_file_off_p}")
    cnt = 0
    while True:
        is_success = read_file(feature_file_off_p)
        if is_success is False:
            logger().warning(
                f"feature_file_off_p: {feature_file_off_p} content SUCCESS flag  is not exist, sleep 2m ...")
            logger().warning(f"已经等待: {cnt} 次，curr_time: {time.strftime('%Y%m%d%H%M')}")
            time.sleep(120)
        else:
            logger().info(
                f"feature_file_off_p: {feature_file_off_p}  content SUCCESS flag is exist.. model train success...")
            break
        cnt += 1
        if cnt >= threshold_cnt:
            raise TimeoutError("Waiting 12 hour, time out!, model train fail...")

    curr_model_time = time.strftime('%Y%m%d%H%M')
    logger().info(f"===> curr_model_time: {curr_model_time}")
    # deep_model/offline/{model_version}/目录下特征文件
    feature_cp_files = [
        "doc__key_one.csv",
        "doc__key_two.csv",
        "doc__key_three.csv",
        "doc__key_four.csv",
        "doc__key_five.csv",
        "doc__key_six.csv",
        "doc__key_seven.csv",
    ]
    logger().info(f" ----- run datawork {model_version} model params  ----- ")
    for file_nm in feature_cp_files:
        # 判断批量特征是否生成
        oss_offline_path = os.path.join(oss_offline_base, model_date, file_nm)
        oss_online_path = os.path.join(oss_online_base, curr_model_time, file_nm)
        logger().info(f"--oss_offline_path={oss_offline_path}, oss_online_path={oss_online_path}--")
        file_exists(oss_offline_path, threshold_cnt)
        # 离线目录->在线目录
        oss_sync_file(oss_offline_path, oss_online_path)

    # 离线features推送到redis
    # odps2redis("vivo_bmd_app_pkg_whitelist_ctr_binning_statistics_hi",
    #            f'ds_date={model_date},ds_hour={model_hour}',
    #            key_col="ctr_binning",
    #            key_prefix="pkg_ctr_binning_",
    #            feats=["ctr_binning", "imp", "clk", "ctr_rate", "activate",
    #                   "activate_rate", "new_alive", "new_alive_rate"],
    #            default="-1024",
    #            sep=",")

    logger().info("")
    # config_v3/{model_version}/目录的待cp的文件目录
    config_cp_files = [
        "fg.json"
        # ,"model_params.json"
        , "doc__key_five_feature_mapping.txt"
        , "doc__key_four_feature_mapping.txt"
        , "doc__key_one_feature_mapping.txt"
        , "doc__key_seven_feature_mapping.txt"
        , "doc__key_six_feature_mapping.txt"
        , "doc__key_three_feature_mapping.txt "
        , "doc__key_two_feature_mapping.txt"
    ]
    for file_nm in config_cp_files:
        config_off_p = os.path.join(oss_config_dir, file_nm)
        config_on_p = os.path.join(oss_online_base, curr_model_time, file_nm)
        logger().info(f"--config_off_p={config_off_p}, config_on_p={config_on_p}--")

    # 链接redis
    feat_rc = get_redis_feature()
    # push version to redis
    feat_rc.set(redis_key, str(curr_model_time))
    logger().info(f"==> push redis key: {redis_key}, value: {feat_rc.get(redis_key)}")
    logger().info("--------- check finish ----------")


if __name__ == '__main__':
    check_eval_metric_isok()
    check_features_isready()
