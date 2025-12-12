#!/usr/bin/env python
# coding=utf-8
import oss2, os
import time, datetime
import logging
from absl import flags,app

from common.aliyun import get_oss
from common.connect_config import *
try:
    from common.utils import train_config
except:
    from utils import train_config

flags.DEFINE_string('model_ver',getattr(train_config, "oss_offline_model_ver", train_config.model_version),'模型版本')
flags.DEFINE_string('oss_offline_root_path',train_config.oss_offline_root_path,'模型离线特征存储路径')
flags.DEFINE_string('upload_oss_path',train_config.upload_oss_path,'模型导出OSS路径')
flags.DEFINE_string('oss_bucket_name',train_config.oss_bucket_name,'oss bucket')
FLAGS = flags.FLAGS


dirname = os.path.dirname(os.path.abspath(__file__))
basename = train_config.model_version

auth = oss2.Auth(upload_oss_access_key_id, upload_oss_access_key_secret)
bucket = oss2.Bucket(auth, upload_oss_endpoint, upload_oss_bucket_name)


def parse_int(x):
    try:
        return int(x)
    except:
        return -1


def upload_folder(folder_name, target_folder=None):
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            print('file_path:',file_path)
            # 构造OSS上的路径
            if target_folder:
                target_path = os.path.join(target_folder, file_path.lstrip(os.path.sep))
            else:
                target_path = file_path.lstrip(os.path.sep)
            # 上传文件
            with open(file_path, 'rb') as fileobj:
                logger().info(f'put file:{file_path} to oss target_path:{target_path}')
                bucket.put_object(target_path, fileobj)


def logger():
    """ 日志输出 """
    log = logging.getLogger(name="run_datawork_C33_model_feat")
    log.setLevel(level=logging.INFO)
    if not log.handlers:
        log_out = logging.StreamHandler()
        fmt = "%(asctime)s - [%(funcName)s-->line:%(lineno)d] - %(levelname)s:%(message)s"
        log_fmt = logging.Formatter(fmt=fmt)
        log_out.setFormatter(fmt=log_fmt)
        log.addHandler(log_out)
    return log


def feat_file_exists():
    model_date = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime('%Y%m%d')
    oss_offline_base = f"{FLAGS.oss_offline_root_path}/{FLAGS.model_ver}"
    feat_file = "_FEATURE_SUCCESS"
    offline_feat_path = os.path.join(oss_offline_base, model_date, feat_file)
    logger().info(f"offline_feat_path: {offline_feat_path}")
    threshold_cnt = 360  # 12 小时

    cnt = 0
    while True:
        is_success = get_oss(FLAGS.oss_bucket_name).object_exists(offline_feat_path)
        if is_success is False:
            logger().warning(f"offline_feat_path: {offline_feat_path} is not exist, sleep 2m ...")
            logger().warning(f"已经等待: {cnt} 次，curr_time: {time.strftime('%Y%m%d%H%M')}")
            time.sleep(120)
        else:
            logger().info(f"offline_feat_path: {offline_feat_path} is exist.. model train success...")
            get_oss().put_object(offline_feat_path, "SUCCESS")
            logger().info(ff"put to {offline_feat_path}")
            break
        cnt += 1
        if cnt >= threshold_cnt:
            raise TimeoutError("Waiting 12 hour, time out!, model train fail...")

def main(argv):
    feat_file_exists()
    upload_oss_path = FLAGS.upload_oss_path
    model_root = os.environ.get("MODEL_ROOT", "/data/share/opt/model")
    os.chdir(f"{model_root}/{basename}/export_dir")
    while True:
        files = sorted(map(parse_int, os.listdir(f"{model_root}/{basename}/export_dir")))
        logger().info(f"files: {files},", files[-1], os.path.exists(f"{files[-1]}"))
        if os.path.exists(f"{files[-1]}") and \
           os.path.exists(f"{files[-1]}/assets.extra") and \
           not bucket.object_exists(f"{upload_oss_path}/{basename}/{files[-1]}/saved_model.pb"):
            logger().info(f"upload_folder: {files[-1]}")
            upload_folder(f"{files[-1]}", f"{upload_oss_path}/{basename}")
            break
        time.sleep(60)

if __name__ == "__main__":
    app.run(main)


