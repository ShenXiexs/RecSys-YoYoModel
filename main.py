# -*- coding: utf-8 -*-
# @Time : 2025/8/25 18:19
# @Author : huangmian
# @File : main_gpu.py
import os
import re
import time
import glob
import json
import logging
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
from tqdm import tqdm
from absl import app, flags
from importlib import import_module
from dateutil.parser import parse
from collections import Counter
from datetime import datetime

from common.utils import write_donefile, serving_input_receiver_dense_fn, tf_config
from common.odps_util import *
from common.aliyun import get_oss
from common.utils import train_config as TrainConfig

data_root = os.environ.get("DATA_ROOT", "/data/share/opt/data")
flags.DEFINE_string('job_name', "", 'job_name')
flags.DEFINE_string('ckpt_dir', "./ckpt", 'ckpt_dir')
flags.DEFINE_string('export_dir', "./export_dir", 'export_dir')
flags.DEFINE_string('mode', "train", 'train or export')
flags.DEFINE_string('data_path', f'{data_root}/{TrainConfig.model_version}', 'data path')
flags.DEFINE_string('time_str', '202403012359', 'training time str')
flags.DEFINE_string('end_time_str', None, 'training time str')
flags.DEFINE_string('file_list', '', 'file list')
flags.DEFINE_string('slot', "", 'miss slot')
FLAGS = flags.FLAGS

dirname = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger('tensorflow')
logger.propagate = False
logger = tf.compat.v1.logging


def set_global_seed(seed=107):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def get_session_config():
    """获取会话配置"""
    strategy, gpu_ids = None, []
    # CPU线程配置
    if getattr(TrainConfig, "device", "CPU").lower() == "gpu" and FLAGS.mode != "export":
        # GPU模式
        # 配置所有GPU的内存增长
        physical_devices = tf.config.list_physical_devices('GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(len(physical_devices))))
        if physical_devices:
            for device in physical_devices:
                try:
                    if getattr(TrainConfig, 'gpu_memory_growth', True):
                        tf.config.experimental.set_memory_growth(device, True)
                    # 设置内存限制（如果指定）
                    if getattr(TrainConfig, 'gpu_memory_limit', 0) > 0:
                        tf.config.set_logical_device_configuration(
                            device,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=getattr(TrainConfig, 'gpu_memory_limit', 0))]
                        )
                except RuntimeError as e:
                    logger.error(f"GPU {device} configuration error: {e}")
        #
        gpu_ids = getattr(TrainConfig, 'gpu_list', '').split(',')
        if len(gpu_ids) > 1:
            # 创建MirroredStrategy用于多GPU训练
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Using multi-GPU with devices: {getattr(TrainConfig, 'gpu_list', '')}")
            logger.info(f"Number of devices: {strategy.num_replicas_in_sync}")
        # GPU模式下的线程配置
        config = tf.compat.v1.ConfigProto(
            device_count={'GPU': len(physical_devices)},
            allow_soft_placement=True,
            log_device_placement=False
        )
        config.gpu_options.visible_device_list = ",".join(gpu_ids)
        config.inter_op_parallelism_threads = 8
        config.intra_op_parallelism_threads = 8
        # GPU内存配置
        if getattr(TrainConfig, 'gpu_memory_growth', True):
            config.gpu_options.allow_growth = True
        # 设置GPU内存分配比例
        if getattr(TrainConfig, 'gpu_memory_limit', 0) == 0:
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
    else:
        config = tf.compat.v1.ConfigProto(
            device_count={'GPU': 0},
            allow_soft_placement=False,
            log_device_placement=False
        )
        config.inter_op_parallelism_threads = os.cpu_count() // 2
        config.intra_op_parallelism_threads = os.cpu_count() // 2
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config.allow_soft_placement = False
        logger.info("Using CPU for training")
    return config, strategy, gpu_ids


def train(filenames, params, model_config, steps=None, strategy=None):
    # ==========  执行任务  ========== #
    model_dir = os.path.dirname(FLAGS.ckpt_dir)
    #
    model_fn_path, input_fn_path = TrainConfig.model_modul, TrainConfig.dataset_modul
    # 解析模型函数
    model_fn_modul, _, model_fn_str = model_fn_path.rpartition('.')
    model_fn = getattr(import_module(model_fn_modul), model_fn_str)
    # 解析数据函数
    input_fn_modul, _, input_fn_str = input_fn_path.rpartition('.')
    input_fn = getattr(import_module(input_fn_modul), input_fn_str)

    # 根据设备类型调整batch size
    train_spec_config = TrainConfig.inp_fn_config.get("train_spec", {})
    eval_spec_config = TrainConfig.inp_fn_config.get("eval_spec", {})
    train_epoch = TrainConfig.inp_fn_config.get("train_epoch", 1)
    train_batch_size = TrainConfig.inp_fn_config.get("train_batch_size", 1024)
    batch_size = TrainConfig.inp_fn_config.get("batch_size", 1024)
    #
    if getattr(TrainConfig, "device", "CPU") == "gpu" and strategy \
            and len(getattr(TrainConfig, 'gpu_list', '').split(",")) > 1:
        # 多GPU模式下，每个GPU使用原始batch size
        train_batch_size = train_batch_size * strategy.num_replicas_in_sync
        batch_size = batch_size * strategy.num_replicas_in_sync
        logger.info(f"Adjusted batch size for multi-GPU: train={train_batch_size}, eval={batch_size}")

    # 创建Estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.ckpt_dir,
        params=params,
        config=model_config
    )

    logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logger.info(f"Device configuration: {getattr(TrainConfig, 'device', 'CPU')}")
    if params.get("device", "CPU").upper() == "GPU":
        logger.info(f"Using GPUs: {getattr(TrainConfig, 'gpu_list', '')}")
    logger.info(f"Batch sizes - Train: {train_batch_size}, Eval: {batch_size}")
    logger.info(TrainConfig.inp_fn_config)
    ###
    if FLAGS.mode == "train":
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(filenames, model_dir, task_number, task_idx,
                                      train_epoch, batch_size=train_batch_size),
            **train_spec_config
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn([f"{os.path.dirname(filenames[0])}/_SUCCESS"], model_dir, task_number, task_idx,
                                      batch_size=batch_size),
            **eval_spec_config
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    elif FLAGS.mode in ['eval', 'feature_eval']:
        if os.path.exists(FLAGS.ckpt_dir):
            estimator.evaluate(input_fn=lambda: input_fn(filenames, model_dir, task_number, task_idx, shuffle=False,
                                                         batch_size=batch_size))
        else:
            logger.warning(f"{FLAGS.ckpt_dir} is not exists, not run evaluate")

    elif FLAGS.mode == "infer":
        if not os.path.exists(FLAGS.ckpt_dir):
            logger.info(f"---{FLAGS.ckpt_dir} is not exists, infer not run---")
            return
        f_idx = 0
        table_name = TrainConfig.infer_table_name
        dt = FLAGS.end_time_str or FLAGS.time_str
        partitions = f"ds_date={dt},feature_type={TrainConfig.model_version},infer_time={datetime.now().strftime('%Y%m%d%H%M%S')}"
        for f in filenames:
            file_path = f"{model_dir}/logs/pred_{dt}_{f_idx}.csv"
            print(f'----predict file:{file_path},data_file:{f}------')
            with open(file_path, "w") as wf:
                for x in tqdm(estimator.predict(input_fn=lambda: input_fn([f], model_dir, batch_size=batch_size))):
                    # lst = [x["requestid"].decode("utf8")] + [x["combination_un_id"].decode("utf-8")] + list(
                    #     map(str, x["out"]))
                    lst = [x["requestid"].decode("utf8")] + [x["combination_un_id"].decode("utf-8")] + \
                          [json.dumps(dict(zip(getattr(TrainConfig, "predict_columns",
                                                       map(lambda x: f"c_{x}", range(len(x["out"])))),
                                               map(str, x["out"]))))]
                    wf.write("\t".join(lst) + "\n")
            df: DataFrame = pd.read_csv(file_path, delimiter='\t', header=None)
            write_df2odps(df, table_name, partitions)
            os.remove(file_path)
            f_idx += 1
        pred_flag_file = f"deep_model/offline/model_pred_flag/{dt}/{TrainConfig.model_version}.pred_flag"
        get_oss().put_object(pred_flag_file, "SUCCESS")

    elif FLAGS.mode == "dump":
        logger.info(f"lx: {estimator.get_variable_names()}")
        keys = estimator.get_variable_value("embeddings/embeddings_mht_1of1-keys")
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        emb = np.concatenate((np.reshape(keys, [-1, 1]), values), axis=1)
        np.savetxt(f"{FLAGS.ckpt_dir}/emb.txt", emb, fmt=['%d'] + ['%.16f'] * 9)
    elif FLAGS.mode == "preview":
        logger.info(f"lx: {estimator.get_variable_names()}")
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        np.savetxt(f"{FLAGS.ckpt_dir}/var.txt", values, fmt='%.16f')
    elif FLAGS.mode == "export" and task_type == "chief" and int(task_idx) == 0:
        tfra.dynamic_embedding.enable_inference_mode()
        estimator.export_saved_model(FLAGS.export_dir, lambda: serving_input_receiver_dense_fn())

    elif FLAGS.mode == "emb":
        outs = estimator.predict(input_fn=lambda: input_fn(filenames, model_dir, batch_size=batch_size))
        with open(f"{model_dir}/emb.txt", "w") as wf:
            for out in tqdm(outs):
                sign = out["sign"]
                value = out["value"].decode()
                emb = "".join(map(lambda x: f"{x:.6f}", out["emb"]))
                wf.write(f"{sign}\t{value}\t{emb}\n")


def main(argv=None):
    set_global_seed(107)
    model_dir = os.path.dirname(FLAGS.ckpt_dir)
    data_nm = getattr(TrainConfig, "data_nm", os.path.basename(FLAGS.data_path))
    data_path = os.path.join(os.path.dirname(FLAGS.data_path), data_nm)
    logger.info(f">>>>>>>>>>>>>>>data_path={data_path}<<<<<<<<<<<<<<<<<<")
    t0 = time.time()
    # 创建RunConfig
    session_config, strategy, gpu_ids = get_session_config()
    model_config = tf.estimator.RunConfig().replace(
        session_config=session_config,
        **TrainConfig.es_run_config
    )
    if gpu_ids and len(gpu_ids) > 1:
        # 多GPU模式使用分布式策略
        model_config = model_config.replace(
            train_distribute=strategy,
            eval_distribute=strategy
        )
    #参数配置
    params = {
        "mode": FLAGS.mode,
        "ps_num": len(gpu_ids) if gpu_ids else ps_num,
        "task_number": task_number,
        "task_type": task_type,
        "task_idx": task_idx,
        "slot": FLAGS.slot,
        "restrict": False,
        "device": "GPU" if gpu_ids else "CPU",  # 添加设备信息到params
        "gpu_ids": gpu_ids
    }
    # 添加用户指定的模型fn所需的参数
    params.update(TrainConfig.train_params)
    if len(FLAGS.file_list) == 0:
        time_str = parse(FLAGS.time_str).strftime('%Y%m%d')
        end_time_str = time_str
        if FLAGS.end_time_str and parse(FLAGS.end_time_str) > parse(FLAGS.time_str):
            end_time_str = parse(FLAGS.end_time_str).strftime('%Y%m%d')
        logger.info(f"time_str={time_str}, end_time_str={end_time_str}")
        dates = pd.date_range(time_str, end_time_str).map(lambda x: x.strftime("%Y%m%d")).tolist()
        if not dates:
            raise ValueError("time_str > end_time_str, exit!")
        filenames = []
        for date_ in dates:
            success = f"{data_path}/{date_}/_SUCCESS"
            for i in range(int(1e20)):
                if os.path.exists(success):
                    break
                if i % 1000 == 0:
                    logger.info(f"success file not exits: {success}")
                time.sleep(30)
            file_pattern = f"{data_path}/{parse(date_).strftime('%Y%m%d/part-*.gz')}"
            filenames.extend(glob.glob(file_pattern))
        counter = Counter()
        counter.update([fpath.split("/")[-2] for fpath in filenames])
        logger.info(f"train_date={dict(sorted(counter.items(), key=lambda x: x[0]))}")
        filenames = sorted(filenames, key=hash)
    else:
        with open(FLAGS.file_list) as f:
            filenames = [l.strip("\n") for l in f]

    logger.info(f"len(filenames)={len(filenames)}, filenames: {filenames[:10]}")
    if FLAGS.mode == "export":
        params["ps_num"] = ps_num
        params["device"] = "CPU"
        setattr(TrainConfig, "device", "CPU")
        setattr(TrainConfig, "gpu_list", "")
        train([], params, model_config, strategy=None)
    elif FLAGS.mode == "train":
        params["restrict"] = False
        train(filenames, params, model_config, strategy=strategy)
        write_donefile(FLAGS.end_time_str, f"{model_dir}/logs/donefile.{task_idx}")
    elif FLAGS.mode == "feature_eval":
        train(filenames, params, model_config, strategy=strategy)
        with open(TrainConfig.slot_path) as rf:
            slots = [re.split(" +", l)[0] for l in rf if not (l.startswith("#") or l.startswith("label"))]
        for slot in slots:
            FLAGS.slot = params["slot"] = slot
            train(filenames, params, model_config, strategy=strategy)
    else:
        train(filenames, params, model_config, strategy=strategy)

    msg = f"mode: {FLAGS.mode} device: {getattr(TrainConfig, 'device', 'CPU')} task_type: {task_type} task_idx: {task_idx} time_str: {FLAGS.time_str} end_time_str: {FLAGS.end_time_str} waste: {(time.time() - t0) / 60:.2f} mins"
    logger.info(msg)


if __name__ == "__main__":
    task_type = tf_config.get('task', {}).get('type', "chief")
    task_idx = task_index = tf_config.get('task', {}).get('index', 0)
    ps_num = len(tf_config.get('cluster', {}).get('ps', []))
    task_number = len(tf_config.get('cluster', {}).get('worker', [])) + 1
    task_idx = task_idx + 1 if task_type == 'worker' else task_idx
    app.run(main)
